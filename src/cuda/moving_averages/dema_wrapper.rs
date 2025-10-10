//! CUDA support for the DEMA (Double Exponential Moving Average) indicator.
//!
//! Mirrors the ALMA "gold standard" wrapper: VRAM-first design, explicit
//! kernel policies (with sensible Auto defaults), device-memory checks, and
//! deterministic benches. DEMA is a recursive filter, so kernels parallelize
//! across parameter combos and series while each thread walks time sequentially
//! for its assigned work item. All variants match the scalar warm-up semantics:
//! indices t < (first_valid + period - 1) are set to NaN.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::dema::{DemaBatchRange, DemaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaDemaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaDemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaDemaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaDemaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaDemaError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput { One, Two }

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    // DEMA does not benefit from tiled dot-product kernels; keep variant for API parity
    Tiled { tile: u32, per_thread: BatchThreadsPerOutput },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    // Kept for parity with ALMA; falls back to OneD for DEMA
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDemaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaDemaPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaDema {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaDemaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDema {
    pub fn new(device_id: usize) -> Result<Self, CudaDemaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/dema_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaDemaError::Cuda(e.to_string()))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaDemaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaDemaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))
    }

    pub fn new_with_policy(device_id: usize, policy: CudaDemaPolicy) -> Result<Self, CudaDemaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaDemaPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaDemaPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DEMA batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDema)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] DEMA many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaDema)).debug_many_logged = true; }
            }
        }
    }

    pub fn dema_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &DemaBatchRange,
    ) -> Result<DeviceArrayF32, CudaDemaError> {
        let (combos, first_valid, series_len, _max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let periods: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();

        // VRAM estimate and guard (add ~64MB headroom like ALMA)
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = periods.len() * std::mem::size_of::<i32>();
        let out_bytes = series_len * periods.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaDemaError::InvalidInput(
                "insufficient device memory for DEMA batch".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len * combos.len())
                .map_err(|e| CudaDemaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            series_len,
            first_valid,
            combos.len(),
            &mut d_out,
        )?;

        self.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: series_len,
        })
    }

    pub fn dema_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: i32,
        first_valid: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDemaError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaDemaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len as usize,
            first_valid.max(0) as usize,
            n_combos as usize,
            d_out,
        )?;
        self.synchronize()
    }

    /// Convenience: run DEMA batch using device-resident prices. Builds the
    /// period vector on the device from the provided sweep and returns a VRAM
    /// handle for the output matrix (rows = combos, cols = series_len).
    pub fn dema_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: &DemaBatchRange,
    ) -> Result<DeviceArrayF32, CudaDemaError> {
        if series_len == 0 { return Err(CudaDemaError::InvalidInput("series_len is zero".into())); }
        let combos = expand_periods(sweep);
        if combos.is_empty() { return Err(CudaDemaError::InvalidInput("no period combinations provided".into())); }
        let periods: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();
        let max_period = combos.iter().map(|p| p.period.unwrap_or(0)).max().unwrap_or(0) as usize;
        if max_period == 0 || series_len.saturating_sub(first_valid) < max_period {
            return Err(CudaDemaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len.saturating_sub(first_valid)
            )));
        }

        // VRAM guard (prices already on device, so only periods + output + headroom)
        let periods_bytes = periods.len() * std::mem::size_of::<i32>();
        let out_bytes = series_len * periods.len() * std::mem::size_of::<f32>();
        let required = periods_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaDemaError::InvalidInput("insufficient device memory for DEMA batch".into()));
        }

        let d_periods = DeviceBuffer::from_slice(&periods)
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len * combos.len())
                .map_err(|e| CudaDemaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            d_prices,
            &d_periods,
            series_len,
            first_valid,
            combos.len(),
            &mut d_out,
        )?;
        self.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: series_len })
    }

    pub fn dema_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &DemaBatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaDemaError> {
        let (combos, _first_valid, series_len, _max_p) = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != combos.len() * series_len {
            return Err(CudaDemaError::InvalidInput("output slice length mismatch".into()));
        }
        let handle = self.dema_batch_dev(data_f32, sweep)?;
        handle
            .buf
            .copy_to(out_flat)
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &DemaBatchRange,
    ) -> Result<(Vec<DemaParams>, usize, usize, usize), CudaDemaError> {
        if data_f32.is_empty() {
            return Err(CudaDemaError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaDemaError::InvalidInput(
                "no period combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaDemaError::InvalidInput("all values are NaN".into()))?;

        let max_period = combos
            .iter()
            .map(|p| p.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 {
            return Err(CudaDemaError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        let needed = 2 * (max_period - 1);
        if series_len < needed {
            return Err(CudaDemaError::InvalidInput(format!(
                "not enough data: needed >= {}, have {}",
                needed, series_len
            )));
        }
        let valid = series_len - first_valid;
        if valid < needed {
            return Err(CudaDemaError::InvalidInput(format!(
                "not enough valid data: needed >= {}, have {}",
                needed, valid
            )));
        }

        Ok((combos, first_valid, series_len, max_period))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDemaError> {
        if n_combos == 0 { return Ok(()); }

        // Prefill outputs with canonical qNaN on this stream
        memset_f32_qnan_async(&self.stream, d_out)?;

        let func = self
            .module
            .get_function("dema_batch_f32")
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;

        // Single-threaded sequential recurrence per combo
        let mut block_x: u32 = 1;
        if let BatchKernelPolicy::Plain { block_x: bx } = self.policy.batch { block_x = bx.max(1); }
        unsafe {
            let this = self as *const _ as *mut CudaDema;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        }

        // No implicit sync; public API decides synchronization
        Ok(())
    }

    // ---------- many-series (time-major) ----------
    pub fn dema_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &DemaParams,
    ) -> Result<DeviceArrayF32, CudaDemaError> {
        let (first_valids, period) = Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        // VRAM estimate and guard (~64MB headroom like ALMA)
        let elems = num_series * series_len;
        let required = elems * 2 * std::mem::size_of::<f32>() + num_series * std::mem::size_of::<i32>();
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaDemaError::InvalidInput(
                "insufficient device memory for DEMA many-series".into(),
            ));
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(elems)
                .map_err(|e| CudaDemaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_first_valids,
            period as i32,
            num_series,
            series_len,
            &mut d_out_tm,
        )?;

        self.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out_tm, rows: series_len, cols: num_series })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dema_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDemaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaDemaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if period <= 0 { return Err(CudaDemaError::InvalidInput("period must be positive".into())); }
        if d_prices_tm.len() != num_series * series_len || d_out_tm.len() != num_series * series_len {
            return Err(CudaDemaError::InvalidInput("time-major buffer length mismatch".into()));
        }
        if d_first_valids.len() != num_series { return Err(CudaDemaError::InvalidInput("first_valids length mismatch".into())); }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period,
            num_series,
            series_len,
            d_out_tm,
        )?;
        self.synchronize()
    }

    pub fn dema_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &DemaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaDemaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaDemaError::InvalidInput("output slice length mismatch".into()));
        }
        let handle = self.dema_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle
            .buf
            .copy_to(out_tm)
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDemaError> {
        let func = self
            .module
            .get_function("dema_many_series_one_param_time_major_f32")
            .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;

        // Prefill output with qNaN (entire time-major buffer)
        memset_f32_qnan_async(&self.stream, d_out_tm)?;

        // Warp-mapped launch config
        let mut block_x_req: u32 = 128; // default 4 warps per block
        if let ManySeriesKernelPolicy::OneD { block_x: bx } = self.policy.many_series {
            block_x_req = bx.max(32);
        }
        let warps_per_block = (block_x_req / 32).max(1);
        let block_x = warps_per_block * 32;
        let total_warps = ((num_series as u32) + 31) / 32;
        let grid_x = (total_warps + warps_per_block - 1) / warps_per_block;

        unsafe {
            let this = self as *const _ as *mut CudaDema;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaDemaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    // ---------- helpers ----------
    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        mem_get_info()
            .map(|(free, _)| required_bytes.saturating_add(headroom_bytes) <= free)
            .unwrap_or(true)
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &DemaParams,
    ) -> Result<(Vec<i32>, usize), CudaDemaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaDemaError::InvalidInput("num_series and series_len must be positive".into()));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaDemaError::InvalidInput("time-major slice length mismatch".into()));
        }
        let period = params.period.unwrap_or(0);
        if period == 0 { return Err(CudaDemaError::InvalidInput("period must be positive".into())); }

        let mut first_valids = Vec::with_capacity(num_series);
        let needed = 2usize.saturating_mul(period.saturating_sub(1));
        for s in 0..num_series {
            let mut found = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + s];
                if v.is_finite() { found = Some(t as i32); break; }
            }
            let fv = found.ok_or_else(|| CudaDemaError::InvalidInput(format!("series {} contains only NaNs", s)))?;
            let remaining = series_len - fv as usize;
            // Match CPU acceptance: require >= 2*(period - 1) valid samples
            if remaining < needed {
                return Err(CudaDemaError::InvalidInput(format!(
                    "series {} does not have enough valid data: need >= {}, have {}",
                    s, needed, remaining
                )));
            }
            first_valids.push(fv);
        }
        Ok((first_valids, period))
    }
}

// --- utility: async memset to canonical quiet-NaN (0x7FC0_0000) ---
#[inline]
fn memset_f32_qnan_async(stream: &Stream, buf: &mut DeviceBuffer<f32>) -> Result<(), CudaDemaError> {
    const QNAN_BITS: u32 = 0x7FC0_0000;
    unsafe {
        let ptr: cu::CUdeviceptr = buf.as_device_ptr().as_raw();
        let n: usize = buf.len();
        let st: cu::CUstream = stream.as_inner();
        let res = cu::cuMemsetD32Async(ptr, QNAN_BITS, n, st);
        match res {
            cu::CUresult::CUDA_SUCCESS => Ok(()),
            e => Err(CudaDemaError::Cuda(format!("cuMemsetD32Async failed: {:?}", e))),
        }
    }
}

fn expand_periods(range: &DemaBatchRange) -> Vec<DemaParams> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![DemaParams {
            period: Some(start),
        }];
    }

    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(DemaParams {
            period: Some(value),
        });
        value = value.saturating_add(step);
    }
    out
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        dema_benches,
        CudaDema,
        crate::indicators::moving_averages::dema::DemaBatchRange,
        crate::indicators::moving_averages::dema::DemaParams,
        dema_batch_dev,
        dema_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::dema::DemaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::dema::DemaParams { period: Some(64) },
        "dema",
        "dema"
    );
    pub use dema_benches::bench_profiles;
}
