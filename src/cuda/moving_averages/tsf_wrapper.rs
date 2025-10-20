//! CUDA scaffolding for the Time Series Forecast (TSF) indicator.
//!
//! Mirrors LINREG’s CUDA path with ALMA-style wrapper behavior:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/tsf_kernel.ptx"))
//! - Stream NON_BLOCKING
//! - Simple 1D policies for batch and many-series
//! - VRAM checks + ~64MB headroom
//! - Warmup/NaN semantics identical to scalar TSF (period >= 2)

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::tsf::{TsfBatchRange, TsfParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaTsfPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaTsfPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Debug)]
pub enum CudaTsfError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaTsfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTsfError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTsfError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaTsfError {}

pub struct CudaTsf {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaTsfPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: i32,
}

impl CudaTsf {
    pub fn new(device_id: usize) -> Result<Self, CudaTsfError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaTsfError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/tsf_kernel.ptx"));
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
            ],
        ) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaTsfError::Cuda(e.to_string()))?
                }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaTsfPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaTsfPolicy) -> Result<Self, CudaTsfError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    pub fn synchronize(&self) -> Result<(), CudaTsfError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] TSF batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTsf)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] TSF many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTsf)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------------- VRAM helpers ----------------
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        cust::memory::mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn grid_1d_for(&self, n: usize, block_x: u32) -> GridSize {
        let bx = block_x.max(64).min(1024);
        let grid_x = ((n as u32) + bx - 1) / bx;
        (grid_x.max(self.sm_count as u32), 1, 1).into()
    }

    // --------------- Inputs (one-series × many-params) ---------------
    #[allow(clippy::type_complexity)]
    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &TsfBatchRange,
    ) -> Result<
        (
            Vec<TsfParams>,
            usize,
            usize,
            Vec<i32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
        ),
        CudaTsfError,
    > {
        if data_f32.is_empty() {
            return Err(CudaTsfError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTsfError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_local(sweep);
        if combos.is_empty() {
            return Err(CudaTsfError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut x_sums = Vec::with_capacity(combos.len());
        let mut denom_invs = Vec::with_capacity(combos.len());
        let mut inv_periods = Vec::with_capacity(combos.len());

        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaTsfError::InvalidInput("period must be >= 2".into()));
            }
            if period > len {
                return Err(CudaTsfError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaTsfError::InvalidInput(format!(
                    "not enough valid data for period {} (tail = {})",
                    period,
                    len - first_valid
                )));
            }

            let pf = period as f64;
            // Use x = 1..p convention to match kernels
            let x_sum = pf * (pf + 1.0) * 0.5;
            let x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
            let denom = pf * x2_sum - x_sum * x_sum;
            periods_i32.push(period as i32);
            x_sums.push(x_sum as f32);
            denom_invs.push((1.0 / denom) as f32);
            inv_periods.push((1.0 / pf) as f32);
        }

        Ok((
            combos,
            first_valid,
            len,
            periods_i32,
            x_sums,
            denom_invs,
            inv_periods,
        ))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_x_sums: &DeviceBuffer<f32>,
        d_denom_invs: &DeviceBuffer<f32>,
        d_inv_periods: &DeviceBuffer<f32>,
        series_len: usize,
        combos_len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsfError> {
        let func = self
            .module
            .get_function("tsf_batch_f32")
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid: GridSize = self.grid_1d_for(combos_len, block_x);
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            (*(self as *const _ as *mut CudaTsf)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut x_sums_ptr = d_x_sums.as_device_ptr().as_raw();
            let mut denom_ptr = d_denom_invs.as_device_ptr().as_raw();
            let mut inv_p_ptr = d_inv_periods.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut combos_i = combos_len as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut x_sums_ptr as *mut _ as *mut c_void,
                &mut denom_ptr as *mut _ as *mut c_void,
                &mut inv_p_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[TsfParams],
        first_valid: usize,
        series_len: usize,
        periods_i32: &[i32],
        x_sums: &[f32],
        denom_invs: &[f32],
        inv_periods: &[f32],
    ) -> Result<DeviceArrayF32, CudaTsfError> {
        // VRAM estimate
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let params_bytes =
            combos.len() * (std::mem::size_of::<i32>() + 3 * std::mem::size_of::<f32>());
        let out_bytes = combos.len() * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaTsfError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let d_periods =
            DeviceBuffer::from_slice(periods_i32).map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let d_x_sums =
            DeviceBuffer::from_slice(x_sums).map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let d_denoms =
            DeviceBuffer::from_slice(denom_invs).map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let d_inv_p =
            DeviceBuffer::from_slice(inv_periods).map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(combos.len() * series_len) }
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_x_sums,
            &d_denoms,
            &d_inv_p,
            series_len,
            combos.len(),
            first_valid,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: series_len,
        })
    }

    pub fn tsf_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &TsfBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<TsfParams>), CudaTsfError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs, inv_periods) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let arr = self.run_batch_kernel(
            data_f32,
            &combos,
            first_valid,
            len,
            &periods_i32,
            &x_sums,
            &denom_invs,
            &inv_periods,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        Ok((arr, combos))
    }

    pub fn tsf_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &TsfBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TsfParams>), CudaTsfError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs, inv_periods) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        if out.len() != combos.len() * len {
            return Err(CudaTsfError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                combos.len() * len
            )));
        }
        let dev = self.run_batch_kernel(
            data_f32,
            &combos,
            first_valid,
            len,
            &periods_i32,
            &x_sums,
            &denom_invs,
            &inv_periods,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        Ok((dev.rows, dev.cols, combos))
    }

    // --------------- Many-series (time-major) ---------------
    #[allow(clippy::type_complexity)]
    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TsfParams,
    ) -> Result<(Vec<i32>, usize, f32, f32, f32), CudaTsfError> {
        if data_tm_f32.len() != cols * rows {
            return Err(CudaTsfError::InvalidInput("data_tm size mismatch".into()));
        }
        let period = params.period.unwrap_or(0);
        if period < 2 {
            return Err(CudaTsfError::InvalidInput("period must be >= 2".into()));
        }
        if period > rows {
            return Err(CudaTsfError::InvalidInput(
                "period exceeds series length".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaTsfError::InvalidInput(format!("series {} all NaN", s)))?;
            if (rows as i32 - fv) < period as i32 {
                return Err(CudaTsfError::InvalidInput(format!(
                    "series {} not enough valid data (needed {}, have {})",
                    s,
                    period,
                    rows as i32 - fv
                )));
            }
            first_valids[s] = fv;
        }

        let pf = period as f64;
        let x_sum = (pf * (pf + 1.0) * 0.5) as f32;
        let x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
        let denom_inv = (1.0 / (pf * x2_sum - (pf * (pf + 1.0) * 0.5).powi(2))) as f32;
        let inv_period = (1.0 / pf) as f32;
        Ok((first_valids, period, x_sum, denom_inv, inv_period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        x_sum: f32,
        denom_inv: f32,
        inv_period: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsfError> {
        let func = self
            .module
            .get_function("tsf_many_series_one_param_f32")
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64).min(1024),
        };
        let grid: GridSize = self.grid_1d_for(cols, block_x);
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaTsf)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut x_sum_f = x_sum;
            let mut denom_inv_f = denom_inv;
            let mut inv_period_f = inv_period;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut x_sum_f as *mut _ as *mut c_void,
                &mut denom_inv_f as *mut _ as *mut c_void,
                &mut inv_period_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
        x_sum: f32,
        denom_inv: f32,
        inv_period: f32,
    ) -> Result<DeviceArrayF32, CudaTsfError> {
        let elems = cols * rows;
        let prices_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let required = prices_bytes + first_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaTsfError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }
        let d_prices =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices, &d_first, cols, rows, period, x_sum, denom_inv, inv_period, &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn tsf_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TsfParams,
    ) -> Result<DeviceArrayF32, CudaTsfError> {
        let (first_valids, period, x_sum, denom_inv, inv_period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            x_sum,
            denom_inv,
            inv_period,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        Ok(dev)
    }

    pub fn tsf_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TsfParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTsfError> {
        if out_tm.len() != cols * rows {
            return Err(CudaTsfError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period, x_sum, denom_inv, inv_period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            x_sum,
            denom_inv,
            inv_period,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))?;
        dev.buf
            .copy_to(out_tm)
            .map_err(|e| CudaTsfError::Cuda(e.to_string()))
    }
}

// ---------------- Benches registration ----------------
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        tsf_benches,
        CudaTsf,
        crate::indicators::tsf::TsfBatchRange,
        crate::indicators::tsf::TsfParams,
        tsf_batch_dev,
        tsf_multi_series_one_param_time_major_dev,
        crate::indicators::tsf::TsfBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::tsf::TsfParams { period: Some(64) },
        "tsf",
        "tsf"
    );
    pub use tsf_benches::bench_profiles;
}

// ---------------- Local helpers ----------------
#[inline]
fn expand_grid_local(r: &TsfBatchRange) -> Vec<TsfParams> {
    let (start, end, step) = r.period;
    if step == 0 {
        return vec![TsfParams {
            period: Some(start),
        }];
    }
    if start > end {
        return Vec::new();
    }
    let mut v = Vec::new();
    let mut p = start;
    while p <= end {
        v.push(TsfParams { period: Some(p) });
        match p.checked_add(step) {
            Some(nxt) => p = nxt,
            None => break,
        }
    }
    v
}
