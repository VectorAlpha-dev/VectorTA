//! CUDA scaffolding for the TrendFlex filter.
//!
//! Aligns with ALMA’s wrapper policy: VRAM-first design, explicit kernel
//! policies, JIT load options, VRAM checks, NON_BLOCKING stream, and optional
//! debug logging of selected kernels. Batch covers one-series × many-params;
//! many-series covers time-major many-series × one-param.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::trendflex::{
    expand_grid_trendflex, TrendFlexBatchRange, TrendFlexParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaTrendflexError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaTrendflexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTrendflexError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTrendflexError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaTrendflexError {}

pub struct CudaTrendflex {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaTrendflexPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

// -------- Kernel selection policy (mirrors ALMA shape; TrendFlex only needs 1D variants) --------

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
pub struct CudaTrendflexPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaTrendflexPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

// -------- Introspection (selected kernel) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

impl CudaTrendflex {
    pub fn new(device_id: usize) -> Result<Self, CudaTrendflexError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/trendflex_kernel.ptx"));
        // Prefer context-targeted JIT with O2 like ALMA/CWMA; fall back gracefully.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[])
                        .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaTrendflexPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaTrendflexPolicy,
    ) -> Result<Self, CudaTrendflexError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaTrendflexPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaTrendflexPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
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
                    eprintln!("[DEBUG] TrendFlex batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTrendflex)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] TrendFlex many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTrendflex)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------- Utilities ----------

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &TrendFlexBatchRange,
    ) -> Result<(Vec<TrendFlexParams>, usize, usize), CudaTrendflexError> {
        if data_f32.is_empty() {
            return Err(CudaTrendflexError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTrendflexError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_trendflex(sweep);
        if combos.is_empty() {
            return Err(CudaTrendflexError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let tail_len = len - first_valid;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaTrendflexError::InvalidInput(
                    "period must be at least 1".into(),
                ));
            }
            if period >= len {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            let ss_period = ((period as f64) / 2.0).round() as usize;
            if tail_len < period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "not enough valid data for period {} (valid tail = {})",
                    period, tail_len
                )));
            }
            if tail_len < ss_period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "not enough valid data for smoother period {} (valid tail = {})",
                    ss_period, tail_len
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_ssf: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrendflexError> {
        let func = self
            .module
            .get_function("trendflex_batch_f32")
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 128,
        };
        // Chunk by combos to keep grid.x blocks under 65_535
        let max_blocks: u32 = 65_535;
        let chunk_cap: usize = (max_blocks as usize) * (block_x as usize);
        let mut launched = 0usize;
        while launched < n_combos {
            let chunk = (n_combos - launched).min(chunk_cap);
            let grid_x = ((chunk as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(launched).as_raw();
                let mut len_i = series_len as i32;
                let mut combos_i = chunk as i32;
                let mut first_valid_i = first_valid as i32;
                let mut ssf_ptr = d_ssf.as_device_ptr().add(launched * series_len).as_raw();
                let mut out_ptr = d_out.as_device_ptr().add(launched * series_len).as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut ssf_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
            }
            launched += chunk;
        }
        // Introspection + optional debug log
        unsafe {
            let this = self as *const _ as *mut CudaTrendflex;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[TrendFlexParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        // VRAM estimate (prices + periods + scratch + out)
        let n_combos = combos.len();
        let prices_bytes = len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let scratch_bytes = n_combos * len * std::mem::size_of::<f32>(); // ssf
        let out_bytes = n_combos * len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + scratch_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(required, headroom) {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        let elems = combos.len() * len;
        let mut d_ssf = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            len,
            combos.len(),
            first_valid,
            &mut d_ssf,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn trendflex_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &TrendFlexBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<TrendFlexParams>), CudaTrendflexError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn trendflex_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &TrendFlexBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TrendFlexParams>), CudaTrendflexError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "output slice length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        // For host copies, we still run a single kernel launch set. Use a pinned buffer for D2H.
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(dev.len())
                .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?
        };
        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        out.copy_from_slice(pinned.as_slice());
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrendFlexParams,
    ) -> Result<(Vec<i32>, usize), CudaTrendflexError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }
        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period >= rows {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        let ss_period = ((period as f64) / 2.0).round() as usize;
        if ss_period == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "smoother period must be positive".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let val = data_tm_f32[idx];
                if !val.is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaTrendflexError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            let tail = rows - fv;
            if tail < period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    series, period, tail
                )));
            }
            if tail < ss_period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "series {} insufficient data for smoother {} (tail = {})",
                    series, ss_period, tail
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_ssf: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrendflexError> {
        let func = self
            .module
            .get_function("trendflex_many_series_one_param_f32")
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => 128,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut ssf_ptr = d_ssf.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut ssf_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        }
        // Introspection + optional debug log
        unsafe {
            let this = self as *const _ as *mut CudaTrendflex;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        // VRAM estimate (prices + first_valids + scratch + out)
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let firsts_bytes = cols * std::mem::size_of::<i32>();
        let scratch_bytes = cols * rows * std::mem::size_of::<f32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + firsts_bytes + scratch_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let elems = cols * rows;
        let mut d_ssf = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            cols,
            rows,
            period,
            &mut d_ssf,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn trendflex_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrendFlexParams,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn trendflex_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrendFlexParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTrendflexError> {
        if out_tm.len() != cols * rows {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "output slice mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(dev.len())
                .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?
        };
        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        trendflex_benches,
        CudaTrendflex,
        crate::indicators::moving_averages::trendflex::TrendFlexBatchRange,
        crate::indicators::moving_averages::trendflex::TrendFlexParams,
        trendflex_batch_dev,
        trendflex_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::trendflex::TrendFlexBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::trendflex::TrendFlexParams { period: Some(64) },
        "trendflex",
        "trendflex"
    );
    pub use trendflex_benches::bench_profiles;
}
