//! CUDA scaffolding for the Normalized Moving Average (NMA).
//!
//! Brought up to the ALMA wrapper standard:
//! - PTX is JITed with DetermineTargetFromContext and O2, with graceful
//!   fallbacks for stability across driver/toolkit versions.
//! - Policy enums for explicit kernel selection (kept simple — NMA only has
//!   plain kernels today). Selected kernels are recorded and optionally logged
//!   once per instance when `BENCH_DEBUG=1`.
//! - VRAM usage is estimated and validated with a small headroom; grid.y is
//!   chunked to <= 65_535 combos for large sweeps.
//! - Public API mirrors ALMA: one-series × many-params and many-series × one
//!   param, returning DeviceArrayF32 VRAM handles.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::nma::{NmaBatchRange, NmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// ---- Kernel selection policy (kept intentionally minimal for NMA) ----

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
pub struct CudaNmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaNmaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

// Introspection of which kernel got picked (for BENCH_DEBUG=1)
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Debug)]
pub enum CudaNmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaNmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaNmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaNmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaNmaError {}

pub struct CudaNma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaNmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaNma {
    /// Create a new CUDA NMA wrapper and load PTX with robust JIT options.
    pub fn new(device_id: usize) -> Result<Self, CudaNmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaNmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/nma_kernel.ptx"));
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaNmaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaNmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create with an explicit kernel policy (useful for tests/benches).
    pub fn new_with_policy(device_id: usize, policy: CudaNmaPolicy) -> Result<Self, CudaNmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaNmaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scn =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scn || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] NMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNma)).debug_batch_logged = true;
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
                let per_scn =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scn || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] NMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNma)).debug_many_logged = true;
                }
            }
        }
    }

    // ---- VRAM checks + launch chunking ----
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _total)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }
    #[inline]
    fn grid_y_chunks(n_combos: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535;
        (0..n_combos).step_by(MAX).map(move |start| {
            let len = (n_combos - start).min(MAX);
            (start, len)
        })
    }

    fn expand_range(range: &NmaBatchRange) -> Vec<NmaParams> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![NmaParams {
                period: Some(start),
            }];
        }
        (start..=end)
            .step_by(step)
            .map(|period| NmaParams {
                period: Some(period),
            })
            .collect()
    }

    fn compute_abs_diffs(data: &[f32]) -> Vec<f32> {
        let len = data.len();
        let mut ln_values = vec![0f32; len];
        for (idx, value) in data.iter().enumerate() {
            let safe = value.max(1e-10f32);
            ln_values[idx] = safe.ln();
        }
        let mut diffs = vec![0f32; len];
        for i in 1..len {
            diffs[i] = (ln_values[i] - ln_values[i - 1]).abs();
        }
        diffs
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &NmaBatchRange,
    ) -> Result<(Vec<NmaParams>, usize, usize, usize, Vec<f32>), CudaNmaError> {
        if data_f32.is_empty() {
            return Err(CudaNmaError::InvalidInput("empty data".into()));
        }

        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaNmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaNmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut max_period = 0usize;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period == 0 || period > len {
                return Err(CudaNmaError::InvalidInput(format!(
                    "invalid period {} for data length {}",
                    period, len
                )));
            }
            if len - first_valid < period + 1 {
                return Err(CudaNmaError::InvalidInput(format!(
                    "not enough valid data for period {} (have {})",
                    period,
                    len - first_valid
                )));
            }
            max_period = max_period.max(period);
        }

        let abs_diffs = Self::compute_abs_diffs(data_f32);
        Ok((combos, first_valid, len, max_period, abs_diffs))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_abs_diffs: &DeviceBuffer<f32>,
        periods_ptr: cust::memory::DevicePointer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        out_ptr: cust::memory::DevicePointer<f32>,
    ) -> Result<(), CudaNmaError> {
        let func = self
            .module
            .get_function("nma_batch_f32")
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;

        // Block size selection (simple, occupancy is adequate here)
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared = (max_period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut diffs_ptr = d_abs_diffs.as_device_ptr().as_raw();
            let mut periods_ptr = periods_ptr.as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = out_ptr.as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut diffs_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, shared, args)
                .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        }

        // Introspection: record selection once per call-site
        unsafe {
            let this = self as *const _ as *mut CudaNma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[NmaParams],
        first_valid: usize,
        len: usize,
        max_period: usize,
        abs_diffs: &[f32],
    ) -> Result<DeviceArrayF32, CudaNmaError> {
        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        let d_abs_diffs =
            DeviceBuffer::from_slice(abs_diffs).map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaNmaError::Cuda(e.to_string()))?;

        // VRAM estimate and headroom (~64MB)
        let bytes_inputs = len * std::mem::size_of::<f32>();
        let bytes_diffs = len * std::mem::size_of::<f32>();
        let bytes_periods = periods.len() * std::mem::size_of::<i32>();
        let bytes_out = combos.len() * len * std::mem::size_of::<f32>();
        let required = bytes_inputs + bytes_diffs + bytes_periods + bytes_out;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaNmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let elems = combos.len() * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;

        // Chunk grid.y to respect CUDA limits
        for (start, chunk_len) in Self::grid_y_chunks(combos.len()) {
            let periods_ptr = unsafe { d_periods.as_device_ptr().add(start) };
            let out_ptr = unsafe { d_out.as_device_ptr().add(start * len) };
            self.launch_batch_kernel(
                &d_prices,
                &d_abs_diffs,
                periods_ptr,
                len,
                chunk_len,
                first_valid,
                max_period,
                out_ptr,
            )?;
        }

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn nma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &NmaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<NmaParams>), CudaNmaError> {
        let (combos, first_valid, len, max_period, abs_diffs) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev =
            self.run_batch_kernel(data_f32, &combos, first_valid, len, max_period, &abs_diffs)?;
        Ok((dev, combos))
    }

    pub fn nma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &NmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<NmaParams>), CudaNmaError> {
        let (combos, first_valid, len, max_period, abs_diffs) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaNmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }

        let dev =
            self.run_batch_kernel(data_f32, &combos, first_valid, len, max_period, &abs_diffs)?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &NmaParams,
    ) -> Result<(Vec<i32>, Vec<f32>, usize), CudaNmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNmaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaNmaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaNmaError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period >= rows {
            return Err(CudaNmaError::InvalidInput(format!(
                "period {} must be less than series length {}",
                period, rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaNmaError::InvalidInput(format!("series {} contains only NaN", series))
            })?;
            if rows - fv < period + 1 {
                return Err(CudaNmaError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    series,
                    period,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        let mut abs_diffs_tm = vec![0f32; cols * rows];
        for series in 0..cols {
            let mut prev_ln = 0f32;
            for row in 0..rows {
                let idx = row * cols + series;
                let safe = data_tm_f32[idx].max(1e-10f32);
                let ln_val = safe.ln();
                if row == 0 {
                    abs_diffs_tm[idx] = 0.0f32;
                } else {
                    abs_diffs_tm[idx] = (ln_val - prev_ln).abs();
                }
                prev_ln = ln_val;
            }
        }

        Ok((first_valids, abs_diffs_tm, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_abs_diffs_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNmaError> {
        let func = self
            .module
            .get_function("nma_many_series_one_param_f32")
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared = (period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut diffs_ptr = d_abs_diffs_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut period_i = period as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut diffs_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, shared, args)
                .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        }

        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaNma;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        abs_diffs_tm: &[f32],
        first_valids: &[i32],
        num_series: usize,
        series_len: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaNmaError> {
        let d_prices =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        let d_abs_diffs = DeviceBuffer::from_slice(abs_diffs_tm)
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;
        // VRAM estimate
        let bytes_inputs = num_series * series_len * std::mem::size_of::<f32>();
        let bytes_diffs = bytes_inputs;
        let bytes_first = num_series * std::mem::size_of::<i32>();
        let bytes_out = bytes_inputs;
        let required = bytes_inputs + bytes_diffs + bytes_first + bytes_out;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaNmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let elems = num_series * series_len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_abs_diffs,
            &d_first,
            num_series,
            series_len,
            period,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    pub fn nma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &NmaParams,
    ) -> Result<DeviceArrayF32, CudaNmaError> {
        let (first_valids, abs_diffs_tm, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(
            data_tm_f32,
            &abs_diffs_tm,
            &first_valids,
            cols,
            rows,
            period,
        )
    }

    pub fn nma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &NmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaNmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaNmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, abs_diffs_tm, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            data_tm_f32,
            &abs_diffs_tm,
            &first_valids,
            cols,
            rows,
            period,
        )?;
        dev.buf
            .copy_to(out_tm)
            .map_err(|e| CudaNmaError::Cuda(e.to_string()))
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        nma_benches,
        CudaNma,
        crate::indicators::moving_averages::nma::NmaBatchRange,
        crate::indicators::moving_averages::nma::NmaParams,
        nma_batch_dev,
        nma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::nma::NmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::nma::NmaParams { period: Some(64) },
        "nma",
        "nma"
    );
    pub use nma_benches::bench_profiles;
}
