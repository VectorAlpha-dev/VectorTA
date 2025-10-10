//! CUDA wrapper for TRIMA (Triangular Moving Average) batch kernels.
//!
//! Aligns with the ALMA wrapper conventions for API surface, policy selection,
//! PTX JIT options, VRAM checks, grid chunking, and debug logging. Kernels use
//! triangular weights (fixed per period), precomputed on-device in shared
//! memory for the batch path and passed as a vector for the many-series path.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::trima::{TrimaBatchRange, TrimaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaTrimaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaTrimaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTrimaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTrimaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaTrimaError {}

// -------- Kernel selection policy (ALMA/CWMA-style minimal surface) --------

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
pub struct CudaTrimaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaTrimaPolicy {
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
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaTrima {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaTrimaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaTrima {
    pub fn new(device_id: usize) -> Result<Self, CudaTrimaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;

        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/trima_kernel.ptx"));
        // Prefer context-targeted JIT with moderate optimization; then fall back
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaTrimaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaTrimaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Expose a simple synchronize for callers (e.g., benches) that want deterministic timing.
    pub fn synchronize(&self) -> Result<(), CudaTrimaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))
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
                    eprintln!("[DEBUG] TRIMA batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaTrima)).debug_batch_logged = true; }
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
                    eprintln!("[DEBUG] TRIMA many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaTrima)).debug_many_logged = true; }
            }
        }
    }

    // Policy controls/inspection
    pub fn set_policy(&mut self, policy: CudaTrimaPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaTrimaPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    fn expand_periods(range: &TrimaBatchRange) -> Vec<usize> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![start];
        }
        if start > end {
            return Vec::new();
        }
        (start..=end).step_by(step).collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &TrimaBatchRange,
    ) -> Result<(Vec<usize>, usize), CudaTrimaError> {
        if data_f32.is_empty() {
            return Err(CudaTrimaError::InvalidInput("empty data".into()));
        }

        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaTrimaError::InvalidInput("all values are NaN".into()))?;

        let periods = Self::expand_periods(sweep);
        if periods.is_empty() {
            return Err(CudaTrimaError::InvalidInput("no periods in sweep".into()));
        }

        let len = data_f32.len();
        for &period in &periods {
            if period <= 3 {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "period {} too small (must be > 3)",
                    period
                )));
            }
            if period > len {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
        }

        Ok((periods, first_valid))
    }

    fn compute_weights(period: usize) -> Vec<f32> {
        let mut weights = vec![0.0f32; period];
        let m1 = (period + 1) / 2;
        let m2 = period - m1 + 1;
        let norm = (m1 * m2) as f32;
        for i in 0..period {
            let w = if i < m1 {
                (i + 1) as f32
            } else if i < m2 {
                m1 as f32
            } else {
                (m1 + m2 - 1 - i) as f32
            };
            weights[i] = w / norm;
        }
        weights
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrimaParams,
    ) -> Result<(Vec<i32>, usize), CudaTrimaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTrimaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaTrimaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(30);
        if period <= 3 {
            return Err(CudaTrimaError::InvalidInput(format!(
                "period {} too small (must be > 3)",
                period
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv_row = fv.ok_or_else(|| {
                CudaTrimaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv_row < period {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv_row
                )));
            }
            first_valids[series] = fv_row as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrimaError> {
        if series_len == 0 {
            return Err(CudaTrimaError::InvalidInput("series_len is zero".into()));
        }
        if n_combos == 0 {
            return Err(CudaTrimaError::InvalidInput("no parameter combos".into()));
        }
        if max_period == 0 {
            return Err(CudaTrimaError::InvalidInput("max_period is zero".into()));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaTrimaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("trima_batch_f32")
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;

        // Policy-selected block size; default to 256 like the original
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        unsafe { (*(self as *const _ as *mut CudaTrima)).last_batch = Some(BatchKernelSelected::OneD { block_x }); }
        self.maybe_log_batch_debug();

        // Chunk grid.y to <= 65_535
        const MAX_GRID_Y: usize = 65_535;
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();

        let mut launched = 0usize;
        while launched < n_combos {
            let len = (n_combos - launched).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x.max(1), len as u32, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                // Offset periods/warms/out by launched
                let periods_ptr = d_periods.as_device_ptr().add(launched);
                let mut periods_ptr = periods_ptr.as_raw();
                let warms_ptr = d_warms.as_device_ptr().add(launched);
                let mut warms_ptr = warms_ptr.as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = len as i32;
                let mut max_period_i = max_period as i32;
                let out_ptr = d_out.as_device_ptr().add(launched * series_len);
                let mut out_ptr = out_ptr.as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;
            }

            launched += len;
        }

        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrimaError> {
        if period == 0 || cols == 0 || rows == 0 {
            return Err(CudaTrimaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize || cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaTrimaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("trima_multi_series_one_param_f32")
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 128,
        };
        unsafe { (*(self as *const _ as *mut CudaTrima)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        self.maybe_log_many_debug();

        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?
        }

        Ok(())
    }

    pub fn trima_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrimaError> {
        self.launch_kernel(
            d_prices, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn trima_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &TrimaBatchRange,
    ) -> Result<DeviceArrayF32, CudaTrimaError> {
        let (periods, first_valid) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let series_len = data_f32.len();
        let n_combos = periods.len();
        let max_period = periods.iter().copied().max().unwrap_or(0);

        // VRAM check: prices + periods + warms + out with ~64MB headroom
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let warms_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + warms_bytes + out_bytes;
        if let Ok((free, _total)) = mem_get_info() {
            let headroom = 64usize * 1024 * 1024;
            if required.saturating_add(headroom) > free {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (required as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let warms_i32: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + p - 1) as i32)
            .collect();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;
        let d_warms = DeviceBuffer::from_slice(&warms_i32)
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;

        self.launch_kernel(
            &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaTrimaError> {
        // VRAM check: prices + weights + first_valids + out with ~64MB headroom
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let weights_bytes = period * std::mem::size_of::<f32>();
        let first_valids_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + first_valids_bytes + out_bytes;
        if let Ok((free, _total)) = mem_get_info() {
            let headroom = 64usize * 1024 * 1024;
            if required.saturating_add(headroom) > free {
                return Err(CudaTrimaError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (required as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        let weights = Self::compute_weights(period);
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn trima_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrimaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaTrimaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_weights,
            d_first_valids,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn trima_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrimaParams,
    ) -> Result<DeviceArrayF32, CudaTrimaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn trima_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrimaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTrimaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaTrimaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaTrimaError::Cuda(e.to_string()))?;
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        trima_benches,
        CudaTrima,
        crate::indicators::moving_averages::trima::TrimaBatchRange,
        crate::indicators::moving_averages::trima::TrimaParams,
        trima_batch_dev,
        trima_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::trima::TrimaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::trima::TrimaParams { period: Some(64) },
        "trima",
        "trima"
    );
    pub use trima_benches::bench_profiles;
}
