//! CUDA scaffolding for the TEMA (Triple Exponential Moving Average) kernels.
//!
//! ALMA-parity wrapper surface:
//! - PTX loaded with DetermineTargetFromContext and OptLevel O2 (with fallbacks)
//! - NON_BLOCKING stream, VRAM estimates via mem_get_info, ~64MB headroom
//! - Policy enums and one-shot BENCH_DEBUG logging of selected kernels
//! - Batch (one series × many params) and time-major many-series × one param
//! - Chunk extremely large grids to avoid launch dimension limits
//!
//! Math category: recurrence/IIR (no shared host precomputation). The kernels
//! execute the triple-EMA recurrence per row/series entirely on device while the
//! price series remains resident in VRAM.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::tema::{TemaBatchRange, TemaParams};
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

// ---------------- Kernel policy & selection (mirrors ALMA structure) ----------------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    // Only one plain path today; preserved for parity and tests
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    // Only one 1D path today; preserved for parity and tests
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaTemaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaTemaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

// Introspection (selected at last launch)
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Debug)]
pub enum CudaTemaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaTemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTemaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTemaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaTemaError {}

pub struct CudaTema {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaTemaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaTema {
    pub fn new(device_id: usize) -> Result<Self, CudaTemaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTemaError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaTemaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaTemaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/tema_kernel.ptx"));
        // JIT to current context target with moderate optimization; fall back conservatively
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaTemaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaTemaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create with explicit policy (useful for tests/benches)
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaTemaPolicy,
    ) -> Result<Self, CudaTemaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaTemaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaTemaPolicy {
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
                    eprintln!("[DEBUG] TEMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTema)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] TEMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTema)).debug_many_logged = true;
                }
            }
        }
    }

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

    #[inline]
    fn grid_x_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535;
        (0..n).step_by(MAX).map(move |start| {
            let len = (n - start).min(MAX);
            (start, len)
        })
    }

    pub fn tema_batch_dev(
        &self,
        prices: &[f32],
        sweep: &TemaBatchRange,
    ) -> Result<DeviceArrayF32, CudaTemaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn tema_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &TemaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TemaParams>), CudaTemaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaTemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    pub fn tema_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTemaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaTemaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaTemaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn tema_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTemaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaTemaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaTemaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            period,
            num_series,
            series_len,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn tema_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaTemaError> {
        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)
    }

    pub fn tema_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTemaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaTemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }

        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaTemaError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaTemaError::InvalidInput(
                "insufficient device memory for TEMA batch launch".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaTemaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&inputs.periods)
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }
                .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;

        // Chunk very large grids to avoid legacy launch limits
        for (start, len) in Self::grid_x_chunks(n_combos) {
            let periods_ptr_raw =
                d_periods.as_device_ptr().as_raw() + (start * core::mem::size_of::<i32>()) as u64;
            let out_ptr_raw = d_out.as_device_ptr().as_raw()
                + (start * series_len * core::mem::size_of::<f32>()) as u64;

            self.launch_batch_kernel_chunk(
                &d_prices,
                periods_ptr_raw,
                series_len,
                len,
                inputs.first_valid,
                out_ptr_raw,
            )?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        prepared: &ManySeriesInputs,
    ) -> Result<DeviceArrayF32, CudaTemaError> {
        let num_series = cols;
        let series_len = rows;

        let prices_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let first_valid_bytes = prepared.first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; // 32MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaTemaError::InvalidInput(
                "insufficient device memory for TEMA many-series launch".into(),
            ));
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32)
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prices_tm_f32.len()) }
                .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            num_series,
            series_len,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows: series_len,
            cols: num_series,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTemaError> {
        self.launch_batch_kernel_chunk(
            d_prices,
            d_periods.as_device_ptr().as_raw(),
            series_len,
            n_combos,
            first_valid,
            d_out.as_device_ptr().as_raw(),
        )
    }

    fn launch_batch_kernel_chunk(
        &self,
        d_prices: &DeviceBuffer<f32>,
        periods_ptr_raw: u64,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        out_ptr_raw: u64,
    ) -> Result<(), CudaTemaError> {
        let func = self
            .module
            .get_function("tema_batch_f32")
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;

        // Policy: currently only Plain; record selection for debug parity
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto | BatchKernelPolicy::Plain { .. } => 1u32,
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaTema)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = periods_ptr_raw;
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = out_ptr_raw;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTemaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTemaError> {
        let func = self
            .module
            .get_function("tema_multi_series_one_param_f32")
            .map_err(|e| CudaTemaError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto | ManySeriesKernelPolicy::OneD { .. } => 1u32,
        };
        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaTema)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTemaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &TemaBatchRange,
    ) -> Result<BatchInputs, CudaTemaError> {
        if prices.is_empty() {
            return Err(CudaTemaError::InvalidInput("empty prices".into()));
        }

        let combos = expand_grid_tema(sweep);
        if combos.is_empty() {
            return Err(CudaTemaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTemaError::InvalidInput("all values are NaN".into()))?;

        let series_len = prices.len();
        let mut periods = Vec::with_capacity(combos.len());
        let mut max_period = 0usize;
        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaTemaError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            if period > i32::MAX as usize {
                return Err(CudaTemaError::InvalidInput(
                    "period exceeds i32 kernel limit".into(),
                ));
            }
            periods.push(period as i32);
            max_period = max_period.max(period);
        }

        if series_len - first_valid < max_period {
            return Err(CudaTemaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }

        Ok(BatchInputs {
            combos,
            periods,
            first_valid,
            series_len,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<ManySeriesInputs, CudaTemaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTemaError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaTemaError::InvalidInput("matrix shape mismatch".into()));
        }
        if period == 0 {
            return Err(CudaTemaError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        if period > i32::MAX as usize {
            return Err(CudaTemaError::InvalidInput(
                "period exceeds i32 kernel limit".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series_idx in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series_idx;
                let price = prices_tm_f32[idx];
                if !price.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let first = fv.ok_or_else(|| {
                CudaTemaError::InvalidInput(format!("series {} has all NaN values", series_idx))
            })?;
            if rows - first < period {
                return Err(CudaTemaError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series_idx,
                    period,
                    rows - first
                )));
            }
            first_valids[series_idx] = first as i32;
        }

        Ok(ManySeriesInputs { first_valids })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches_batch_only;

    define_ma_period_benches_batch_only!(
        tema_benches,
        CudaTema,
        crate::indicators::moving_averages::tema::TemaBatchRange,
        tema_batch_dev,
        crate::indicators::moving_averages::tema::TemaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        "tema",
        "tema"
    );
    pub use tema_benches::bench_profiles;
}

struct BatchInputs {
    combos: Vec<TemaParams>,
    periods: Vec<i32>,
    first_valid: usize,
    series_len: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
}

fn expand_grid_tema(range: &TemaBatchRange) -> Vec<TemaParams> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![TemaParams {
            period: Some(start),
        }];
    }
    (start..=end)
        .step_by(step)
        .map(|p| TemaParams { period: Some(p) })
        .collect()
}
