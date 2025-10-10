//! CUDA wrapper for SMMA (Smoothed Moving Average) kernels.
//!
//! Mirrors the ALMA/SWMA scaffolding: validate host inputs, upload FP32 data
//! once, and launch kernels that keep the dependency-aware recursion on the
//! device. Supports both the single-series × many-parameter sweep and the
//! many-series × one-parameter time-major path.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::smma::{expand_grid, SmmaBatchRange, SmmaParams};
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
pub enum CudaSmmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSmmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSmmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSmmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaSmmaError {}

// -------- Kernel selection policy (mirrors ALMA/CWMA shape) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy { Auto, Plain { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy { Auto, OneD { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub struct CudaSmmaPolicy { pub batch: BatchKernelPolicy, pub many_series: ManySeriesKernelPolicy }
impl Default for CudaSmmaPolicy {
    fn default() -> Self { Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto } }
}

// -------- Introspection (selected kernel) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { OneD { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaSmma {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaSmmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaSmma {
    pub fn new(device_id: usize) -> Result<Self, CudaSmmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/smma_kernel.ptx"));
        // Prefer context-targeted JIT with moderate optimization, fallback progressively
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaSmmaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaSmmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SMMA batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaSmma)).debug_batch_logged = true; }
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
                    eprintln!("[DEBUG] SMMA many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaSmma)).debug_many_logged = true; }
            }
        }
    }

    pub fn set_policy(&mut self, policy: CudaSmmaPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaSmmaPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SmmaBatchRange,
    ) -> Result<(Vec<SmmaParams>, usize, usize), CudaSmmaError> {
        if data_f32.is_empty() {
            return Err(CudaSmmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaSmmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaSmmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let len = data_f32.len();
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaSmmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaSmmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaSmmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaSmmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaSmmaError::InvalidInput(
                "series_len or n_combos exceed i32::MAX".into(),
            ));
        }
        if first_valid > i32::MAX as usize {
            return Err(CudaSmmaError::InvalidInput(
                "first_valid exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("smma_batch_f32")
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        let block_x = match self.policy.batch { BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x, _ => 128 };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe { (*(self as *const _ as *mut CudaSmma)).last_batch = Some(BatchKernelSelected::OneD { block_x }); }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warms_ptr = d_warms.as_device_ptr().as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warms_ptr as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        }

        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn smma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmmaError> {
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_warms,
            first_valid,
            series_len,
            n_combos,
            d_out,
        )
    }

    pub fn smma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SmmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaSmmaError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // Basic VRAM check (prices + periods + warms + output) with ~64MB headroom
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let warms_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + warms_bytes + out_bytes;
        if let Ok((free, _total)) = mem_get_info() {
            let headroom = 64usize * 1024 * 1024;
            if required.saturating_add(headroom) > free {
                return Err(CudaSmmaError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (required as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warms_i32 = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            if period > i32::MAX as usize {
                return Err(CudaSmmaError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            let warm = first_valid + period - 1;
            if warm > i32::MAX as usize {
                return Err(CudaSmmaError::InvalidInput(
                    "warm index exceeds i32::MAX".into(),
                ));
            }
            periods_i32.push(period as i32);
            warms_i32.push(warm as i32);
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_warms,
            first_valid,
            series_len,
            n_combos,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmmaParams,
    ) -> Result<(Vec<i32>, usize), CudaSmmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSmmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaSmmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let period = params.period.unwrap_or(7);
        if period == 0 {
            return Err(CudaSmmaError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaSmmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaSmmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period {
                return Err(CudaSmmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaSmmaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmmaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaSmmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaSmmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("smma_multi_series_one_param_f32")
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x, _ => 128 };
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe { (*(self as *const _ as *mut CudaSmma)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }

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
                .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        }

        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn smma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaSmmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn smma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmmaParams,
    ) -> Result<DeviceArrayF32, CudaSmmaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // Basic VRAM check (input + first_valids + output) with ~64MB headroom
        let input_bytes = cols * rows * std::mem::size_of::<f32>();
        let fv_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = input_bytes + fv_bytes + out_bytes;
        if let Ok((free, _total)) = mem_get_info() {
            let headroom = 64usize * 1024 * 1024;
            if required.saturating_add(headroom) > free {
                return Err(CudaSmmaError::InvalidInput(format!(
                    "estimated device memory {:.2} MB exceeds free VRAM",
                    (required as f64) / (1024.0 * 1024.0)
                )));
            }
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn smma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSmmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaSmmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;

        d_out_tm
            .copy_to(out_tm)
            .map_err(|e| CudaSmmaError::Cuda(e.to_string()))?;
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        smma_benches,
        CudaSmma,
        crate::indicators::moving_averages::smma::SmmaBatchRange,
        crate::indicators::moving_averages::smma::SmmaParams,
        smma_batch_dev,
        smma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::smma::SmmaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::smma::SmmaParams { period: Some(64) },
        "smma",
        "smma"
    );
    pub use smma_benches::bench_profiles;
}
