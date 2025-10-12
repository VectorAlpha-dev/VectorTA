//! CUDA wrapper for FWMA (Fibonacci Weighted Moving Average) kernels.
//!
//! Aligned with the ALMA wrapper API/policy:
//! - PTX loaded with target-from-context and JIT O2→fallbacks
//! - Policy-driven kernel selection (Plain only for FWMA today)
//! - VRAM estimates + headroom checks; grid.y chunking (<= 65_535)
//! - Warmup/NaN semantics identical to scalar reference
//!
//! Kernels expected (present minimal set):
//! - "fwma_batch_f32"                          // one-series × many-params
//! - "fwma_multi_series_one_param_f32"        // many-series × one-param (time-major)
//!
//! Optional symbols may be added in the future (tiled/2x/on-device weights),
//! but the wrapper preserves the public API and selection fields today.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::fwma::{FwmaBatchRange, FwmaParams};
use cust::context::CacheConfig;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cuda_sys;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// -------- Kernel selection policy (mirrors ALMA, minimal variants enabled) --------

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
pub struct CudaFwmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaFwmaPolicy {
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

#[derive(Debug)]
pub enum CudaFwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaFwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaFwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaFwmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaFwmaError {}

pub struct CudaFwma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaFwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaFwma {
    // ---- Keep these aligned with the kernel compile-time macros ----
    // Must equal FWMA_TILE_T used when compiling the PTX (default 256 in .cu).
    const FWMA_TILE_T_HOST: u32 = 256;
    // Must equal FWMA_TIME_STEPS_PER_BLOCK used by the many-series kernel (default 4 in .cu).
    const FWMA_TIME_STEPS_PER_BLOCK_HOST: u32 = 4;

    #[inline]
    fn set_kernel_smem_prefs(
        &self,
        func: &mut cust::function::Function,
        smem_bytes: usize,
    ) -> Result<(), CudaFwmaError> {
        // Prefer shared memory over L1 (hint; driver may choose otherwise)
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        // Opt-in to larger dynamic shared memory and prefer SMEM carveout when supported
        unsafe {
            let raw = func.to_raw();
            let _ = cuda_sys::cuFuncSetAttribute(
                raw,
                cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_bytes as i32,
            );
            let _ = cuda_sys::cuFuncSetAttribute(
                raw,
                cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            );
        }
        Ok(())
    }
    pub fn new(device_id: usize) -> Result<Self, CudaFwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/fwma_kernel.ptx"));
        // Try with target-from-context and O2, then relax progressively
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaFwmaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaFwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaFwmaPolicy,
    ) -> Result<Self, CudaFwmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaFwmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaFwmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaFwmaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
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
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }
    #[inline]
    fn grid_y_chunks(n_combos: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_Y: usize = 65_535;
        (0..n_combos).step_by(MAX_GRID_Y).map(move |start| {
            let len = (n_combos - start).min(MAX_GRID_Y);
            (start, len)
        })
    }
    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_s = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_s || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] FWMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaFwma)).debug_batch_logged = true;
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
                let per_s = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_s || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] FWMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaFwma)).debug_many_logged = true;
                }
            }
        }
    }

    fn expand_grid(range: &FwmaBatchRange) -> Vec<FwmaParams> {
        fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            (start..=end).step_by(step).collect()
        }
        let periods = axis(range.period);
        let mut out = Vec::with_capacity(periods.len());
        for p in periods {
            out.push(FwmaParams { period: Some(p) });
        }
        out
    }

    fn fibonacci_weights_f32(period: usize) -> Result<Vec<f32>, CudaFwmaError> {
        if period == 0 {
            return Err(CudaFwmaError::InvalidInput(
                "period must be greater than zero".into(),
            ));
        }
        if period == 1 {
            return Ok(vec![1.0f32]);
        }
        let mut fib = vec![1.0f64; period];
        for i in 2..period {
            fib[i] = fib[i - 1] + fib[i - 2];
        }
        let sum: f64 = fib.iter().sum();
        if sum == 0.0 {
            return Err(CudaFwmaError::InvalidInput(format!(
                "Fibonacci weights sum to zero for period {}",
                period
            )));
        }
        let inv = 1.0 / sum;
        Ok(fib.into_iter().map(|v| (v * inv) as f32).collect())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &FwmaBatchRange,
    ) -> Result<(Vec<FwmaParams>, usize, usize, usize, Vec<f32>), CudaFwmaError> {
        if data_f32.is_empty() {
            return Err(CudaFwmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaFwmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaFwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaFwmaError::InvalidInput("period must be > 0".into()));
            }
            if period > series_len {
                return Err(CudaFwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            if series_len - first_valid < period {
                return Err(CudaFwmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    series_len - first_valid
                )));
            }
            if period > max_period {
                max_period = period;
            }
        }

        let n_combos = combos.len();
        let mut weights_flat = vec![0.0f32; n_combos * max_period];
        for (row, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap();
            let weights = Self::fibonacci_weights_f32(period)?;
            let base = row * max_period;
            weights_flat[base..base + period].copy_from_slice(&weights);
        }

        Ok((combos, first_valid, series_len, max_period, weights_flat))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFwmaError> {
        if series_len == 0 || n_combos == 0 || max_period == 0 {
            return Err(CudaFwmaError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaFwmaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }
        // Select plain batch kernel (only variant available today)
        let mut func = self
            .module
            .get_function("fwma_batch_f32")
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => Self::FWMA_TILE_T_HOST, // match device tile width
        };
        if block_x < Self::FWMA_TILE_T_HOST {
            return Err(CudaFwmaError::InvalidInput(format!(
                "block_x ({}) must be >= FWMA_TILE_T ({}) used by the kernel",
                block_x,
                Self::FWMA_TILE_T_HOST
            )));
        }
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();
        // Dynamic SMEM: weights (max_period) + input tile (block_x + max_period - 1)
        let shared_bytes = ((max_period + (block_x as usize + max_period - 1))
            * std::mem::size_of::<f32>()) as u32;
        self.set_kernel_smem_prefs(&mut func, shared_bytes as usize)?;

        // Chunk grid.y and offset pointers for each slice
        for (start, len) in Self::grid_y_chunks(n_combos) {
            let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut weights_ptr = d_weights.as_device_ptr().add(start * max_period).as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut warms_ptr = d_warms.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = len as i32;
                let mut max_period_i = max_period as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
            }
        }
        // Introspection/logging once per scenario
        unsafe {
            let this = self as *const _ as *mut CudaFwma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn fwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFwmaError> {
        self.launch_batch_kernel(
            d_prices, d_weights, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn fwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &FwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaFwmaError> {
        let (combos, first_valid, series_len, max_period, weights_flat) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // VRAM estimate and headroom (64MB)
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let weights_bytes = n_combos * max_period * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let warms_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + periods_bytes + warms_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(required, headroom) {
            return Err(CudaFwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let warms_i32: Vec<i32> = combos
            .iter()
            .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
            .collect();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices, &d_weights, &d_periods, &d_warms, series_len, n_combos, max_period,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;

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
        params: &FwmaParams,
    ) -> Result<(Vec<i32>, Vec<f32>, usize), CudaFwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaFwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaFwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(CudaFwmaError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaFwmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let weights = Self::fibonacci_weights_f32(period)?;

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
                CudaFwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period {
                return Err(CudaFwmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaFwmaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, weights, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFwmaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaFwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaFwmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let mut func = self
            .module
            .get_function("fwma_multi_series_one_param_f32")
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        // Threads map across series (x), small time loop per block (y tiles time)
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 256,
        };
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid_y = ((series_len as u32) + Self::FWMA_TIME_STEPS_PER_BLOCK_HOST - 1)
            / Self::FWMA_TIME_STEPS_PER_BLOCK_HOST;
        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32; // weights only
        self.set_kernel_smem_prefs(&mut func, shared_bytes as usize)?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
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
                .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        }

        // Introspection/logging
        unsafe {
            let this = self as *const _ as *mut CudaFwma;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        Ok(())
    }

    pub fn fwma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFwmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaFwmaError::InvalidInput(
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

    pub fn fwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &FwmaParams,
    ) -> Result<DeviceArrayF32, CudaFwmaError> {
        let (first_valids, weights, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM estimate
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let weights_bytes = period * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + first_valid_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaFwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn fwma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &FwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaFwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaFwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, weights, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;

        // Direct device → host slice copy (one copy)
        d_out_tm
            .copy_to(out_tm)
            .map_err(|e| CudaFwmaError::Cuda(e.to_string()))?;

        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        fwma_benches,
        CudaFwma,
        crate::indicators::moving_averages::fwma::FwmaBatchRange,
        crate::indicators::moving_averages::fwma::FwmaParams,
        fwma_batch_dev,
        fwma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::fwma::FwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::fwma::FwmaParams { period: Some(64) },
        "fwma",
        "fwma"
    );
    pub use fwma_benches::bench_profiles;
}
