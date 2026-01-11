//! CUDA wrapper for FWMA (Fibonacci Weighted Moving Average) kernels.
//!
//! Aligned with the ALMA wrapper API/policy:
//! - PTX loaded with target-from-context and JIT O2→fallbacks
//! - Policy-driven kernel selection (Plain only for FWMA today)
//! - VRAM estimates + headroom checks; grid.y chunking (<= 65_535)
//! - Warmup/NaN semantics identical to scalar reference
//!
//! Kernels expected (present minimal set):
//! - "fwma_batch_f32"                          
//! - "fwma_multi_series_one_param_f32"        
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
use std::sync::Arc;
use thiserror::Error;
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



#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Debug, Error)]
pub enum CudaFwmaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("arithmetic overflow while computing {context}")]
    ArithmeticOverflow { context: &'static str },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

pub struct CudaFwma {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
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
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/fwma_kernel.ptx"));
        // Try with target-from-context and O2, then relax progressively
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
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

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
    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> { self._context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }
    pub fn synchronize(&self) -> Result<(), CudaFwmaError> {
        self.stream.synchronize().map_err(Into::into)
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
    fn will_fit_required(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaFwmaError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _)) = Self::device_mem_info() {
            let need = required_bytes
                .checked_add(headroom_bytes)
                .ok_or(CudaFwmaError::ArithmeticOverflow { context: "required+headroom (bytes)" })?;
            if need > free {
                return Err(CudaFwmaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes });
            }
        }
        Ok(())
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
            let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
            (lo..=hi).step_by(step).collect()
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
            .map_err(|_| CudaFwmaError::MissingKernelSymbol { name: "fwma_batch_f32" })?;
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
        // Guard block size with device attribute (max threads per block)
        let dev = Device::get_device(self.device_id)?;
        let max_tpb = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)? as u32;
        if block_x > max_tpb {
            return Err(CudaFwmaError::LaunchConfigTooLarge { gx: ((series_len as u32) + block_x - 1) / block_x, gy: n_combos as u32, gz: 1, bx: block_x, by: 1, bz: 1 });
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
                self.stream.launch(&func, grid, block, shared_bytes, args)?;
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
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prices_bytes = series_len
            .checked_mul(sz_f32)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "series_len*sizeof(f32)" })?;
        let weights_elems = n_combos
            .checked_mul(max_period)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "n_combos*max_period" })?;
        let weights_bytes = weights_elems
            .checked_mul(sz_f32)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "weights_elems*sizeof(f32)" })?;
        let periods_bytes = n_combos
            .checked_mul(sz_i32)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "n_combos*sizeof(i32)" })?;
        let warms_bytes = n_combos
            .checked_mul(sz_i32)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "n_combos*sizeof(i32)" })?;
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "n_combos*series_len" })?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "out_elems*sizeof(f32)" })?;
        let required = prices_bytes
            .checked_add(weights_bytes).ok_or(CudaFwmaError::ArithmeticOverflow { context: "prices+weights" })?
            .checked_add(periods_bytes).ok_or(CudaFwmaError::ArithmeticOverflow { context: "…+periods" })?
            .checked_add(warms_bytes).ok_or(CudaFwmaError::ArithmeticOverflow { context: "…+warms" })?
            .checked_add(out_bytes).ok_or(CudaFwmaError::ArithmeticOverflow { context: "…+out" })?;
        let headroom = 64 * 1024 * 1024; // ~64MB
        Self::will_fit_required(required, headroom)?;

        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let warms_i32: Vec<i32> = combos
            .iter()
            .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
            .collect();

        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_warms = DeviceBuffer::from_slice(&warms_i32)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }?;

        self.launch_batch_kernel(
            &d_prices, &d_weights, &d_periods, &d_warms, series_len, n_combos, max_period,
            &mut d_out,
        )?;
        self.stream.synchronize()?;

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
            .map_err(|_| CudaFwmaError::MissingKernelSymbol { name: "fwma_multi_series_one_param_f32" })?;
        // Threads map across series (y), time tiles in x.
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 256,
        };
        let grid_x = ((series_len as u32) + Self::FWMA_TIME_STEPS_PER_BLOCK_HOST - 1)
            / Self::FWMA_TIME_STEPS_PER_BLOCK_HOST;
        let grid_y = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        // Conservative launch validation (optional): ensure dims are non-zero and within typical limits
        if grid_x == 0 || grid_y == 0 || block_x == 0 {
            return Err(CudaFwmaError::LaunchConfigTooLarge { gx: grid_x, gy: grid_y, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
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
            self.stream.launch(&func, grid, block, shared_bytes, args)?;
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
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prices_elems = cols
            .checked_mul(rows)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "cols*rows" })?;
        let prices_bytes = prices_elems
            .checked_mul(sz_f32)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "prices_elems*sizeof(f32)" })?;
        let weights_bytes = period
            .checked_mul(sz_f32)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "period*sizeof(f32)" })?;
        let first_valid_bytes = cols
            .checked_mul(sz_i32)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "cols*sizeof(i32)" })?;
        let out_bytes = prices_elems
            .checked_mul(sz_f32)
            .ok_or(CudaFwmaError::ArithmeticOverflow { context: "out_elems*sizeof(f32)" })?;
        let required = prices_bytes
            .checked_add(weights_bytes).ok_or(CudaFwmaError::ArithmeticOverflow { context: "prices+weights" })?
            .checked_add(first_valid_bytes).ok_or(CudaFwmaError::ArithmeticOverflow { context: "…+first_valids" })?
            .checked_add(out_bytes).ok_or(CudaFwmaError::ArithmeticOverflow { context: "…+out" })?;
        let headroom = 32 * 1024 * 1024;
        Self::will_fit_required(required, headroom)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_weights = DeviceBuffer::from_slice(&weights)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream.synchronize()?;

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
        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_weights = DeviceBuffer::from_slice(&weights)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream.synchronize()?;

        // Use pinned buffer for potentially large copies
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(cols * rows)? };
        d_out_tm.copy_to(out_tm)?;

        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::fwma::FwmaParams;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct FwmaBatchDevState {
        cuda: CudaFwma,
        d_prices: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_warms: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for FwmaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_weights,
                    &self.d_periods,
                    &self.d_warms,
                    self.series_len,
                    self.n_combos,
                    self.max_period,
                    &mut self.d_out,
                )
                .expect("fwma batch kernel");
            self.cuda.stream.synchronize().expect("fwma sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaFwma::new(0).expect("cuda fwma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = FwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (combos, first_valid, series_len, max_period, weights_flat) =
            CudaFwma::prepare_batch_inputs(&price, &sweep).expect("fwma prepare batch inputs");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let warms_i32: Vec<i32> = combos
            .iter()
            .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
            .collect();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("d_warms");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(FwmaBatchDevState {
            cuda,
            d_prices,
            d_weights,
            d_periods,
            d_warms,
            series_len,
            n_combos,
            max_period,
            d_out,
        })
    }

    struct FwmaManyDevState {
        cuda: CudaFwma,
        d_prices_tm: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for FwmaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_weights,
                    &self.d_first_valids,
                    self.period,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("fwma many-series kernel");
            self.cuda.stream.synchronize().expect("fwma sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaFwma::new(0).expect("cuda fwma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = FwmaParams { period: Some(64) };
        let (first_valids, weights, period) =
            CudaFwma::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("fwma prepare many-series inputs");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_weights = DeviceBuffer::from_slice(&weights).expect("d_weights");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(FwmaManyDevState {
            cuda,
            d_prices_tm,
            d_weights,
            d_first_valids,
            cols,
            rows,
            period,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "fwma",
                "one_series_many_params",
                "fwma_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "fwma",
                "many_series_one_param",
                "fwma_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
