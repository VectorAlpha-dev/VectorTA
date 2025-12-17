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
use cust::memory::AsyncCopyDestination;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::{c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

// Must match device-side array in kernels/cuda/moving_averages/nma_kernel.cu
const NMA_MAX_PERIOD: usize = 4096;

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

#[derive(Debug, Error)]
pub enum CudaNmaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")] 
    NotImplemented,
}

pub struct CudaNma {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaNmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    // Tracks how many weights are uploaded to __constant__ c_sqrt_diffs
    weights_len_uploaded: AtomicUsize,
}

impl CudaNma {
    /// Create a new CUDA NMA wrapper and load PTX with robust JIT options.
    pub fn new(device_id: usize) -> Result<Self, CudaNmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

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
            policy: CudaNmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            weights_len_uploaded: AtomicUsize::new(0),
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
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> { self._context.clone() }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

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
    fn vram_check(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaNmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _total)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaNmaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
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

    fn expand_range(range: &NmaBatchRange) -> Result<Vec<NmaParams>, CudaNmaError> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return Ok(vec![NmaParams { period: Some(start) }]);
        }
        let mut out: Vec<usize> = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                out.push(cur);
                cur = cur
                    .checked_add(step)
                    .ok_or_else(|| CudaNmaError::InvalidInput(format!(
                        "invalid range: start={} end={} step={}", start, end, step
                    )))?;
            }
        } else {
            let mut cur = start;
            loop {
                out.push(cur);
                if cur <= end { break; }
                cur = cur
                    .checked_sub(step)
                    .ok_or_else(|| CudaNmaError::InvalidInput(format!(
                        "invalid range: start={} end={} step={}", start, end, step
                    )))?;
                if cur < end { break; }
            }
        }
        if out.is_empty() {
            return Err(CudaNmaError::InvalidInput(format!(
                "invalid range: start={} end={} step={}", start, end, step
            )));
        }
        Ok(out.into_iter().map(|p| NmaParams { period: Some(p) }).collect())
    }

    fn compute_abs_diffs(data: &[f32]) -> Vec<f32> {
        let len = data.len();
        let mut diffs = vec![0f32; len];
        if len == 0 {
            return diffs;
        }
        let mut prev_ln = data[0].max(1e-10f32).ln();
        for i in 1..len {
            let ln = data[i].max(1e-10f32).ln();
            diffs[i] = (ln - prev_ln).abs();
            prev_ln = ln;
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

        let combos = Self::expand_range(sweep)?;
        if combos.is_empty() {
            return Err(CudaNmaError::InvalidInput("no parameter combinations".into()));
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

    // Upload constant-memory weights (c_sqrt_diffs) once per module
    #[inline]
    fn ensure_const_weights(&self, need: usize) -> Result<(), CudaNmaError> {
        if need == 0 {
            return Ok(());
        }
        if need > NMA_MAX_PERIOD {
            return Err(CudaNmaError::InvalidInput(format!(
                "period {} exceeds NMA_MAX_PERIOD ({})",
                need, NMA_MAX_PERIOD
            )));
        }
        let already = self.weights_len_uploaded.load(Ordering::Relaxed);
        if already >= need {
            return Ok(());
        }

        // Build host weights sqrt(i+1) - sqrt(i)
        let mut host = [0f32; NMA_MAX_PERIOD];
        for i in 0..need {
            let s0 = (i as f32).sqrt();
            let s1 = ((i + 1) as f32).sqrt();
            host[i] = s1 - s0;
        }

        // Resolve symbol and copy to constant memory
        let name = CString::new("c_sqrt_diffs").unwrap();
        let mut sym = self
            .module
            .get_global::<[f32; NMA_MAX_PERIOD]>(&name)
            .map_err(|_| CudaNmaError::MissingKernelSymbol { name: "c_sqrt_diffs" })?;
        unsafe {
            sym.copy_from(&host)?;
        }
        self.weights_len_uploaded.store(need, Ordering::Relaxed);
        Ok(())
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
            .map_err(|_| CudaNmaError::MissingKernelSymbol { name: "nma_batch_f32" })?;

        // Block size selection (default 256; good occupancy on Ada)
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        // Kernel uses static shared memory for tile+halo; no dynamic shared memory required
        let shared = 0u32;

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

            self.stream.launch(&func, grid, block, shared, args)?;
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
        // Ensure device-side constant weights are ready
        self.ensure_const_weights(max_period)?;

        // Pinned HtoD uploads with async copies
        let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let mut d_abs_diffs = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let h_prices = cust::memory::LockedBuffer::from_slice(data_f32)?;
        let h_diffs = cust::memory::LockedBuffer::from_slice(abs_diffs)?;
        unsafe {
            d_prices.async_copy_from(h_prices.as_slice(), &self.stream)?;
            d_abs_diffs.async_copy_from(h_diffs.as_slice(), &self.stream)?;
        }
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;

        // VRAM estimate and headroom (~64MB)
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let bytes_inputs = len.checked_mul(sz_f32).ok_or_else(|| CudaNmaError::InvalidInput("byte size overflow".into()))?;
        let bytes_diffs = len.checked_mul(sz_f32).ok_or_else(|| CudaNmaError::InvalidInput("byte size overflow".into()))?;
        let bytes_periods = periods.len().checked_mul(sz_i32).ok_or_else(|| CudaNmaError::InvalidInput("byte size overflow".into()))?;
        let combos_len = combos.len();
        let out_elems = combos_len.checked_mul(len).ok_or_else(|| CudaNmaError::InvalidInput("rows*cols overflow".into()))?;
        let bytes_out = out_elems.checked_mul(sz_f32).ok_or_else(|| CudaNmaError::InvalidInput("byte size overflow".into()))?;
        let required = bytes_inputs
            .checked_add(bytes_diffs).and_then(|v| v.checked_add(bytes_periods)).and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaNmaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::vram_check(required, headroom)?;

        let elems = out_elems;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

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
        let expected = combos.len().checked_mul(len).ok_or_else(|| CudaNmaError::InvalidInput("rows*cols overflow".into()))?;
        if out.len() != expected {
            return Err(CudaNmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }

        let dev =
            self.run_batch_kernel(data_f32, &combos, first_valid, len, max_period, &abs_diffs)?;
        // Use pinned DtoH for faster copy-back
        let mut h_out = unsafe { cust::memory::LockedBuffer::<f32>::uninitialized(expected) }
            .map_err(|e| CudaNmaError::Cuda(e))?;
        unsafe {
            dev.buf
                .async_copy_to(h_out.as_mut_slice(), &self.stream)
                .map_err(|e| CudaNmaError::Cuda(e))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaNmaError::Cuda(e))?;
        out.copy_from_slice(h_out.as_slice());
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
        let expected = cols.checked_mul(rows).ok_or_else(|| CudaNmaError::InvalidInput("rows*cols overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaNmaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                expected,
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
            .map_err(|_| CudaNmaError::MissingKernelSymbol { name: "nma_many_series_one_param_f32" })?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 256,
        };
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        // No dynamic shared memory required; kernel uses constant memory weights
        let shared = 0u32;

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

            self.stream.launch(&func, grid, block, shared, args)?;
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
        // Ensure constant-memory weights are available for this period
        self.ensure_const_weights(period)?;

        // Pinned HtoD uploads with async copies
        let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized(num_series * series_len) }?;
        let mut d_abs_diffs = unsafe { DeviceBuffer::<f32>::uninitialized(num_series * series_len) }?;
        let h_prices = cust::memory::LockedBuffer::from_slice(data_tm_f32)?;
        let h_diffs = cust::memory::LockedBuffer::from_slice(abs_diffs_tm)?;
        unsafe {
            d_prices.async_copy_from(h_prices.as_slice(), &self.stream)?;
            d_abs_diffs.async_copy_from(h_diffs.as_slice(), &self.stream)?;
        }
        let d_first = DeviceBuffer::from_slice(first_valids)?;
        // VRAM estimate
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let elems_inputs = num_series.checked_mul(series_len).ok_or_else(|| CudaNmaError::InvalidInput("rows*cols overflow".into()))?;
        let bytes_inputs = elems_inputs.checked_mul(sz_f32).ok_or_else(|| CudaNmaError::InvalidInput("byte size overflow".into()))?;
        let bytes_diffs = bytes_inputs;
        let bytes_first = num_series.checked_mul(sz_i32).ok_or_else(|| CudaNmaError::InvalidInput("byte size overflow".into()))?;
        let bytes_out = bytes_inputs;
        let required = bytes_inputs
            .checked_add(bytes_diffs).and_then(|v| v.checked_add(bytes_first)).and_then(|v| v.checked_add(bytes_out))
            .ok_or_else(|| CudaNmaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::vram_check(required, headroom)?;

        let elems = elems_inputs;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

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
        let expected = cols.checked_mul(rows).ok_or_else(|| CudaNmaError::InvalidInput("rows*cols overflow".into()))?;
        if out_tm.len() != expected {
            return Err(CudaNmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
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
        let expected = cols.checked_mul(rows).ok_or_else(|| CudaNmaError::InvalidInput("rows*cols overflow".into()))?;
        let mut h_out = unsafe { cust::memory::LockedBuffer::<f32>::uninitialized(expected) }
            .map_err(CudaNmaError::Cuda)?;
        unsafe {
            dev.buf
                .async_copy_to(h_out.as_mut_slice(), &self.stream)
                .map_err(CudaNmaError::Cuda)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaNmaError::Cuda)?;
        out_tm.copy_from_slice(h_out.as_slice());
        Ok(())
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
