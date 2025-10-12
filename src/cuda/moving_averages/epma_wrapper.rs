//! CUDA wrapper for the Endpoint Moving Average (EPMA) kernels.
//!
//! Aligned with the ALMA wrapper API/policy:
//! - PTX loaded with target-from-context and JIT O4→fallbacks
//! - Policy-driven kernel selection (Plain only for EPMA today)
//! - VRAM estimates + headroom checks; grid.y chunking (<= 65_535)
//! - Warmup/NaN semantics identical to scalar reference
//!
//! Kernels expected (present minimal set):
//! - "epma_batch_f32"                                   // one-series × many-params
//! - "epma_many_series_one_param_time_major_f32"       // many-series × one-param (time-major)

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::epma::{EpmaBatchRange, EpmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// Keep TILE in sync with kTile() in the CUDA kernels.
const EPMA_TILE: u32 = 8;

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
pub struct CudaEpmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaEpmaPolicy {
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
pub enum CudaEpmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEpmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEpmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEpmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaEpmaError {}

pub struct CudaEpma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaEpmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEpma {
    pub fn new(device_id: usize) -> Result<Self, CudaEpmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/epma_kernel.ptx"));
        // Prefer context-derived target and most optimized JIT level.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaEpmaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaEpmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create using an explicit policy.
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaEpmaPolicy,
    ) -> Result<Self, CudaEpmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaEpmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaEpmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    /// Expose synchronize for benches/tests that pre-stage device buffers.
    pub fn synchronize(&self) -> Result<(), CudaEpmaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

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
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_s = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_s || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] EPMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaEpma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] EPMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaEpma)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_Y: usize = 65_535;
        (0..n).step_by(MAX_GRID_Y).map(move |start| {
            let len = (n - start).min(MAX_GRID_Y);
            (start, len)
        })
    }

    fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            vec![start]
        } else if start <= end {
            (start..=end).step_by(step).collect()
        } else {
            vec![start]
        }
    }

    fn expand_range(range: &EpmaBatchRange) -> Vec<EpmaParams> {
        let periods = Self::axis_usize(range.period);
        let offsets = Self::axis_usize(range.offset);
        let mut combos = Vec::with_capacity(periods.len() * offsets.len());
        for &p in &periods {
            for &o in &offsets {
                combos.push(EpmaParams {
                    period: Some(p),
                    offset: Some(o),
                });
            }
        }
        combos
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EpmaBatchRange,
    ) -> Result<(Vec<EpmaParams>, usize, usize, usize), CudaEpmaError> {
        if data_f32.is_empty() {
            return Err(CudaEpmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaEpmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaEpmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            let offset = prm.offset.unwrap_or(usize::MAX);
            if period < 2 {
                return Err(CudaEpmaError::InvalidInput(format!(
                    "invalid period {} (must be >= 2)",
                    period
                )));
            }
            if offset >= period {
                return Err(CudaEpmaError::InvalidInput(format!(
                    "offset {} must be < period {}",
                    offset, period
                )));
            }
            if period > series_len {
                return Err(CudaEpmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            let needed = period + offset + 1;
            let valid = series_len - first_valid;
            if valid < needed {
                return Err(CudaEpmaError::InvalidInput(format!(
                    "not enough valid data: need >= {}, valid = {}",
                    needed, valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, series_len, max_period))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_offsets: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        _max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEpmaError> {
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        } as u32;
        let func = self
            .module
            .get_function("epma_batch_f32")
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;

        // Chunk over grid.y to respect device limit and offset params/output pointers per chunk.
        for (start_combo, len_combo) in Self::grid_y_chunks(n_combos) {
            // Each thread emits EPMA_TILE outputs; size grid.x accordingly.
            let bx_times_tile = block_x.saturating_mul(EPMA_TILE);
            let grid_x = ((series_len as u32 + bx_times_tile - 1) / bx_times_tile).max(1);
            let grid: GridSize = (grid_x, len_combo as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                // Advance param pointers by start_combo
                let mut periods_ptr = d_periods.as_device_ptr().add(start_combo).as_raw();
                let mut offsets_ptr = d_offsets.as_device_ptr().add(start_combo).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len_combo as i32;
                let mut first_valid_i = first_valid as i32;
                // Advance output pointer by whole rows already skipped
                let mut out_ptr = d_out.as_device_ptr().add(start_combo * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut offsets_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                // No dynamic shared memory in kernels now.
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
            }
        }
        unsafe {
            let this = self as *const _ as *mut CudaEpma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[EpmaParams],
        first_valid: usize,
        series_len: usize,
        max_period: usize,
    ) -> Result<DeviceArrayF32, CudaEpmaError> {
        let n_combos = combos.len();
        let mut periods_i32 = vec![0i32; n_combos];
        let mut offsets_i32 = vec![0i32; n_combos];
        for (idx, prm) in combos.iter().enumerate() {
            periods_i32[idx] = prm.period.unwrap() as i32;
            offsets_i32[idx] = prm.offset.unwrap() as i32;
        }

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let offsets_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + offsets_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64 MB safety margin (align with ALMA)
        if !Self::will_fit(required, headroom) {
            return Err(CudaEpmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let d_offsets = DeviceBuffer::from_slice(&offsets_i32)
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_offsets,
            series_len,
            n_combos,
            first_valid,
            max_period,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn epma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EpmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEpmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)
    }

    pub fn epma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EpmaBatchRange,
        out: &mut [f32],
    ) -> Result<(), CudaEpmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        if out.len() != combos.len() * series_len {
            return Err(CudaEpmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                combos.len() * series_len
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))
    }

    /// Batch into pinned host memory (optional fast path).
    pub fn epma_batch_into_pinned_host_f32(
        &self,
        data_pinned: &LockedBuffer<f32>,
        sweep: &crate::indicators::moving_averages::epma::EpmaBatchRange,
        out_pinned: &mut LockedBuffer<f32>,
    ) -> Result<(), CudaEpmaError> {
        // Reuse existing input validation
        let data_f32: &[f32] = data_pinned.as_slice();
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;

        if out_pinned.len() != combos.len() * series_len {
            return Err(CudaEpmaError::InvalidInput(format!(
                "out pinned buffer wrong length: got {}, expected {}",
                out_pinned.len(),
                combos.len() * series_len
            )));
        }

        // Build small host arrays for params
        let n_combos = combos.len();
        let mut periods_i32 = vec![0i32; n_combos];
        let mut offsets_i32 = vec![0i32; n_combos];
        for (idx, prm) in combos.iter().enumerate() {
            periods_i32[idx] = prm.period.unwrap() as i32;
            offsets_i32[idx] = prm.offset.unwrap() as i32;
        }

        // Allocate device buffers
        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let d_offsets = DeviceBuffer::from_slice(&offsets_i32)
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;

        // Async H->D copy from pinned host
        unsafe {
            d_prices
                .async_copy_from(data_pinned.as_slice(), &self.stream)
                .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        }

        // Launch kernel (shared mem = 0 inside launch) then async D->H to pinned host
        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_offsets,
            series_len,
            n_combos,
            first_valid,
            max_period,
            &mut d_out,
        )?;

        unsafe {
            d_out
                .async_copy_to(out_pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        }

        // Wait for kernel + copies to finish
        self.stream
            .synchronize()
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))
    }

    pub fn epma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_offsets: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEpmaError> {
        if n_combos == 0 {
            return Err(CudaEpmaError::InvalidInput("n_combos must be > 0".into()));
        }
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_offsets,
            series_len,
            n_combos,
            first_valid,
            max_period,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EpmaParams,
    ) -> Result<(Vec<i32>, usize, usize), CudaEpmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEpmaError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaEpmaError::InvalidInput(format!(
                "expected {} elements, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }
        let period = params.period.unwrap_or(0);
        let offset = params.offset.unwrap_or(usize::MAX);
        if period < 2 {
            return Err(CudaEpmaError::InvalidInput(format!(
                "invalid period {} (must be >= 2)",
                period
            )));
        }
        if offset >= period {
            return Err(CudaEpmaError::InvalidInput(format!(
                "offset {} must be < period {}",
                offset, period
            )));
        }

        let mut first_valids = vec![0i32; cols];
        let needed = period + offset + 1;
        for series in 0..cols {
            let mut found = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + series];
                if !v.is_nan() {
                    found = Some(t);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaEpmaError::InvalidInput(format!("series {} is entirely NaN", series))
            })?;
            let valid = rows - fv;
            if valid < needed {
                return Err(CudaEpmaError::InvalidInput(format!(
                    "series {} lacks data: need >= {}, valid = {}",
                    series, needed, valid
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period, offset))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: usize,
        offset: usize,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEpmaError> {
        if period < 2 {
            return Err(CudaEpmaError::InvalidInput("period must be >= 2".into()));
        }
        if offset >= period {
            return Err(CudaEpmaError::InvalidInput(format!(
                "offset {} must be < period {}",
                offset, period
            )));
        }

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        } as u32;

        // rows == series_len along time; size grid.x using EPMA_TILE
        let bx_times_tile = block_x.saturating_mul(EPMA_TILE);
        let grid_x = ((rows as u32 + bx_times_tile - 1) / bx_times_tile).max(1);
        let grid: GridSize = (grid_x, cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let func = self
            .module
            .get_function("epma_many_series_one_param_time_major_f32")
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut offset_i = offset as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut offset_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            // No dynamic shared memory in kernels now.
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        }
        unsafe {
            let this = self as *const _ as *mut CudaEpma;
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
        offset: usize,
    ) -> Result<DeviceArrayF32, CudaEpmaError> {
        let total = cols * rows;
        let prices_bytes = total * std::mem::size_of::<f32>();
        let first_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = total * std::mem::size_of::<f32>();
        let required = prices_bytes + first_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaEpmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            period,
            offset,
            cols,
            rows,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn epma_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EpmaParams,
    ) -> Result<DeviceArrayF32, CudaEpmaError> {
        let (first_valids, period, offset) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period, offset)
    }

    pub fn epma_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EpmaParams,
        out: &mut [f32],
    ) -> Result<(), CudaEpmaError> {
        if out.len() != cols * rows {
            return Err(CudaEpmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                cols * rows
            )));
        }
        let arr =
            self.epma_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaEpmaError::Cuda(e.to_string()))
    }

    pub fn epma_many_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: i32,
        offset: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEpmaError> {
        if period < 2 || offset >= period {
            return Err(CudaEpmaError::InvalidInput(format!(
                "period {}, offset {} invalid",
                period, offset
            )));
        }
        if num_series <= 0 || series_len <= 0 {
            return Err(CudaEpmaError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            period as usize,
            offset as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out,
        )
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        epma_benches,
        CudaEpma,
        crate::indicators::moving_averages::epma::EpmaBatchRange,
        crate::indicators::moving_averages::epma::EpmaParams,
        epma_batch_dev,
        epma_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::epma::EpmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            offset: (4, 4, 0)
        },
        crate::indicators::moving_averages::epma::EpmaParams {
            period: Some(64),
            offset: Some(4)
        },
        "epma",
        "epma"
    );
    pub use epma_benches::bench_profiles;
}
