//! CUDA wrapper for the Sine Weighted Moving Average (SINWMA) kernels.
//!
//! Mirrors the ALMA/CWMA wrapper shape for API parity:
//! - VRAM-first design returning `DeviceArrayF32` handles
//! - JIT options: DetermineTargetFromContext + OptLevel O2 with graceful fallbacks
//! - NON_BLOCKING stream
//! - Policy enums and introspection (Plain 1D for batch and many-series)
//! - VRAM checks using `mem_get_info` and grid.y chunking (≤ 65_535)
//!
//! Math category: dot-product MA (fixed weights per period). We compute sine
//! weights on-device per kernel launch to avoid extra VRAM traffic. This keeps
//! parity with warmup/NaN semantics of the scalar path: outputs are NaN for
//! `first_valid + period - 1` warmup, then normalized windows thereafter.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::sinwma::{SinWmaBatchRange, SinWmaParams};
use cust::context::{CacheConfig, Context, SharedMemoryConfig};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer, LockedBuffer, AsyncCopyDestination};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// raw driver symbol to opt-in >48KB dynamic shared memory per block
use cust::sys::{cuFuncSetAttribute, CUfunction_attribute_enum as CUfuncAttr};

#[derive(Debug)]
pub enum CudaSinwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSinwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSinwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSinwmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaSinwmaError {}

// Kernel policy (mirrors ALMA/CWMA). We only implement 1D-plain variants,
// but expose the enums for parity and future growth.
#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput { One, Two }

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy { Auto, Plain { block_x: u32 }, Tiled { tile: u32, per_thread: BatchThreadsPerOutput } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy { Auto, OneD { block_x: u32 }, Tiled2D { tx: u32, ty: u32 } }

#[derive(Clone, Copy, Debug)]
pub struct CudaSinwmaPolicy { pub batch: BatchKernelPolicy, pub many_series: ManySeriesKernelPolicy }
impl Default for CudaSinwmaPolicy {
    fn default() -> Self { Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto } }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaSinwma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaSinwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaSinwma {
    pub fn new(device_id: usize) -> Result<Self, CudaSinwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/sinwma_kernel.ptx"));
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaSinwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaSinwmaPolicy) -> Result<Self, CudaSinwmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaSinwmaPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaSinwmaPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn synchronize(&self) -> Result<(), CudaSinwmaError> { self.stream.synchronize().map_err(|e| CudaSinwmaError::Cuda(e.to_string())) }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SINWMA batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaSinwma)).debug_batch_logged = true; }
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
                    eprintln!("[DEBUG] SINWMA many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaSinwma)).debug_many_logged = true; }
            }
        }
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }

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

    fn expand_periods(range: &SinWmaBatchRange) -> Vec<SinWmaParams> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect::<Vec<_>>()
        };
        periods
            .into_iter()
            .map(|p| SinWmaParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SinWmaBatchRange,
    ) -> Result<(Vec<SinWmaParams>, usize, usize, usize), CudaSinwmaError> {
        if data_f32.is_empty() {
            return Err(CudaSinwmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaSinwmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaSinwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaSinwmaError::InvalidInput("period must be >= 1".into()));
            }
            if period > series_len {
                return Err(CudaSinwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid < period {
                return Err(CudaSinwmaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period, valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, series_len, max_period))
    }

    // ======== Helpers for tiled kernels ========

    #[inline]
    fn dynamic_smem_bytes(period: usize, block_x: u32) -> usize {
        // floats needed = (2*period - 1) for weights+halo + block_x for the tile
        (2usize.saturating_mul(period).saturating_sub(1) + block_x as usize)
            * std::mem::size_of::<f32>()
    }

    fn try_pick_block_x(
        &self,
        func: &Function,
        period: usize,
        prefer: Option<u32>,
    ) -> Result<(u32, usize), CudaSinwmaError> {
        // Candidates from large to small; always warp-multiples.
        let mut candidates = [512u32, 384, 256, 192, 128, 96, 64, 48, 32];
        if let Some(px) = prefer {
            if !candidates.contains(&px) {
                candidates[0] = px;
            } else {
                let mut order = vec![px];
                order.extend(candidates.into_iter().filter(|&b| b != px));
                candidates = order
                    .try_into()
                    .unwrap_or([px, 512, 384, 256, 192, 128, 96, 64, 32]);
            }
        }

        // Guard by device thread-per-block limit
        let device = Device::get_device(self.device_id)
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;

        for &bx in &candidates {
            if bx > max_threads { continue; }
            let need = Self::dynamic_smem_bytes(period, bx);
            let avail = func
                .available_dynamic_shared_memory_per_block(GridSize::xy(1, 1), BlockSize::xyz(bx, 1, 1))
                .unwrap_or(48 * 1024);
            if need <= avail { return Ok((bx, need)); }
        }
        // As a last resort, derive a minimal bx that fits
        let probe_bx = 64u32.min(max_threads);
        let avail = func
            .available_dynamic_shared_memory_per_block(GridSize::xy(1, 1), BlockSize::xyz(probe_bx, 1, 1))
            .unwrap_or(48 * 1024);
        let avail_floats = avail / std::mem::size_of::<f32>();
        let min_floats = (2 * period - 1) as usize;
        if avail_floats <= min_floats {
            return Err(CudaSinwmaError::InvalidInput(format!(
                "period={} requires too much shared memory (need > {}B, have {}B)",
                period,
                (min_floats + 32) * 4,
                avail
            )));
        }
        let mut bx = (avail_floats - min_floats) as u32;
        bx = (bx / 32).max(1) * 32;
        bx = bx.min(max_threads).max(32);
        let need = Self::dynamic_smem_bytes(period, bx);
        Ok((bx, need))
    }

    #[inline]
    fn prefer_shared_and_optin(&self, func: &mut Function, dynamic_smem: usize) -> Result<(), CudaSinwmaError> {
        // Prefer shared carve-out and 4-byte bank size for f32 traffic
        func.set_cache_config(CacheConfig::PreferShared)
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
        func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize)
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;

        // Opt-in to >48KB if requested; non-fatal on failure
        unsafe {
            let raw = func.to_raw();
            let rc = cuFuncSetAttribute(
                raw,
                CUfuncAttr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                dynamic_smem as i32,
            );
            let _ = rc; // ignore non-success; launch may still succeed if within 48KB
        }
        Ok(())
    }

    #[inline]
    fn grid_y_chunks(n_combos: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_Y: usize = 65_535;
        (0..n_combos)
            .step_by(MAX_GRID_Y)
            .map(move |start| {
                let len = (n_combos - start).min(MAX_GRID_Y);
                (start, len)
            })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSinwmaError> {
        if max_period == 0 {
            return Err(CudaSinwmaError::InvalidInput(
                "max_period must be positive".into(),
            ));
        }

        let mut func = self
            .module
            .get_function("sinwma_batch_f32")
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;

        // Auto-choose a block size that fits dynamic shared memory for worst-case period
        let prefer = match self.policy.batch { BatchKernelPolicy::Plain { block_x } => Some(block_x), _ => None };
        let (block_x, shared_bytes) = self.try_pick_block_x(&func, max_period, prefer)?;
        self.prefer_shared_and_optin(&mut func, shared_bytes)?;
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();

        for (start, len) in Self::grid_y_chunks(n_combos) {
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let grid_chunk: GridSize = (grid_x, len as u32, 1).into();
                self.stream
                    .launch(&func, grid_chunk, block, shared_bytes as u32, args)
                    .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
            }
        }
        unsafe { (*(self as *const _ as *mut CudaSinwma)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[SinWmaParams],
        first_valid: usize,
        series_len: usize,
        max_period: usize,
    ) -> Result<DeviceArrayF32, CudaSinwmaError> {
        let n_combos = combos.len();
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB safety margin
        if !Self::will_fit(required, headroom) {
            return Err(CudaSinwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let use_pinned = std::env::var("CUDA_PINNED").ok().as_deref() == Some("1");

        let d_prices: DeviceBuffer<f32> = if use_pinned {
            let h_prices = LockedBuffer::from_slice(data_f32)
                .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
            unsafe {
                let mut d = DeviceBuffer::uninitialized_async(series_len, &self.stream)
                    .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
                d.async_copy_from(h_prices.as_slice(), &self.stream)
                    .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
                d
            }
        } else {
            DeviceBuffer::from_slice(data_f32)
                .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?
        };

        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let d_periods: DeviceBuffer<i32> = if use_pinned {
            let h_periods = LockedBuffer::from_slice(&periods_i32)
                .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
            unsafe {
                let mut d = DeviceBuffer::uninitialized_async(n_combos, &self.stream)
                    .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
                d.async_copy_from(h_periods.as_slice(), &self.stream)
                    .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
                d
            }
        } else {
            DeviceBuffer::from_slice(&periods_i32)
                .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?
        };

        let mut d_out: DeviceBuffer<f32> = if use_pinned {
            unsafe { DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream) }
                .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?
        } else {
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            series_len,
            n_combos,
            first_valid,
            max_period,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn sinwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SinWmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaSinwmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)
    }

    pub fn sinwma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &SinWmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<SinWmaParams>), CudaSinwmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaSinwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn sinwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        max_period: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSinwmaError> {
        if series_len <= 0 || n_combos <= 0 || max_period <= 0 {
            return Err(CudaSinwmaError::InvalidInput(
                "series_len, n_combos and period must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            max_period as usize,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SinWmaParams,
    ) -> Result<(Vec<i32>, usize), CudaSinwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSinwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaSinwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaSinwmaError::InvalidInput("period must be >= 1".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(t as i32);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaSinwmaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if (rows as i32 - fv) < period as i32 {
                return Err(CudaSinwmaError::InvalidInput(format!(
                    "series {} lacks data: need >= {}, valid = {}",
                    series,
                    period,
                    rows as i32 - fv
                )));
            }
            first_valids[series] = fv;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSinwmaError> {
        let mut func = self
            .module
            .get_function("sinwma_many_series_one_param_time_major_f32")
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;

        let prefer = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } => Some(block_x), _ => None };
        let (block_x, shared_bytes) = self.try_pick_block_x(&func, period, prefer)?;
        self.prefer_shared_and_optin(&mut func, shared_bytes)?;
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
        }
        unsafe { (*(self as *const _ as *mut CudaSinwma)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
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
    ) -> Result<DeviceArrayF32, CudaSinwmaError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; // 32MB cushion
        if !Self::will_fit(required, headroom) {
            return Err(CudaSinwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let use_pinned = std::env::var("CUDA_PINNED").ok().as_deref() == Some("1");

        let d_prices: DeviceBuffer<f32> = if use_pinned {
            let h = LockedBuffer::from_slice(data_tm_f32).map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
            unsafe {
                let mut d = DeviceBuffer::uninitialized_async(cols * rows, &self.stream)
                    .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
                d.async_copy_from(h.as_slice(), &self.stream)
                    .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
                d
            }
        } else {
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?
        };

        let d_first_valids: DeviceBuffer<i32> = if use_pinned {
            let h = LockedBuffer::from_slice(first_valids).map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
            unsafe {
                let mut d = DeviceBuffer::uninitialized_async(cols, &self.stream)
                    .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
                d.async_copy_from(h.as_slice(), &self.stream)
                    .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;
                d
            }
        } else {
            DeviceBuffer::from_slice(first_valids).map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?
        };

        let mut d_out: DeviceBuffer<f32> = if use_pinned {
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }
                .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?
        } else {
            unsafe { DeviceBuffer::uninitialized(cols * rows) }
                .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(&d_prices, period, cols, rows, &d_first_valids, &mut d_out)?;

        self.stream
            .synchronize()
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn sinwma_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SinWmaParams,
    ) -> Result<DeviceArrayF32, CudaSinwmaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn sinwma_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SinWmaParams,
        out: &mut [f32],
    ) -> Result<(), CudaSinwmaError> {
        if out.len() != cols * rows {
            return Err(CudaSinwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                cols * rows
            )));
        }
        let arr =
            self.sinwma_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaSinwmaError::Cuda(e.to_string()))
    }

    pub fn sinwma_many_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSinwmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaSinwmaError::InvalidInput(
                "period must be >= 1 and dimensions positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            period as usize,
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
        sinwma_benches,
        CudaSinwma,
        crate::indicators::moving_averages::sinwma::SinWmaBatchRange,
        crate::indicators::moving_averages::sinwma::SinWmaParams,
        sinwma_batch_dev,
        sinwma_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::sinwma::SinWmaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::sinwma::SinWmaParams { period: Some(64) },
        "sinwma",
        "sinwma"
    );
    pub use sinwma_benches::bench_profiles;
}
