#![cfg(feature = "cuda")]

//! CUDA wrapper for FVG Trailing Stop
//!
//! Parity goals (mirrors ALMA/CWMA wrappers where applicable):
//! - PTX embed via include_str!(concat!(env!("OUT_DIR"), "/fvg_trailing_stop_kernel.ptx"))
//! - JIT options: DetermineTargetFromContext + OptLevel O2 with fallbacks
//! - NON_BLOCKING stream
//! - VRAM checks (with env override) and grid chunking where needed
//! - Public device entries for:
//!    - Batch (one series × many params): returns four DeviceArrayF32 buffers (upper, lower, upper_ts, lower_ts)
//!    - Many-series × one param (time-major): returns same four buffers (rows × cols)
//! - Warmup/NaN semantics identical to scalar: warm = first_valid + 2 + smoothing_len − 1

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::fvg_trailing_stop::{FvgTsBatchRange, FvgTrailingStopParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaFvgTsError {
    Cuda(String),
    InvalidInput(String),
}
pub enum CudaFvgTsError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaFvgTsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaFvgTsError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaFvgTsError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaFvgTsError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaFvgTsPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
pub struct CudaFvgTsPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaFvgTsBatch {
    pub upper: DeviceArrayF32,
    pub lower: DeviceArrayF32,
    pub upper_ts: DeviceArrayF32,
    pub lower_ts: DeviceArrayF32,
    pub combos: Vec<FvgTrailingStopParams>,
}
pub struct CudaFvgTsBatch {
    pub upper: DeviceArrayF32,
    pub lower: DeviceArrayF32,
    pub upper_ts: DeviceArrayF32,
    pub lower_ts: DeviceArrayF32,
    pub combos: Vec<FvgTrailingStopParams>,
}

pub struct CudaFvgTs {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaFvgTsPolicy,
}

impl CudaFvgTs {
    pub fn new(device_id: usize) -> Result<Self, CudaFvgTsError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/fvg_trailing_stop_kernel.ptx"));
        let jit = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaFvgTsPolicy::default(),
        })
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaFvgTsPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, p: CudaFvgTsPolicy) {
        self.policy = p;
    }
    pub fn set_policy(&mut self, p: CudaFvgTsPolicy) {
        self.policy = p;
    }

    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> bool {
        if env::var("CUDA_MEM_CHECK").ok().as_deref() == Some("0") {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
        if env::var("CUDA_MEM_CHECK").ok().as_deref() == Some("0") {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    fn first_valid_ohlc_f32(h: &[f32], l: &[f32], c: &[f32]) -> Option<usize> {
        let n = h.len().min(l.len()).min(c.len());
        for i in 0..n {
            if h[i].is_finite() && l[i].is_finite() && c[i].is_finite() {
                return Some(i);
            }
        }
        for i in 0..n {
            if h[i].is_finite() && l[i].is_finite() && c[i].is_finite() {
                return Some(i);
            }
        }
        None
    }

    fn expand_grid(range: &FvgTsBatchRange) -> Vec<FvgTrailingStopParams> {
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                vec![s]
            } else {
                (s..=e).step_by(st).collect()
            }
        }
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e {
                vec![s]
            } else {
                (s..=e).step_by(st).collect()
            }
        }
        let looks = axis_usize(range.lookback);
        let smooth = axis_usize(range.smoothing);
        let mut resets = Vec::new();
        if range.reset_on_cross.0 {
            resets.push(false);
        }
        if range.reset_on_cross.1 {
            resets.push(true);
        }
        if resets.is_empty() {
            resets.push(false);
        }
        let mut out = Vec::with_capacity(looks.len() * smooth.len() * resets.len());
        for &lb in &looks {
            for &sm in &smooth {
                for &rs in &resets {
                    out.push(FvgTrailingStopParams {
                        unmitigated_fvg_lookback: Some(lb),
                        smoothing_length: Some(sm),
                        reset_on_cross: Some(rs),
                    });
                }
            }
        }
        let smooth = axis_usize(range.smoothing);
        let mut resets = Vec::new();
        if range.reset_on_cross.0 {
            resets.push(false);
        }
        if range.reset_on_cross.1 {
            resets.push(true);
        }
        if resets.is_empty() {
            resets.push(false);
        }
        let mut out = Vec::with_capacity(looks.len() * smooth.len() * resets.len());
        for &lb in &looks {
            for &sm in &smooth {
                for &rs in &resets {
                    out.push(FvgTrailingStopParams {
                        unmitigated_fvg_lookback: Some(lb),
                        smoothing_length: Some(sm),
                        reset_on_cross: Some(rs),
                    });
                }
            }
        }
        out
    }

    pub fn fvg_ts_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &FvgTsBatchRange,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &FvgTsBatchRange,
    ) -> Result<CudaFvgTsBatch, CudaFvgTsError> {
        let len = high.len();
        if len == 0 || low.len() != len || close.len() != len {
            return Err(CudaFvgTsError::InvalidInput(
                "inconsistent or empty inputs".into(),
            ));
        }
        let _first = Self::first_valid_ohlc_f32(high, low, close)
            .ok_or_else(|| CudaFvgTsError::InvalidInput("all values are NaN".into()))?;
        if len == 0 || low.len() != len || close.len() != len {
            return Err(CudaFvgTsError::InvalidInput(
                "inconsistent or empty inputs".into(),
            ));
        }
        let _first = Self::first_valid_ohlc_f32(high, low, close)
            .ok_or_else(|| CudaFvgTsError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaFvgTsError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if combos.is_empty() {
            return Err(CudaFvgTsError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        // Conservative param limits must match kernel constants
        const MAX_LOOK: usize = 256;
        const MAX_W: usize = 256;
        const MAX_LOOK: usize = 256;
        const MAX_W: usize = 256;
        for p in &combos {
            let lb = p.unmitigated_fvg_lookback.unwrap_or(5);
            let w = p.smoothing_length.unwrap_or(9);
            if lb == 0 || lb > MAX_LOOK {
                return Err(CudaFvgTsError::InvalidInput(format!(
                    "lookback {} exceeds max {}",
                    lb, MAX_LOOK
                )));
            }
            if w == 0 || w > MAX_W {
                return Err(CudaFvgTsError::InvalidInput(format!(
                    "smoothing_length {} exceeds max {}",
                    w, MAX_W
                )));
            }
            let w = p.smoothing_length.unwrap_or(9);
            if lb == 0 || lb > MAX_LOOK {
                return Err(CudaFvgTsError::InvalidInput(format!(
                    "lookback {} exceeds max {}",
                    lb, MAX_LOOK
                )));
            }
            if w == 0 || w > MAX_W {
                return Err(CudaFvgTsError::InvalidInput(format!(
                    "smoothing_length {} exceeds max {}",
                    w, MAX_W
                )));
            }
        }

        // Upload invariants
        let d_high =
            DeviceBuffer::from_slice(high).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_high =
            DeviceBuffer::from_slice(high).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;

        let nrows = combos.len();
        let mut h_lb: Vec<i32> = Vec::with_capacity(nrows);
        let mut h_sw: Vec<i32> = Vec::with_capacity(nrows);
        let mut h_rs: Vec<i32> = Vec::with_capacity(nrows);
        for p in &combos {
            h_lb.push(p.unmitigated_fvg_lookback.unwrap_or(5) as i32);
            h_sw.push(p.smoothing_length.unwrap_or(9) as i32);
            h_rs.push(if p.reset_on_cross.unwrap_or(false) {
                1
            } else {
                0
            });
        }
        let d_lb =
            DeviceBuffer::from_slice(&h_lb).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_sw =
            DeviceBuffer::from_slice(&h_sw).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_rs =
            DeviceBuffer::from_slice(&h_rs).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        for p in &combos {
            h_lb.push(p.unmitigated_fvg_lookback.unwrap_or(5) as i32);
            h_sw.push(p.smoothing_length.unwrap_or(9) as i32);
            h_rs.push(if p.reset_on_cross.unwrap_or(false) {
                1
            } else {
                0
            });
        }
        let d_lb =
            DeviceBuffer::from_slice(&h_lb).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_sw =
            DeviceBuffer::from_slice(&h_sw).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_rs =
            DeviceBuffer::from_slice(&h_rs).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;

        // Outputs (four matrices: rows=nrows, cols=len)
        let bytes_out = 4usize * nrows * len * std::mem::size_of::<f32>();
        if !Self::will_fit(bytes_out, 64 * 1024 * 1024) {
            return Err(CudaFvgTsError::InvalidInput(
                "insufficient VRAM for outputs".into(),
            ));
        if !Self::will_fit(bytes_out, 64 * 1024 * 1024) {
            return Err(CudaFvgTsError::InvalidInput(
                "insufficient VRAM for outputs".into(),
            ));
        }
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(nrows * len) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(nrows * len) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_upper_ts: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(nrows * len) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_lower_ts: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(nrows * len) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(nrows * len) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(nrows * len) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_upper_ts: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(nrows * len) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_lower_ts: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(nrows * len) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;

        // ---- choose launch + shared-mem heuristic ----
        let mut func = self
            .module
            .get_function("fvg_trailing_stop_batch_f32")
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;

        // Default block size or policy override
        let mut block_x = match self.policy.batch { BatchKernelPolicy::OneD { block_x } => block_x, _ => 128 };

        // Runtime heuristic: use shared rings only if max(w) ≤ 64
        let max_w: usize = h_sw.iter().copied().map(|v| v as i32).max().unwrap_or(0).max(1) as usize;
        let want_shmem = max_w <= 64;

        // Each thread needs 3 * smem_stride floats; we pack per-thread slices back-to-back.
        let smem_stride: usize = if want_shmem { max_w } else { 0 };
        let bytes_per_thread: usize = 3 * smem_stride * std::mem::size_of::<f32>();

        // If using shared memory, clamp block_x so that dynamic shared memory fits per block.
        // If query fails, fall back to ~48KB (typical default without opt-in).
        let mut use_shmem_rings = 0i32;
        let mut dynamic_smem_bytes: usize = 0;

        if want_shmem && bytes_per_thread > 0 {
            let grid_probe: GridSize = (1, 1, 1).into();
            let block_probe: BlockSize = (block_x, 1, 1).into();
            let avail_dyn = func
                .available_dynamic_shared_memory_per_block(grid_probe, block_probe)
                .unwrap_or(48 * 1024);

            let max_threads_by_smem = if bytes_per_thread > 0 {
                (avail_dyn / bytes_per_thread) as u32
            } else { block_x };

            if max_threads_by_smem >= 32 {
                block_x = block_x.min(max_threads_by_smem);
                use_shmem_rings = 1;
                dynamic_smem_bytes = bytes_per_thread * (block_x as usize);
                let _ = func.set_cache_config(CacheConfig::PreferShared);
            } else {
                let _ = func.set_cache_config(CacheConfig::PreferL1);
            }
        } else {
            let _ = func.set_cache_config(CacheConfig::PreferL1);
        }

        // Final launch shape
        let grid_x = (((nrows as u32) + block_x - 1) / block_x).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p_hi = d_high.as_device_ptr().as_raw();
            let mut p_lo = d_low.as_device_ptr().as_raw();
            let mut p_lo = d_low.as_device_ptr().as_raw();
            let mut p_cl = d_close.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut p_lb = d_lb.as_device_ptr().as_raw();
            let mut p_sw = d_sw.as_device_ptr().as_raw();
            let mut p_rs = d_rs.as_device_ptr().as_raw();
            let mut n_i = nrows as i32;
            let mut p_u = d_upper.as_device_ptr().as_raw();
            let mut p_l = d_lower.as_device_ptr().as_raw();
            let mut n_i = nrows as i32;
            let mut p_u = d_upper.as_device_ptr().as_raw();
            let mut p_l = d_lower.as_device_ptr().as_raw();
            let mut p_ut = d_upper_ts.as_device_ptr().as_raw();
            let mut p_lt = d_lower_ts.as_device_ptr().as_raw();
            let mut use_shmem_i = use_shmem_rings as i32;
            let mut smem_stride_i = if use_shmem_rings != 0 { smem_stride as i32 } else { 0i32 };
            let args: &mut [*mut c_void] = &mut [
                &mut p_hi as *mut _ as *mut c_void,
                &mut p_lo as *mut _ as *mut c_void,
                &mut p_cl as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut p_lb as *mut _ as *mut c_void,
                &mut p_sw as *mut _ as *mut c_void,
                &mut p_rs as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut p_u as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut p_u as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_ut as *mut _ as *mut c_void,
                &mut p_lt as *mut _ as *mut c_void,
                &mut use_shmem_i as *mut _ as *mut c_void,
                &mut smem_stride_i as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, dynamic_smem_bytes as u32, args)
                .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;

        Ok(CudaFvgTsBatch {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows: nrows,
                cols: len,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows: nrows,
                cols: len,
            },
            upper_ts: DeviceArrayF32 {
                buf: d_upper_ts,
                rows: nrows,
                cols: len,
            },
            lower_ts: DeviceArrayF32 {
                buf: d_lower_ts,
                rows: nrows,
                cols: len,
            },
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows: nrows,
                cols: len,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows: nrows,
                cols: len,
            },
            upper_ts: DeviceArrayF32 {
                buf: d_upper_ts,
                rows: nrows,
                cols: len,
            },
            lower_ts: DeviceArrayF32 {
                buf: d_lower_ts,
                rows: nrows,
                cols: len,
            },
            combos,
        })
    }

    pub fn fvg_ts_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &FvgTrailingStopParams,
    ) -> Result<
        (
            DeviceArrayF32,
            DeviceArrayF32,
            DeviceArrayF32,
            DeviceArrayF32,
        ),
        CudaFvgTsError,
    > {
        if cols == 0 || rows == 0 {
            return Err(CudaFvgTsError::InvalidInput("cols/rows must be > 0".into()));
        }
        if high_tm.len() != low_tm.len()
            || high_tm.len() != close_tm.len()
            || high_tm.len() != cols * rows
        {
            return Err(CudaFvgTsError::InvalidInput(
                "time-major arrays must match cols*rows".into(),
            ));
    ) -> Result<
        (
            DeviceArrayF32,
            DeviceArrayF32,
            DeviceArrayF32,
            DeviceArrayF32,
        ),
        CudaFvgTsError,
    > {
        if cols == 0 || rows == 0 {
            return Err(CudaFvgTsError::InvalidInput("cols/rows must be > 0".into()));
        }
        if high_tm.len() != low_tm.len()
            || high_tm.len() != close_tm.len()
            || high_tm.len() != cols * rows
        {
            return Err(CudaFvgTsError::InvalidInput(
                "time-major arrays must match cols*rows".into(),
            ));
        }
        let lb = params.unmitigated_fvg_lookback.unwrap_or(5);
        let w = params.smoothing_length.unwrap_or(9);
        const MAX_LOOK: usize = 256;
        const MAX_W: usize = 256;
        if lb == 0 || lb > MAX_LOOK {
            return Err(CudaFvgTsError::InvalidInput("lookback out of range".into()));
        }
        if w == 0 || w > MAX_W {
            return Err(CudaFvgTsError::InvalidInput(
                "smoothing_length out of range".into(),
            ));
        }
        let rst = if params.reset_on_cross.unwrap_or(false) {
            1i32
        } else {
            0i32
        };
        let w = params.smoothing_length.unwrap_or(9);
        const MAX_LOOK: usize = 256;
        const MAX_W: usize = 256;
        if lb == 0 || lb > MAX_LOOK {
            return Err(CudaFvgTsError::InvalidInput("lookback out of range".into()));
        }
        if w == 0 || w > MAX_W {
            return Err(CudaFvgTsError::InvalidInput(
                "smoothing_length out of range".into(),
            ));
        }
        let rst = if params.reset_on_cross.unwrap_or(false) {
            1i32
        } else {
            0i32
        };

        let d_hi =
            DeviceBuffer::from_slice(high_tm).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_lo =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_cl =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_u: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_l: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_ut: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_lt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_hi =
            DeviceBuffer::from_slice(high_tm).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_lo =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let d_cl =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_u: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_l: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_ut: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        let mut d_lt: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;

        let mut func = self.module.get_function("fvg_trailing_stop_many_series_one_param_f32")
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        // Prefer L1 on this path (per-thread local histories)
        let _ = func.set_cache_config(CacheConfig::PreferL1);
        let block_x = match self.policy.many_series { ManySeriesKernelPolicy::OneD{block_x} => block_x, _ => 128 };
        let grid_x = (((cols as u32) + block_x - 1) / block_x).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p_hi = d_hi.as_device_ptr().as_raw();
            let mut p_lo = d_lo.as_device_ptr().as_raw();
            let mut p_cl = d_cl.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut lb_i = lb as i32;
            let mut w_i = w as i32;
            let mut rst_i = rst as i32;
            let mut p_u = d_u.as_device_ptr().as_raw();
            let mut p_l = d_l.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut lb_i = lb as i32;
            let mut w_i = w as i32;
            let mut rst_i = rst as i32;
            let mut p_u = d_u.as_device_ptr().as_raw();
            let mut p_l = d_l.as_device_ptr().as_raw();
            let mut p_ut = d_ut.as_device_ptr().as_raw();
            let mut p_lt = d_lt.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_hi as *mut _ as *mut c_void,
                &mut p_lo as *mut _ as *mut c_void,
                &mut p_cl as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut lb_i as *mut _ as *mut c_void,
                &mut w_i as *mut _ as *mut c_void,
                &mut w_i as *mut _ as *mut c_void,
                &mut rst_i as *mut _ as *mut c_void,
                &mut p_u as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_u as *mut _ as *mut c_void,
                &mut p_l as *mut _ as *mut c_void,
                &mut p_ut as *mut _ as *mut c_void,
                &mut p_lt as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaFvgTsError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_u,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_l,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_ut,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_lt,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_u,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_l,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_ut,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_lt,
                rows,
                cols,
            },
        ))
    }
}

// ---- Lightweight bench profiles ----
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    struct FvgTsBatchState {
        cuda: CudaFvgTs,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: FvgTsBatchRange,
    }
    impl CudaBenchState for FvgTsBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .fvg_ts_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .unwrap();
        }
    }
    struct FvgTsBatchState {
        cuda: CudaFvgTs,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: FvgTsBatchRange,
    }
    impl CudaBenchState for FvgTsBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .fvg_ts_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .unwrap();
        }
    }
    fn prep_batch() -> Box<dyn CudaBenchState> {
        let len = 1_000_000usize;
        let close = gen_series(len);
        let mut high = close.clone();
        let mut low = close.clone();
        for i in 0..len {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.002;
            let off = 0.20 + 0.01 * (x.sin().abs());
            high[i] = v + off;
            low[i] = v - off;
        }
        let sweep = FvgTsBatchRange {
            lookback: (3, 10, 1),
            smoothing: (5, 20, 5),
            reset_on_cross: (true, true),
        };
        Box::new(FvgTsBatchState {
            cuda: CudaFvgTs::new(0).unwrap(),
            high,
            low,
            close,
            sweep,
        })
        let len = 1_000_000usize;
        let close = gen_series(len);
        let mut high = close.clone();
        let mut low = close.clone();
        for i in 0..len {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.002;
            let off = 0.20 + 0.01 * (x.sin().abs());
            high[i] = v + off;
            low[i] = v - off;
        }
        let sweep = FvgTsBatchRange {
            lookback: (3, 10, 1),
            smoothing: (5, 20, 5),
            reset_on_cross: (true, true),
        };
        Box::new(FvgTsBatchState {
            cuda: CudaFvgTs::new(0).unwrap(),
            high,
            low,
            close,
            sweep,
        })
    }

    struct FvgTsManySeriesState {
        cuda: CudaFvgTs,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: FvgTrailingStopParams,
    }
    impl CudaBenchState for FvgTsManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .fvg_ts_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .unwrap();
        }
    }
    struct FvgTsManySeriesState {
        cuda: CudaFvgTs,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: FvgTrailingStopParams,
    }
    impl CudaBenchState for FvgTsManySeriesState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .fvg_ts_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .unwrap();
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cols = 128usize;
        let rows = 1_000_000usize / cols;
        let n = cols * rows;
        let cols = 128usize;
        let rows = 1_000_000usize / cols;
        let n = cols * rows;
        let close = gen_series(n);
        let mut high = close.clone();
        let mut low = close.clone();
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let v = close[idx];
                if v.is_nan() {
                    continue;
                }
                let x = t as f32 * 0.002 + s as f32 * 0.01;
                let off = 0.18 + 0.01 * (x.cos().abs());
                high[idx] = v + off;
                low[idx] = v - off;
            }
        }
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: Some(5),
            smoothing_length: Some(9),
            reset_on_cross: Some(false),
        };
        Box::new(FvgTsManySeriesState {
            cuda: CudaFvgTs::new(0).unwrap(),
            high_tm: high,
            low_tm: low,
            close_tm: close,
            cols,
            rows,
            params,
        })
        let mut high = close.clone();
        let mut low = close.clone();
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let v = close[idx];
                if v.is_nan() {
                    continue;
                }
                let x = t as f32 * 0.002 + s as f32 * 0.01;
                let off = 0.18 + 0.01 * (x.cos().abs());
                high[idx] = v + off;
                low[idx] = v - off;
            }
        }
        let params = FvgTrailingStopParams {
            unmitigated_fvg_lookback: Some(5),
            smoothing_length: Some(9),
            reset_on_cross: Some(false),
        };
        Box::new(FvgTsManySeriesState {
            cuda: CudaFvgTs::new(0).unwrap(),
            high_tm: high,
            low_tm: low,
            close_tm: close,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "fvg_trailing_stop",
                "batch_dev",
                "fvg_trailing_stop_cuda_batch_dev",
                "1m_x_params",
                prep_batch,
            )
            .with_inner_iters(3),
            CudaBenchScenario::new(
                "fvg_trailing_stop",
                "many_series_one_param",
                "fvg_trailing_stop_cuda_many_series_one_param_dev",
                "128x8k",
                prep_many_series,
            )
            .with_inner_iters(3),
            CudaBenchScenario::new(
                "fvg_trailing_stop",
                "batch_dev",
                "fvg_trailing_stop_cuda_batch_dev",
                "1m_x_params",
                prep_batch,
            )
            .with_inner_iters(3),
            CudaBenchScenario::new(
                "fvg_trailing_stop",
                "many_series_one_param",
                "fvg_trailing_stop_cuda_many_series_one_param_dev",
                "128x8k",
                prep_many_series,
            )
            .with_inner_iters(3),
        ]
    }
}
