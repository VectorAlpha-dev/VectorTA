//! CUDA scaffolding for the Fractal Adaptive Moving Average (FRAMA).
//!
//! Aligned with the ALMA integration:
//! - Policy-based kernel selection with introspection (record last choice).
//! - PTX JIT with DetermineTargetFromContext and stable O2 fallback.
//! - VRAM-first checks and simple grid chunking to respect launch limits.
//! - Warmup/NaN handling and semantics match the scalar reference.
//!
//! Kernels expected (recurrence/time-marching; one thread per combo/series):
//! - "frama_batch_f32"                   // one-series × many-params
//! - "frama_many_series_one_param_f32"   // many-series × one-param (time-major)

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::frama::{FramaBatchRange, FramaParams};
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

const FRAMA_MAX_WINDOW: usize = 1024;

// -------- Kernel selection policy (keep simple for recurrence kernels) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    // Each thread processes one combo across time. Tunable block_x for occupancy.
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    // One thread per series (time-major). Tunable block_x.
    OneD { block_x: u32 },
    // Placeholder for API symmetry with ALMA; falls back to OneD.
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaFramaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaFramaPolicy {
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
pub enum CudaFramaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaFramaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaFramaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaFramaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaFramaError {}

fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}

fn evenize(window: usize) -> usize {
    if window & 1 == 1 {
        window + 1
    } else {
        window
    }
}

fn expand_grid(range: &FramaBatchRange) -> Vec<FramaParams> {
    let windows = axis_usize(range.window);
    let scs = axis_usize(range.sc);
    let fcs = axis_usize(range.fc);
    let mut out = Vec::with_capacity(windows.len() * scs.len() * fcs.len());
    for &w in &windows {
        for &s in &scs {
            for &f in &fcs {
                out.push(FramaParams {
                    window: Some(w),
                    sc: Some(s),
                    fc: Some(f),
                });
            }
        }
    }
    out
}

fn first_valid_index(high: &[f32], low: &[f32], close: &[f32]) -> Option<usize> {
    for idx in 0..high.len() {
        if !high[idx].is_nan() && !low[idx].is_nan() && !close[idx].is_nan() {
            return Some(idx);
        }
    }
    None
}

pub struct CudaFrama {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaFramaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaFrama {
    pub fn new(device_id: usize) -> Result<Self, CudaFramaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/frama_kernel.ptx"));
        // Prefer most-optimized JIT level (O4) and still fall back if needed.
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaFramaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaFramaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaFramaPolicy,
    ) -> Result<Self, CudaFramaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaFramaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaFramaPolicy {
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
                    eprintln!("[DEBUG] FRAMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaFrama)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] FRAMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaFrama)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------- VRAM helpers ----------
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
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &FramaBatchRange,
    ) -> Result<(Vec<FramaParams>, usize, usize), CudaFramaError> {
        if high.is_empty() {
            return Err(CudaFramaError::InvalidInput("empty input".into()));
        }
        if low.len() != high.len() || close.len() != high.len() {
            return Err(CudaFramaError::InvalidInput(format!(
                "mismatched slice lengths: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaFramaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = high.len();
        let first_valid = first_valid_index(high, low, close)
            .ok_or_else(|| CudaFramaError::InvalidInput("all values are NaN".into()))?;

        let mut max_even = 0usize;
        for combo in &combos {
            let window = combo.window.unwrap_or(0);
            let sc = combo.sc.unwrap_or(0);
            let fc = combo.fc.unwrap_or(0);
            if window == 0 {
                return Err(CudaFramaError::InvalidInput(
                    "window must be greater than zero".into(),
                ));
            }
            if window > len {
                return Err(CudaFramaError::InvalidInput(format!(
                    "window {} exceeds data length {}",
                    window, len
                )));
            }
            if sc == 0 {
                return Err(CudaFramaError::InvalidInput(
                    "sc smoothing constant must be greater than zero".into(),
                ));
            }
            if fc == 0 {
                return Err(CudaFramaError::InvalidInput(
                    "fc smoothing constant must be greater than zero".into(),
                ));
            }
            let even = evenize(window);
            if even > FRAMA_MAX_WINDOW {
                return Err(CudaFramaError::InvalidInput(format!(
                    "evenized window {} exceeds CUDA limit {}",
                    even, FRAMA_MAX_WINDOW
                )));
            }
            if len - first_valid < even {
                return Err(CudaFramaError::InvalidInput(format!(
                    "not enough valid data: need >= {}, have {}",
                    even,
                    len - first_valid
                )));
            }
            max_even = max_even.max(even);
        }

        if max_even == 0 {
            return Err(CudaFramaError::InvalidInput(
                "invalid parameter grid (zero window)".into(),
            ));
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_windows: &DeviceBuffer<i32>,
        d_scs: &DeviceBuffer<i32>,
        d_fcs: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFramaError> {
        let func = self
            .module
            .get_function("frama_batch_f32")
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        // Auto policy picks occupancy-suggested block size; env override stays.
        let auto_block_x: u32 = match func.suggested_launch_configuration(0, (1024, 1, 1).into()) {
            Ok((_min_grid, suggested_block)) => suggested_block,
            Err(_) => 256,
        };
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => std::env::var("FRAMA_BLOCK_X")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(auto_block_x),
        };

        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaFrama;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Compute grid size; no artificial 65,535 cap for x-dimension.
        let total_blocks_u64 = ((n_combos as u64) + (block_x as u64) - 1) / (block_x as u64);
        let max_grid_x = 2_147_483_647u64; // 2^31 - 1 per CUDA spec

        if total_blocks_u64 <= max_grid_x {
            let grid: GridSize = ((total_blocks_u64 as u32).max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut high_ptr = d_high.as_device_ptr().as_raw();
                let mut low_ptr = d_low.as_device_ptr().as_raw();
                let mut close_ptr = d_close.as_device_ptr().as_raw();
                let mut win_ptr = d_windows.as_device_ptr().as_raw();
                let mut sc_ptr = d_scs.as_device_ptr().as_raw();
                let mut fc_ptr = d_fcs.as_device_ptr().as_raw();
                let mut len_i = series_len as i32;
                let mut combos_i = n_combos as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut win_ptr as *mut _ as *mut c_void,
                    &mut sc_ptr as *mut _ as *mut c_void,
                    &mut fc_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
            }
        } else {
            // Extremely rare: chunk only if grid.x would overflow the limit.
            let max_blocks_per_launch = max_grid_x as usize;
            let mut start = 0usize;
            while start < n_combos {
                let len = (n_combos - start).min(max_blocks_per_launch * (block_x as usize));
                let blocks = ((len as u32) + block_x - 1) / block_x;
                let grid: GridSize = (blocks.max(1), 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();

                unsafe {
                    let mut high_ptr = d_high.as_device_ptr().as_raw();
                    let mut low_ptr = d_low.as_device_ptr().as_raw();
                    let mut close_ptr = d_close.as_device_ptr().as_raw();
                    let mut win_ptr = d_windows.as_device_ptr().add(start).as_raw();
                    let mut sc_ptr = d_scs.as_device_ptr().add(start).as_raw();
                    let mut fc_ptr = d_fcs.as_device_ptr().add(start).as_raw();
                    let mut len_i = series_len as i32;
                    let mut combos_i = len as i32;
                    let mut first_valid_i = first_valid as i32;
                    let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();

                    let args: &mut [*mut c_void] = &mut [
                        &mut high_ptr as *mut _ as *mut c_void,
                        &mut low_ptr as *mut _ as *mut c_void,
                        &mut close_ptr as *mut _ as *mut c_void,
                        &mut win_ptr as *mut _ as *mut c_void,
                        &mut sc_ptr as *mut _ as *mut c_void,
                        &mut fc_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut combos_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];

                    self.stream
                        .launch(&func, grid, block, 0, args)
                        .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
                }
                start += len;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        combos: &[FramaParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaFramaError> {
        // VRAM estimate (inputs + params + outputs) + headroom
        let prices_bytes = len * 3 * std::mem::size_of::<f32>();
        let params_bytes = combos.len() * 3 * std::mem::size_of::<i32>();
        let out_bytes = len * combos.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaFramaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_high =
            DeviceBuffer::from_slice(high).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let windows: Vec<i32> = combos.iter().map(|c| c.window.unwrap() as i32).collect();
        let scs: Vec<i32> = combos.iter().map(|c| c.sc.unwrap() as i32).collect();
        let fcs: Vec<i32> = combos.iter().map(|c| c.fc.unwrap() as i32).collect();

        let d_windows =
            DeviceBuffer::from_slice(&windows).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_scs =
            DeviceBuffer::from_slice(&scs).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_fcs =
            DeviceBuffer::from_slice(&fcs).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(combos.len() * len) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            &d_close,
            &d_windows,
            &d_scs,
            &d_fcs,
            len,
            combos.len(),
            first_valid,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn frama_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &FramaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<FramaParams>), CudaFramaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(high, low, close, sweep)?;
        let dev = self.run_batch_kernel(high, low, close, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    // Device-resident fast path (one series × many params): reuse device price buffers.
    pub fn frama_batch_dev_from_device(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        sweep: &FramaBatchRange,
        series_len: usize,
        first_valid: usize,
    ) -> Result<(DeviceArrayF32, Vec<FramaParams>), CudaFramaError> {
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaFramaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let windows: Vec<i32> = combos.iter().map(|c| c.window.unwrap() as i32).collect();
        let scs: Vec<i32> = combos.iter().map(|c| c.sc.unwrap() as i32).collect();
        let fcs: Vec<i32> = combos.iter().map(|c| c.fc.unwrap() as i32).collect();
        let d_windows =
            DeviceBuffer::from_slice(&windows).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_scs =
            DeviceBuffer::from_slice(&scs).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_fcs =
            DeviceBuffer::from_slice(&fcs).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(combos.len() * series_len) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            d_high,
            d_low,
            d_close,
            &d_windows,
            &d_scs,
            &d_fcs,
            series_len,
            combos.len(),
            first_valid,
            &mut d_out,
        )?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: series_len,
            },
            combos,
        ))
    }

    pub fn frama_batch_into_host_f32(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &FramaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<FramaParams>), CudaFramaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(high, low, close, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaFramaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(high, low, close, &combos, first_valid, len)?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        Ok((dev.rows, dev.cols, combos))
    }

    // Optional pinned/async fast path using the same stream; still synchronizes before return.
    pub fn frama_batch_into_host_f32_pinned(
        &self,
        high_locked: &LockedBuffer<f32>,
        low_locked: &LockedBuffer<f32>,
        close_locked: &LockedBuffer<f32>,
        sweep: &FramaBatchRange,
        out_locked: &mut LockedBuffer<f32>,
    ) -> Result<(usize, usize, Vec<FramaParams>), CudaFramaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(
            high_locked.as_slice(),
            low_locked.as_slice(),
            close_locked.as_slice(),
            sweep,
        )?;
        let expected = combos.len() * len;
        if out_locked.len() != expected {
            return Err(CudaFramaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out_locked.len()
            )));
        }

        // Device inputs
        let mut d_high = unsafe { DeviceBuffer::<f32>::uninitialized(len) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let mut d_low = unsafe { DeviceBuffer::<f32>::uninitialized(len) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let mut d_close = unsafe { DeviceBuffer::<f32>::uninitialized(len) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        unsafe {
            d_high
                .async_copy_from(high_locked.as_slice(), &self.stream)
                .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
            d_low
                .async_copy_from(low_locked.as_slice(), &self.stream)
                .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
            d_close
                .async_copy_from(close_locked.as_slice(), &self.stream)
                .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        }

        let windows: Vec<i32> = combos.iter().map(|c| c.window.unwrap() as i32).collect();
        let scs: Vec<i32> = combos.iter().map(|c| c.sc.unwrap() as i32).collect();
        let fcs: Vec<i32> = combos.iter().map(|c| c.fc.unwrap() as i32).collect();
        let d_windows =
            DeviceBuffer::from_slice(&windows).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_scs =
            DeviceBuffer::from_slice(&scs).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_fcs =
            DeviceBuffer::from_slice(&fcs).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(expected) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            &d_close,
            &d_windows,
            &d_scs,
            &d_fcs,
            len,
            combos.len(),
            first_valid,
            &mut d_out,
        )?;

        unsafe {
            d_out
                .async_copy_to(out_locked.as_mut_slice(), &self.stream)
                .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &FramaParams,
    ) -> Result<(Vec<i32>, usize, i32, i32, i32), CudaFramaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaFramaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        let expected = cols * rows;
        if high_tm.len() != expected || low_tm.len() != expected || close_tm.len() != expected {
            return Err(CudaFramaError::InvalidInput(format!(
                "time-major buffer mismatch: expected {}, got high={}, low={}, close={}",
                expected,
                high_tm.len(),
                low_tm.len(),
                close_tm.len()
            )));
        }

        let window = params.window.ok_or_else(|| {
            CudaFramaError::InvalidInput("window parameter must be provided".into())
        })?;
        let sc = params
            .sc
            .ok_or_else(|| CudaFramaError::InvalidInput("sc parameter must be provided".into()))?;
        let fc = params
            .fc
            .ok_or_else(|| CudaFramaError::InvalidInput("fc parameter must be provided".into()))?;

        if window == 0 {
            return Err(CudaFramaError::InvalidInput(
                "window must be greater than zero".into(),
            ));
        }
        if sc == 0 {
            return Err(CudaFramaError::InvalidInput(
                "sc smoothing constant must be greater than zero".into(),
            ));
        }
        if fc == 0 {
            return Err(CudaFramaError::InvalidInput(
                "fc smoothing constant must be greater than zero".into(),
            ));
        }

        let even = evenize(window);
        if even > FRAMA_MAX_WINDOW {
            return Err(CudaFramaError::InvalidInput(format!(
                "evenized window {} exceeds CUDA limit {}",
                even, FRAMA_MAX_WINDOW
            )));
        }
        if even > rows {
            return Err(CudaFramaError::InvalidInput(format!(
                "window {} exceeds series length {}",
                even, rows
            )));
        }

        let stride = cols;
        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut first = None;
            for row in 0..rows {
                let idx = row * stride + series;
                if !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan() {
                    first = Some(row);
                    break;
                }
            }
            let fv = first.ok_or_else(|| {
                CudaFramaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < even {
                return Err(CudaFramaError::InvalidInput(format!(
                    "series {} lacks sufficient tail length: need >= {}, have {}",
                    series,
                    even,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, even, window as i32, sc as i32, fc as i32))
    }

    fn launch_many_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
        window: i32,
        sc: i32,
        fc: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaFramaError> {
        let func = self
            .module
            .get_function("frama_many_series_one_param_f32")
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        // Auto policy picks occupancy-suggested block size; env override stays.
        let auto_block_x: u32 = match func.suggested_launch_configuration(0, (1024, 1, 1).into()) {
            Ok((_min_grid, suggested_block)) => suggested_block,
            Err(_) => 128,
        };
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Tiled2D { tx, .. } => tx, // fall back to OneD geometry
            ManySeriesKernelPolicy::Auto => std::env::var("FRAMA_MS1P_BLOCK_X")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(auto_block_x),
        };
        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaFrama;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let total_blocks_u64 = ((num_series as u64) + (block_x as u64) - 1) / (block_x as u64);
        let max_grid_x = 2_147_483_647u64; // 2^31 - 1
        if total_blocks_u64 > max_grid_x {
            return Err(CudaFramaError::InvalidInput(format!(
                "too many series for one launch: need {} blocks, max {}",
                total_blocks_u64, max_grid_x
            )));
        }
        let grid: GridSize = ((total_blocks_u64 as u32).max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut window_i = window;
            let mut sc_i = sc;
            let mut fc_i = fc;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut window_i as *mut _ as *mut c_void,
                &mut sc_i as *mut _ as *mut c_void,
                &mut fc_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        }

        // Ensure completion for consistent timing in benches/tests
        self.stream
            .synchronize()
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))
    }

    fn run_many_series_kernel(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        window: i32,
        sc: i32,
        fc: i32,
    ) -> Result<DeviceArrayF32, CudaFramaError> {
        // VRAM estimate
        let elems = cols * rows;
        let prices_bytes = elems * 3 * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let first_valids_bytes = cols * std::mem::size_of::<i32>();
        let required = prices_bytes + out_bytes + first_valids_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaFramaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_high =
            DeviceBuffer::from_slice(high_tm).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_low =
            DeviceBuffer::from_slice(low_tm).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_close =
            DeviceBuffer::from_slice(close_tm).map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_high, &d_low, &d_close, &d_first, cols, rows, window, sc, fc, &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn frama_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &FramaParams,
    ) -> Result<DeviceArrayF32, CudaFramaError> {
        let (first_valids, _even_window, window_i, sc_i, fc_i) =
            Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows, params)?;
        self.run_many_series_kernel(
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            &first_valids,
            window_i,
            sc_i,
            fc_i,
        )
    }

    // Device-resident fast path: prices and first_valids already in VRAM; avoid host copies.
    pub fn frama_many_series_one_param_time_major_dev_from_device(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        params: &FramaParams,
    ) -> Result<DeviceArrayF32, CudaFramaError> {
        let window = params.window.ok_or_else(|| {
            CudaFramaError::InvalidInput("window parameter must be provided".into())
        })? as i32;
        let sc = params
            .sc
            .ok_or_else(|| CudaFramaError::InvalidInput("sc parameter must be provided".into()))?
            as i32;
        let fc = params
            .fc
            .ok_or_else(|| CudaFramaError::InvalidInput("fc parameter must be provided".into()))?
            as i32;

        let d_first = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            d_high_tm, d_low_tm, d_close_tm, &d_first, cols, rows, window, sc, fc, &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn frama_many_series_one_param_time_major_into_host_f32(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &FramaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaFramaError> {
        let expected = cols * rows;
        if out_tm.len() != expected {
            return Err(CudaFramaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out_tm.len()
            )));
        }
        let (first_valids, _even_window, window_i, sc_i, fc_i) =
            Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            &first_valids,
            window_i,
            sc_i,
            fc_i,
        )?;
        dev.buf
            .copy_to(out_tm)
            .map_err(|e| CudaFramaError::Cuda(e.to_string()))
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * 3 * std::mem::size_of::<f32>(); // high/low/close
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * 3 * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn make_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0021;
            let off = (0.003 * x.sin()).abs() + 0.2;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    fn make_hlc_tm_from_close(close_tm: &[f32], cols: usize, rows: usize) -> (Vec<f32>, Vec<f32>) {
        let mut high = close_tm.to_vec();
        let mut low = close_tm.to_vec();
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let v = close_tm[idx];
                if v.is_nan() {
                    continue;
                }
                let x = (row as f32) * 0.0017 + (col as f32) * 0.01;
                let off = (0.0033 * x.cos()).abs() + 0.18;
                high[idx] = v + off;
                low[idx] = v - off;
            }
        }
        (high, low)
    }

    struct FramaBatchState {
        cuda: CudaFrama,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
        sweep: FramaBatchRange,
    }
    impl CudaBenchState for FramaBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .frama_batch_dev(&self.high, &self.low, &self.close, &self.sweep)
                .expect("frama batch launch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaFrama::new(0).expect("cuda frama");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = make_hlc_from_close(&close);
        let sweep = FramaBatchRange {
            window: (10, 10 + PARAM_SWEEP - 1, 1),
            sc: (300, 300, 0),
            fc: (1, 1, 0),
        };
        Box::new(FramaBatchState {
            cuda,
            high,
            low,
            close,
            sweep,
        })
    }

    struct FramaManyState {
        cuda: CudaFrama,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: FramaParams,
    }
    impl CudaBenchState for FramaManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .frama_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("frama many-series launch");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaFrama::new(0).expect("cuda frama");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let close_tm = gen_time_major_prices(cols, rows);
        let (high_tm, low_tm) = make_hlc_tm_from_close(&close_tm, cols, rows);
        let params = FramaParams {
            window: Some(64),
            sc: Some(300),
            fc: Some(1),
        };
        Box::new(FramaManyState {
            cuda,
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "frama",
                "one_series_many_params",
                "frama_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "frama",
                "many_series_one_param",
                "frama_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
