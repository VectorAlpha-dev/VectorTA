//! CUDA scaffolding for the Fractal Adaptive Moving Average (FRAMA).
//!
//! Aligned with the ALMA integration:
//! - Policy-based kernel selection with introspection (record last choice).
//! - PTX JIT with DetermineTargetFromContext and stable O2 fallback.
//! - VRAM-first checks and simple grid chunking to respect launch limits.
//! - Warmup/NaN handling and semantics match the scalar reference.
//!
//! Kernels expected (recurrence/time-marching; one thread per combo/series):
//! - "frama_batch_f32"                   
//! - "frama_many_series_one_param_f32"   

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
use thiserror::Error;
use cust::error::CudaError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const FRAMA_MAX_WINDOW: usize = 1024;



#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    
    OneD { block_x: u32 },
    
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



#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Debug, Error)]
pub enum CudaFramaError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Out of memory on device: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch configuration too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buf on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Arithmetic overflow while computing {context}")]
    ArithmeticOverflow { context: &'static str },
    #[error("Not implemented")]
    NotImplemented,
}

fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
    let mut v = Vec::new();
    let mut x = lo;
    loop {
        v.push(x);
        match x.checked_add(step) {
            Some(nx) if nx <= hi => x = nx,
            _ => break,
        }
    }
    if start > end { v.reverse(); }
    v
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
    ctx: Arc<Context>,
    device_id: u32,
    policy: CudaFramaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaFrama {
    pub fn new(device_id: usize) -> Result<Self, CudaFramaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/frama_kernel.ptx"));
        // Keep original O4 perf default, but fall back progressively for driver/toolkit quirks.
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
            ],
        )
        .or_else(|_| {
            Module::from_ptx(
                ptx,
                &[
                    ModuleJitOption::DetermineTargetFromContext,
                    ModuleJitOption::OptLevel(OptLevel::O2),
                ],
            )
        })
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            ctx: Arc::new(context),
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
    pub fn ctx(&self) -> Arc<Context> { Arc::clone(&self.ctx) }

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
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaFramaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaFramaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
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
            .map_err(|_| CudaFramaError::MissingKernelSymbol { name: "frama_batch_f32" })?;

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

                self.stream.launch(&func, grid, block, 0, args)?;
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

                    self.stream.launch(&func, grid, block, 0, args)?;
                }
                start += len;
            }
        }

        self.stream.synchronize()?;

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
        let prices_bytes = len
            .checked_mul(3)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaFramaError::ArithmeticOverflow { context: "prices bytes" })?;
        let params_bytes = combos.len()
            .checked_mul(3)
            .and_then(|x| x.checked_mul(std::mem::size_of::<i32>()))
            .ok_or(CudaFramaError::ArithmeticOverflow { context: "params bytes" })?;
        let out_bytes = len
            .checked_mul(combos.len())
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaFramaError::ArithmeticOverflow { context: "output bytes" })?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or(CudaFramaError::ArithmeticOverflow { context: "total bytes" })?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit_checked(required, headroom)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;

        let windows: Vec<i32> = combos.iter().map(|c| c.window.unwrap() as i32).collect();
        let scs: Vec<i32> = combos.iter().map(|c| c.sc.unwrap() as i32).collect();
        let fcs: Vec<i32> = combos.iter().map(|c| c.fc.unwrap() as i32).collect();

        let d_windows = DeviceBuffer::from_slice(&windows)?;
        let d_scs = DeviceBuffer::from_slice(&scs)?;
        let d_fcs = DeviceBuffer::from_slice(&fcs)?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(combos.len() * len) }?;

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
        let d_windows = DeviceBuffer::from_slice(&windows)?;
        let d_scs = DeviceBuffer::from_slice(&scs)?;
        let d_fcs = DeviceBuffer::from_slice(&fcs)?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(combos.len() * series_len) }?;

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
        dev.buf.copy_to(out)?;
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
        let mut d_high = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let mut d_low = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;
        let mut d_close = unsafe { DeviceBuffer::<f32>::uninitialized(len) }?;

        unsafe {
            d_high.async_copy_from(high_locked.as_slice(), &self.stream)?;
            d_low.async_copy_from(low_locked.as_slice(), &self.stream)?;
            d_close.async_copy_from(close_locked.as_slice(), &self.stream)?;
        }

        let windows: Vec<i32> = combos.iter().map(|c| c.window.unwrap() as i32).collect();
        let scs: Vec<i32> = combos.iter().map(|c| c.sc.unwrap() as i32).collect();
        let fcs: Vec<i32> = combos.iter().map(|c| c.fc.unwrap() as i32).collect();
        let d_windows = DeviceBuffer::from_slice(&windows)?;
        let d_scs = DeviceBuffer::from_slice(&scs)?;
        let d_fcs = DeviceBuffer::from_slice(&fcs)?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(expected) }?;

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
            d_out.async_copy_to(out_locked.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;

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
            .map_err(|_| CudaFramaError::MissingKernelSymbol { name: "frama_many_series_one_param_f32" })?;

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
            return Err(CudaFramaError::LaunchConfigTooLarge { gx: max_grid_x as u32, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
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

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        // Ensure completion for consistent timing in benches/tests
        self.stream.synchronize().map_err(Into::into)
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
        let elems = cols
            .checked_mul(rows)
            .ok_or(CudaFramaError::ArithmeticOverflow { context: "cols*rows" })?;
        let prices_bytes = elems
            .checked_mul(3)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaFramaError::ArithmeticOverflow { context: "prices bytes" })?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaFramaError::ArithmeticOverflow { context: "output bytes" })?;
        let first_valids_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(CudaFramaError::ArithmeticOverflow { context: "first_valids bytes" })?;
        let required = prices_bytes
            .checked_add(out_bytes)
            .and_then(|x| x.checked_add(first_valids_bytes))
            .ok_or(CudaFramaError::ArithmeticOverflow { context: "total bytes" })?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit_checked(required, headroom)?;

        let d_high = DeviceBuffer::from_slice(high_tm)?;
        let d_low = DeviceBuffer::from_slice(low_tm)?;
        let d_close = DeviceBuffer::from_slice(close_tm)?;
        let d_first = DeviceBuffer::from_slice(first_valids)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }?;

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

        let d_first = DeviceBuffer::from_slice(first_valids)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }?;

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
        dev.buf.copy_to(out_tm).map_err(Into::into)
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

    struct FramaBatchDevState {
        cuda: CudaFrama,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_windows: DeviceBuffer<i32>,
        d_scs: DeviceBuffer<i32>,
        d_fcs: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for FramaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_high,
                    &self.d_low,
                    &self.d_close,
                    &self.d_windows,
                    &self.d_scs,
                    &self.d_fcs,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("frama batch kernel");
            self.cuda.stream.synchronize().expect("frama sync");
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
        let (combos, first_valid, series_len) =
            CudaFrama::prepare_batch_inputs(&high, &low, &close, &sweep).expect("frama prepare batch");
        let n_combos = combos.len();
        let windows_i32: Vec<i32> = combos.iter().map(|p| p.window.unwrap_or(0) as i32).collect();
        let scs_i32: Vec<i32> = combos.iter().map(|p| p.sc.unwrap_or(0) as i32).collect();
        let fcs_i32: Vec<i32> = combos.iter().map(|p| p.fc.unwrap_or(0) as i32).collect();

        let d_high = DeviceBuffer::from_slice(&high).expect("d_high");
        let d_low = DeviceBuffer::from_slice(&low).expect("d_low");
        let d_close = DeviceBuffer::from_slice(&close).expect("d_close");
        let d_windows = DeviceBuffer::from_slice(&windows_i32).expect("d_windows");
        let d_scs = DeviceBuffer::from_slice(&scs_i32).expect("d_scs");
        let d_fcs = DeviceBuffer::from_slice(&fcs_i32).expect("d_fcs");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len.checked_mul(n_combos).expect("out size"))
        }
        .expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(FramaBatchDevState {
            cuda,
            d_high,
            d_low,
            d_close,
            d_windows,
            d_scs,
            d_fcs,
            series_len,
            n_combos,
            first_valid,
            d_out,
        })
    }

    struct FramaManyDevState {
        cuda: CudaFrama,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_close_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        window: i32,
        sc: i32,
        fc: i32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for FramaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_high_tm,
                    &self.d_low_tm,
                    &self.d_close_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.window,
                    self.sc,
                    self.fc,
                    &mut self.d_out_tm,
                )
                .expect("frama many-series kernel");
            self.cuda.stream.synchronize().expect("frama many-series sync");
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
        let (first_valids, _even, window, sc, fc) =
            CudaFrama::prepare_many_series_inputs(&high_tm, &low_tm, &close_tm, cols, rows, &params)
                .expect("frama prepare many");

        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_close_tm = DeviceBuffer::from_slice(&close_tm).expect("d_close_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols.checked_mul(rows).expect("out size")) }
                .expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(FramaManyDevState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_close_tm,
            d_first_valids,
            cols,
            rows,
            window,
            sc,
            fc,
            d_out_tm,
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
