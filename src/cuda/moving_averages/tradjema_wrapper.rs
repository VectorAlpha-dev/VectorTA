//! CUDA wrapper for the TrAdjEMA (Trend Adjusted EMA) kernels.
//!
//! Parity goals with ALMA wrapper:
//! - PTX load with target-from-context and OptLevel O2, with relaxed fallbacks
//! - NON_BLOCKING stream
//! - VRAM estimates + headroom checks; safe grid sizing
//! - Warmup/NaN semantics match scalar (prefix NaNs until warm)
//! - Minimal policy + introspection for kernel selection (OneD with block_x)
//!
//! Kernels expected:
//! - `tradjema_batch_f32`                          // one-series × many-params (OHLC)
//! - `tradjema_many_series_one_param_time_major_f32` // many-series × one-param (time-major OHLC)

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::tradjema::{TradjemaBatchRange, TradjemaParams};
use cust::context::{Context, SharedMemoryConfig};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::AsyncCopyDestination;
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub enum CudaTradjemaError {
    Cuda(cust::error::CudaError),
    InvalidInput(String),
    InvalidPolicy(&'static str),
    OutOfMemory { required: usize, free: usize, headroom: usize },
    MissingKernelSymbol { name: &'static str },
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    DeviceMismatch { buf: u32, current: u32 },
    NotImplemented,
}

impl fmt::Display for CudaTradjemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTradjemaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTradjemaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            CudaTradjemaError::InvalidPolicy(p) => write!(f, "Invalid policy: {}", p),
            CudaTradjemaError::OutOfMemory { required, free, headroom } => write!(
                f,
                "out of memory: required={} free={} headroom={}",
                required, free, headroom
            ),
            CudaTradjemaError::MissingKernelSymbol { name } => {
                write!(f, "missing kernel symbol: {}", name)
            }
            CudaTradjemaError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz } => write!(
                f,
                "launch config too large: grid=({},{},{}) block=({},{},{})",
                gx, gy, gz, bx, by, bz
            ),
            CudaTradjemaError::DeviceMismatch { buf, current } => write!(
                f,
                "device mismatch: buf={} current={}",
                buf, current
            ),
            CudaTradjemaError::NotImplemented => write!(f, "not implemented"),
        }
    }
}

impl std::error::Error for CudaTradjemaError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaTradjemaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaTradjema {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaTradjemaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaTradjema {
    pub fn new(device_id: usize) -> Result<Self, CudaTradjemaError> {
        cust::init(CudaFlags::empty()).map_err(CudaTradjemaError::Cuda)?;
        let device = Device::get_device(device_id as u32).map_err(CudaTradjemaError::Cuda)?;
        let context = Context::new(device).map_err(CudaTradjemaError::Cuda)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/tradjema_kernel.ptx"));
        // Prefer target-from-context + O2, then relax
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[]).map_err(CudaTradjemaError::Cuda)?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(CudaTradjemaError::Cuda)?;

        Ok(Self {
            module,
            stream,
            _context: Arc::new(context),
            device_id: device_id as u32,
            policy: CudaTradjemaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaTradjemaPolicy,
    ) -> Result<Self, CudaTradjemaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaTradjemaPolicy) {
        self.policy = policy;
    }
    pub fn context_arc(&self) -> Arc<Context> { self._context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }
    pub fn policy(&self) -> &CudaTradjemaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaTradjemaError> {
        self.stream
            .synchronize()
            .map_err(CudaTradjemaError::Cuda)
    }

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
    fn smem_bytes_for_len(length: usize) -> usize {
        // Matches CUDA kernel shared layout:
        // Batch uses f32 shared arrays; many-series uses f64.
        // Size for f64 is a safe upper bound for both.
        // [ f64 min_vals[length] ][ f64 max_vals[length] ][ i32 min_idx[length] ][ i32 max_idx[length] ]
        length
            * (2 * std::mem::size_of::<f64>() + 2 * std::mem::size_of::<i32>())
    }

    #[inline]
    fn validate_launch(device_id: u32, block_x: u32, grid_x: u32) -> Result<(), CudaTradjemaError> {
        let dev = Device::get_device(device_id).map_err(CudaTradjemaError::Cuda)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .map_err(CudaTradjemaError::Cuda)? as u32;
        let max_grid_x = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(CudaTradjemaError::Cuda)? as u32;
        if block_x == 0 || block_x > max_threads || grid_x > max_grid_x {
            return Err(CudaTradjemaError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        Ok(())
    }

    fn expand_range(range: &TradjemaBatchRange) -> Vec<TradjemaParams> {
        let (ls, le, lstep) = range.length;
        let (ms, me, mstep) = range.mult;

        #[inline]
        fn axis_usize(start: usize, end: usize, step: usize) -> Vec<usize> {
            if step == 0 { return vec![start]; }
            let mut vals = Vec::new();
            if start <= end {
                let mut v = start;
                while v <= end {
                    vals.push(v);
                    match v.checked_add(step) { Some(n) if n > v => v = n, _ => break }
                }
            } else {
                let mut v = start;
                loop {
                    vals.push(v);
                    if v <= end { break; }
                    v = v.saturating_sub(step);
                    if v < end { break; }
                }
            }
            vals
        }

        #[inline]
        fn axis_f64(start: f64, end: f64, step: f64) -> Vec<f64> {
            if step == 0.0 { return vec![start]; }
            let mut vals = Vec::new();
            if start <= end {
                let mut v = start;
                while v <= end { vals.push(v); v += step; if !v.is_finite() { break; } }
            } else {
                let mut v = start;
                while v >= end { vals.push(v); v -= step.abs(); if !v.is_finite() { break; } }
            }
            vals
        }

        let lengths = axis_usize(ls, le, lstep);
        let mults = axis_f64(ms, me, mstep);
        if lengths.is_empty() || mults.is_empty() { return Vec::new(); }
        let mut combos = Vec::with_capacity(lengths.len().saturating_mul(mults.len()));
        for &l in &lengths { for &m in &mults { combos.push(TradjemaParams { length: Some(l), mult: Some(m) }); } }
        combos
    }

    fn prepare_batch_inputs(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &TradjemaBatchRange,
    ) -> Result<(Vec<TradjemaParams>, usize, usize, usize), CudaTradjemaError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaTradjemaError::InvalidInput("empty OHLC data".into()));
        }
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "OHLC length mismatch: h={}, l={}, c={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaTradjemaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = close.len();
        let first_valid = close
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTradjemaError::InvalidInput("all close values are NaN".into()))?;

        let mut max_length = 0usize;
        for prm in &combos {
            let length = prm.length.unwrap_or(0);
            let mult = prm.mult.unwrap_or(0.0) as f32;
            if length < 2 || length > series_len {
                return Err(CudaTradjemaError::InvalidInput(format!(
                    "invalid length {} (series len {})",
                    length, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid < length {
                return Err(CudaTradjemaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    length, valid
                )));
            }
            if !mult.is_finite() || mult <= 0.0f32 {
                return Err(CudaTradjemaError::InvalidInput(format!(
                    "invalid mult {}",
                    prm.mult.unwrap_or(0.0)
                )));
            }
            max_length = max_length.max(length);
        }

        Ok((combos, first_valid, series_len, max_length))
    }

    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_s = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_s || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] TRADJEMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTradjema)).debug_batch_logged = true;
                }
            }
        }
    }

    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_s = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_s || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] TRADJEMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTradjema)).debug_many_logged = true;
                }
            }
        }
    }

    fn choose_batch_block_x(&self) -> u32 {
        match self.policy.batch {
            BatchKernelPolicy::Auto => std::env::var("TRADJEMA_BLOCK_X")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&bx| bx >= 1 && bx <= 1024)
                .unwrap_or(32),
            BatchKernelPolicy::OneD { block_x } => block_x.clamp(1, 1024),
        }
    }

    fn choose_many_block_x(&self) -> u32 {
        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.clamp(1, 1024),
        }
    }

    fn launch_batch_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_mults: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_length: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTradjemaError> {
        // Basic bounds for i32 kernel args
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || first_valid > i32::MAX as usize
            || max_length > i32::MAX as usize
        {
            return Err(CudaTradjemaError::InvalidInput(
                "series_len/n_combos/first_valid/max_length exceed i32".into(),
            ));
        }
        let mut func = self
            .module
            .get_function("tradjema_batch_f32")
            .map_err(|_| CudaTradjemaError::MissingKernelSymbol { name: "tradjema_batch_f32" })?;

        // Compute dynamic shared memory and validate against device capability
        let shared_bytes = Self::smem_bytes_for_len(max_length);
        let dev = Device::get_device(self.device_id).map_err(CudaTradjemaError::Cuda)?;
        let max_optin = dev
            .get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
            .map_err(CudaTradjemaError::Cuda)? as usize;
        if shared_bytes > max_optin {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "requested {} B dynamic shared memory exceeds device limit {} B",
                shared_bytes, max_optin
            )));
        }

        // Favor occupancy-based suggestion; fall back to policy/env
        let fallback_bx = self.choose_batch_block_x();
        let _ = func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize);
        let block_x: u32 = func
            .suggested_launch_configuration(shared_bytes, BlockSize::xyz(0, 0, 0))
            .map(|(_, bx)| bx)
            .unwrap_or(fallback_bx)
            .max(32)
            .min(1024);
        // Record selection for introspection/logging
        unsafe {
            (*(self as *const _ as *mut CudaTradjema)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }
        self.maybe_log_batch_debug();

        let grid_x = n_combos as u32;
        Self::validate_launch(self.device_id, block_x, grid_x)?;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut lengths_ptr = d_lengths.as_device_ptr().as_raw();
            let mut mults_ptr = d_mults.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut lengths_ptr as *mut _ as *mut c_void,
                &mut mults_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(CudaTradjemaError::Cuda)?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        combos: &[TradjemaParams],
        first_valid: usize,
        series_len: usize,
        max_length: usize,
    ) -> Result<DeviceArrayF32, CudaTradjemaError> {
        let n_combos = combos.len();
        let mut lengths_i32 = vec![0i32; n_combos];
        let mut mults_f32 = vec![0f32; n_combos];
        for (idx, prm) in combos.iter().enumerate() {
            lengths_i32[idx] = prm.length.unwrap() as i32;
            mults_f32[idx] = prm.mult.unwrap() as f32;
        }

        let item_f32 = std::mem::size_of::<f32>();
        let bytes_ohlc = series_len
            .checked_mul(item_f32)
            .and_then(|b| b.checked_mul(3))
            .ok_or_else(|| CudaTradjemaError::InvalidInput("byte size overflow".into()))?;
        let lengths_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaTradjemaError::InvalidInput("byte size overflow".into()))?;
        let mults_bytes = n_combos
            .checked_mul(item_f32)
            .ok_or_else(|| CudaTradjemaError::InvalidInput("byte size overflow".into()))?;
        let out_elems = n_combos.checked_mul(series_len)
            .ok_or_else(|| CudaTradjemaError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(item_f32)
            .ok_or_else(|| CudaTradjemaError::InvalidInput("byte size overflow".into()))?;
        let required = bytes_ohlc
            .checked_add(lengths_bytes)
            .and_then(|s| s.checked_add(mults_bytes))
            .and_then(|s| s.checked_add(out_bytes))
            .ok_or_else(|| CudaTradjemaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let (free, _total) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaTradjemaError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        // Pinned host buffers + async H2D for better throughput
        let h_high = LockedBuffer::from_slice(high).map_err(CudaTradjemaError::Cuda)?;
        let h_low = LockedBuffer::from_slice(low).map_err(CudaTradjemaError::Cuda)?;
        let h_close = LockedBuffer::from_slice(close).map_err(CudaTradjemaError::Cuda)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths_i32).map_err(CudaTradjemaError::Cuda)?;
        let d_mults = DeviceBuffer::from_slice(&mults_f32).map_err(CudaTradjemaError::Cuda)?;
        let mut d_high: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len) }.map_err(CudaTradjemaError::Cuda)?;
        let mut d_low: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len) }.map_err(CudaTradjemaError::Cuda)?;
        let mut d_close: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len) }.map_err(CudaTradjemaError::Cuda)?;
        unsafe {
            d_high
                .async_copy_from(h_high.as_slice(), &self.stream)
                .map_err(CudaTradjemaError::Cuda)?;
            d_low
                .async_copy_from(h_low.as_slice(), &self.stream)
                .map_err(CudaTradjemaError::Cuda)?;
            d_close
                .async_copy_from(h_close.as_slice(), &self.stream)
                .map_err(CudaTradjemaError::Cuda)?;
        }
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.map_err(CudaTradjemaError::Cuda)?;

        self.launch_batch_kernel(
            &d_high,
            &d_low,
            &d_close,
            &d_lengths,
            &d_mults,
            series_len,
            n_combos,
            first_valid,
            max_length,
            &mut d_out,
        )?;

        self.stream.synchronize().map_err(CudaTradjemaError::Cuda)?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn tradjema_batch_device(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_mults: &DeviceBuffer<f32>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        max_length: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTradjemaError> {
        if series_len <= 0 || n_combos <= 0 || max_length <= 1 {
            return Err(CudaTradjemaError::InvalidInput(
                "series_len, n_combos must be positive and length > 1".into(),
            ));
        }
        self.launch_batch_kernel(
            d_high,
            d_low,
            d_close,
            d_lengths,
            d_mults,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            max_length as usize,
            d_out,
        )
    }

    pub fn tradjema_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &TradjemaBatchRange,
    ) -> Result<DeviceArrayF32, CudaTradjemaError> {
        let (combos, first_valid, series_len, max_length) =
            Self::prepare_batch_inputs(high, low, close, sweep)?;
        self.run_batch_kernel(
            high,
            low,
            close,
            &combos,
            first_valid,
            series_len,
            max_length,
        )
    }

    pub fn tradjema_batch_into_host_f32(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &TradjemaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TradjemaParams>), CudaTradjemaError> {
        let (combos, first_valid, series_len, max_length) =
            Self::prepare_batch_inputs(high, low, close, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(
            high,
            low,
            close,
            &combos,
            first_valid,
            series_len,
            max_length,
        )?;
        // Async device->pinned copy then slice copy for better D2H throughput
        let mut pinned: LockedBuffer<f32> =
            unsafe { LockedBuffer::uninitialized(out.len()) }.map_err(CudaTradjemaError::Cuda)?;
        unsafe {
            arr.buf
                .async_copy_to(&mut pinned, &self.stream)
                .map_err(CudaTradjemaError::Cuda)?;
        }
        self.stream.synchronize().map_err(CudaTradjemaError::Cuda)?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }

    fn prepare_many_series_inputs(
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &TradjemaParams,
    ) -> Result<(Vec<i32>, usize, f32), CudaTradjemaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTradjemaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTradjemaError::InvalidInput("rows*cols overflow".into()))?;
        if high_tm.len() != expected || low_tm.len() != expected || close_tm.len() != expected {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "time-major length mismatch: high={}, low={}, close={}, expected={}",
                high_tm.len(),
                low_tm.len(),
                close_tm.len(),
                expected
            )));
        }

        let length = params.length.unwrap_or(0);
        let mult = params.mult.unwrap_or(0.0) as f32;
        if length < 2 || length > rows {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "invalid length {} (series len {})",
                length, rows
            )));
        }
        if !mult.is_finite() || mult <= 0.0 {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "invalid mult {}",
                params.mult.unwrap_or(0.0)
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let close = close_tm[t * cols + series];
                if !close.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaTradjemaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if rows - fv < length {
                return Err(CudaTradjemaError::InvalidInput(format!(
                    "series {} not enough valid data: needed >= {}, valid = {}",
                    series,
                    length,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, length, mult))
    }

    fn launch_many_series_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        length: usize,
        mult: f32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTradjemaError> {
        let mut func = self
            .module
            .get_function("tradjema_many_series_one_param_time_major_f32")
            .map_err(|_| CudaTradjemaError::MissingKernelSymbol {
                name: "tradjema_many_series_one_param_time_major_f32",
            })?;

        // Shared memory layout:
        // [ f64 min_vals[length] ][ f64 max_vals[length] ][ i32 min_idx[length] ][ i32 max_idx[length] ]
        let shared_bytes = Self::smem_bytes_for_len(length);
        let dev = Device::get_device(self.device_id).map_err(CudaTradjemaError::Cuda)?;
        let max_optin = dev
            .get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
            .map_err(CudaTradjemaError::Cuda)? as usize;
        if shared_bytes > max_optin {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "requested {} B dynamic shared memory exceeds device limit {} B",
                shared_bytes, max_optin
            )));
        }

        let fallback_bx = self.choose_many_block_x();
        let _ = func.set_shared_memory_config(SharedMemoryConfig::EightByteBankSize);
        let block_x: u32 = func
            .suggested_launch_configuration(shared_bytes, BlockSize::xyz(0, 0, 0))
            .map(|(_, bx)| bx)
            .unwrap_or(fallback_bx)
            .max(64)
            .min(1024);
        unsafe {
            (*(self as *const _ as *mut CudaTradjema)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let grid_x = cols as u32;
        Self::validate_launch(self.device_id, block_x, grid_x)?;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut length_i = length as i32;
            let mut mult_f = mult;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut length_i as *mut _ as *mut c_void,
                &mut mult_f as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(CudaTradjemaError::Cuda)?;
        }
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        length: usize,
        mult: f32,
    ) -> Result<DeviceArrayF32, CudaTradjemaError> {
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTradjemaError::InvalidInput("rows*cols overflow".into()))?;
        let bytes_ohlc = elems * std::mem::size_of::<f32>() * 3;
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let required = bytes_ohlc + first_valid_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let (free, _total) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaTradjemaError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        // Pinned + async host-to-device
        let h_high = LockedBuffer::from_slice(high_tm).map_err(CudaTradjemaError::Cuda)?;
        let h_low = LockedBuffer::from_slice(low_tm).map_err(CudaTradjemaError::Cuda)?;
        let h_close = LockedBuffer::from_slice(close_tm).map_err(CudaTradjemaError::Cuda)?;
        let h_first = LockedBuffer::from_slice(first_valids).map_err(CudaTradjemaError::Cuda)?;
        let mut d_high: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTradjemaError::Cuda)?;
        let mut d_low: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTradjemaError::Cuda)?;
        let mut d_close: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTradjemaError::Cuda)?;
        let mut d_first_valids: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized(cols) }.map_err(CudaTradjemaError::Cuda)?;
        unsafe {
            d_high
                .async_copy_from(h_high.as_slice(), &self.stream)
                .map_err(CudaTradjemaError::Cuda)?;
            d_low
                .async_copy_from(h_low.as_slice(), &self.stream)
                .map_err(CudaTradjemaError::Cuda)?;
            d_close
                .async_copy_from(h_close.as_slice(), &self.stream)
                .map_err(CudaTradjemaError::Cuda)?;
            d_first_valids
                .async_copy_from(h_first.as_slice(), &self.stream)
                .map_err(CudaTradjemaError::Cuda)?;
        }
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.map_err(CudaTradjemaError::Cuda)?;

        self.launch_many_series_kernel(
            &d_high,
            &d_low,
            &d_close,
            cols,
            rows,
            length,
            mult,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream.synchronize().map_err(CudaTradjemaError::Cuda)?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn tradjema_many_series_one_param_device(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        num_series: i32,
        series_len: i32,
        length: i32,
        mult: f32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTradjemaError> {
        if num_series <= 0 || series_len <= 0 || length <= 1 || !mult.is_finite() || mult <= 0.0 {
            return Err(CudaTradjemaError::InvalidInput(
                "invalid dimensions or parameters".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_high,
            d_low,
            d_close,
            num_series as usize,
            series_len as usize,
            length as usize,
            mult,
            d_first_valids,
            d_out,
        )
    }

    pub fn tradjema_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &TradjemaParams,
    ) -> Result<DeviceArrayF32, CudaTradjemaError> {
        let (first_valids, length, mult) =
            Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows, params)?;
        self.run_many_series_kernel(
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            &first_valids,
            length,
            mult,
        )
    }

    pub fn tradjema_many_series_one_param_time_major_into_host_f32(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &TradjemaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTradjemaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaTradjemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, length, mult) =
            Self::prepare_many_series_inputs(high_tm, low_tm, close_tm, cols, rows, params)?;
        let arr = self.run_many_series_kernel(
            high_tm,
            low_tm,
            close_tm,
            cols,
            rows,
            &first_valids,
            length,
            mult,
        )?;
        arr.buf
            .copy_to(out_tm)
            .map_err(CudaTradjemaError::Cuda)
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
        // 3 inputs (H/L/C) + outputs (PARAM_SWEEP rows)
        let in_bytes = 3 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = 3 * elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() {
                continue;
            }
            let x = i as f32 * 0.0025;
            let off = (0.003 * x.cos()).abs() + 0.11;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    fn synth_hlc_time_major_from_close(
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut high = close_tm.to_vec();
        let mut low = close_tm.to_vec();
        for t in 0..rows {
            for s in 0..cols {
                let idx = t * cols + s;
                let v = close_tm[idx];
                if v.is_nan() {
                    continue;
                }
                let x = (t as f32) * 0.0023 + (s as f32) * 0.11;
                let off = (0.0029 * x.sin()).abs() + 0.1;
                high[idx] = v + off;
                low[idx] = v - off;
            }
        }
        (high, low)
    }

    struct BatchDevState {
        cuda: CudaTradjema,
        d_high: DeviceBuffer<f32>,
        d_low: DeviceBuffer<f32>,
        d_close: DeviceBuffer<f32>,
        d_lengths: DeviceBuffer<i32>,
        d_mults: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_length: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_high,
                    &self.d_low,
                    &self.d_close,
                    &self.d_lengths,
                    &self.d_mults,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    self.max_length,
                    &mut self.d_out,
                )
                .expect("tradjema batch kernel");
            self.cuda.stream.synchronize().expect("tradjema sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaTradjema::new(0).expect("cuda tradjema");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = TradjemaBatchRange {
            length: (16, 16 + PARAM_SWEEP - 1, 1),
            mult: (8.0, 8.0, 0.0),
        };
        let (combos, first_valid, series_len, max_length) =
            CudaTradjema::prepare_batch_inputs(&high, &low, &close, &sweep)
                .expect("tradjema prepare batch");
        let n_combos = combos.len();
        let lengths_i32: Vec<i32> = combos.iter().map(|p| p.length.unwrap_or(0) as i32).collect();
        let mults_f32: Vec<f32> = combos.iter().map(|p| p.mult.unwrap_or(0.0) as f32).collect();

        let d_high = DeviceBuffer::from_slice(&high).expect("d_high");
        let d_low = DeviceBuffer::from_slice(&low).expect("d_low");
        let d_close = DeviceBuffer::from_slice(&close).expect("d_close");
        let d_lengths = DeviceBuffer::from_slice(&lengths_i32).expect("d_lengths");
        let d_mults = DeviceBuffer::from_slice(&mults_f32).expect("d_mults");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len.checked_mul(n_combos).expect("out size"))
        }
        .expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(BatchDevState {
            cuda,
            d_high,
            d_low,
            d_close,
            d_lengths,
            d_mults,
            series_len,
            n_combos,
            first_valid,
            max_length,
            d_out,
        })
    }

    struct ManyDevState {
        cuda: CudaTradjema,
        d_high_tm: DeviceBuffer<f32>,
        d_low_tm: DeviceBuffer<f32>,
        d_close_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        length: usize,
        mult: f32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_high_tm,
                    &self.d_low_tm,
                    &self.d_close_tm,
                    self.cols,
                    self.rows,
                    self.length,
                    self.mult,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("tradjema many-series kernel");
            self.cuda.stream.synchronize().expect("tradjema many-series sync");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaTradjema::new(0).expect("cuda tradjema");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let close_tm = gen_time_major_prices(cols, rows);
        let (high_tm, low_tm) = synth_hlc_time_major_from_close(&close_tm, cols, rows);
        let params = TradjemaParams {
            length: Some(64),
            mult: Some(8.0),
        };
        let (first_valids, length, mult) =
            CudaTradjema::prepare_many_series_inputs(&high_tm, &low_tm, &close_tm, cols, rows, &params)
                .expect("tradjema prepare many");

        let d_high_tm = DeviceBuffer::from_slice(&high_tm).expect("d_high_tm");
        let d_low_tm = DeviceBuffer::from_slice(&low_tm).expect("d_low_tm");
        let d_close_tm = DeviceBuffer::from_slice(&close_tm).expect("d_close_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols.checked_mul(rows).expect("out size")) }
                .expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(ManyDevState {
            cuda,
            d_high_tm,
            d_low_tm,
            d_close_tm,
            d_first_valids,
            cols,
            rows,
            length,
            mult,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "tradjema",
                "one_series_many_params",
                "tradjema_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "tradjema",
                "many_series_one_param",
                "tradjema_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
