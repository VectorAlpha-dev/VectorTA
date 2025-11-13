//! CUDA support for the Square Root Weighted Moving Average (SRWMA).
//!
//! Provides zero-copy device entry points matching the ALMA CUDA API:
//!  * one-series × many-params batch execution, reusing precomputed
//!    square-root weights staged per combination;
//!  * time-major many-series × one-param execution that shares the same
//!    weights across series.
//!
//! Kernels operate purely in FP32 to keep VRAM usage compact while mirroring
//! the scalar CPU semantics (warm-up handling, NaN propagation).

#![cfg(feature = "cuda")]

use super::cwma_wrapper::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::indicators::moving_averages::srwma::{SrwmaBatchRange, SrwmaParams};
use cust::context::{CacheConfig, Context, SharedMemoryConfig};
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::ffi::CString;
// no Display impl; keep errors via thiserror
use thiserror::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Error)]
pub enum CudaSrwmaError {
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error(transparent)]
    CudaDrv(#[from] CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buf on device {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaSrwmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaSrwmaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaSrwma {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaSrwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

/// VRAM-backed array handle for SRWMA with context guard and device id
pub struct DeviceArrayF32Srwma {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Srwma {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

struct PreparedSrwmaBatch {
    combos: Vec<SrwmaParams>,
    first_valid: usize,
    series_len: usize,
    max_wlen: usize,
    periods_i32: Vec<i32>,
    warm_indices: Vec<i32>,
    inv_norms: Vec<f32>,
    weights_flat: Vec<f32>,
}

struct PreparedSrwmaManySeries {
    first_valids: Vec<i32>,
    period: usize,
    weights: Vec<f32>,
    inv_norm: f32,
}

impl CudaSrwma {
    // ----- Helpers: dynamic shared memory sizing, attributes, occupancy -----
    #[inline]
    fn dyn_smem_bytes_batch(block_x: u32, max_wlen: usize) -> u32 {
        let floats = max_wlen + (block_x as usize + max_wlen - 1);
        (floats * core::mem::size_of::<f32>()) as u32
    }

    #[inline]
    fn dyn_smem_bytes_many(block_x: u32, wlen: usize) -> u32 {
        let floats = wlen + (block_x as usize + wlen - 1);
        (floats * core::mem::size_of::<f32>()) as u32
    }

    fn opt_in_dynamic_smem(func: &Function, bytes: u32) -> Result<(), CudaSrwmaError> {
        let res = unsafe {
            use cust::sys::{cuFuncSetAttribute, CUfunction_attribute_enum as Attr};
            cuFuncSetAttribute(
                func.to_raw(),
                Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes as i32,
            )
        };
        if res != cust::sys::CUresult::CUDA_SUCCESS {
            return Err(CudaSrwmaError::Cuda(format!(
                "cuFuncSetAttribute(MAX_DYNAMIC_SHARED) failed: {:?}",
                res
            )));
        }
        Ok(())
    }

    #[inline]
    fn prefer_shared(func: &mut Function) -> Result<(), CudaSrwmaError> {
        func.set_cache_config(CacheConfig::PreferShared)?;
        func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize)
            .map_err(CudaSrwmaError::from)
    }

    fn pick_block_x_auto(func: &Function, dyn_smem_for: &dyn Fn(u32) -> u32) -> u32 {
        let candidates = [256u32, 128, 512, 64, 32];
        for bx in candidates {
            let smem = dyn_smem_for(bx) as usize;
            if let Ok(active) = func.max_active_blocks_per_multiprocessor(BlockSize::x(bx), smem) {
                if active > 0 {
                    return bx;
                }
            }
        }
        128
    }
    pub fn new(device_id: usize) -> Result<Self, CudaSrwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let context = Arc::new(Context::new(device).map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/srwma_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?
                }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaSrwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaSrwmaPolicy,
    ) -> Result<Self, CudaSrwmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    #[inline]
    pub fn set_policy(&mut self, policy: CudaSrwmaPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaSrwmaPolicy {
        &self.policy
    }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaSrwmaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if self.debug_batch_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SRWMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSrwma)).debug_batch_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSrwma)).debug_batch_logged = true;
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
        if self.debug_many_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                let per_scenario =
                    env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SRWMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSrwma)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSrwma)).debug_many_logged = true;
                }
            }
        }
    }

    // VRAM helpers
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
        if let Some((free, _total)) = Self::device_mem_info() {
            return required_bytes.saturating_add(headroom_bytes) <= free;
        }
        true
    }
    #[inline]
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535;
        (0..n).step_by(MAX).map(move |start| {
            let len = (n - start).min(MAX);
            (start, len)
        })
    }

    pub fn srwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SrwmaBatchRange,
    ) -> Result<DeviceArrayF32Srwma, CudaSrwmaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();
        // VRAM estimate (prices + weights + periods + warm + inv + out) with checked arithmetic
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prices_bytes = prepared
            .series_len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let weights_elems = n_combos
            .checked_mul(prepared.max_wlen)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let weights_bytes = weights_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let warm_bytes = n_combos
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let inv_bytes = n_combos
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let out_elems = n_combos
            .checked_mul(prepared.series_len)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|x| x.checked_add(periods_bytes))
            .and_then(|x| x.checked_add(warm_bytes))
            .and_then(|x| x.checked_add(inv_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(required, headroom) {
            let (free, _) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaSrwmaError::OutOfMemory { required, free, headroom });
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_weights =
            unsafe { DeviceBuffer::from_slice_async(&prepared.weights_flat, &self.stream) }
                .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_periods =
            unsafe { DeviceBuffer::from_slice_async(&prepared.periods_i32, &self.stream) }
                .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_warm =
            unsafe { DeviceBuffer::from_slice_async(&prepared.warm_indices, &self.stream) }
                .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_inv = unsafe { DeviceBuffer::from_slice_async(&prepared.inv_norms, &self.stream) }
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(prepared.series_len * n_combos, &self.stream)
                .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_weights,
            &d_periods,
            &d_warm,
            &d_inv,
            prepared.series_len,
            prepared.max_wlen,
            n_combos,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32Srwma {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn srwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights_flat: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warm_indices: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        series_len: usize,
        _first_valid: usize,
        max_wlen: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSrwmaError> {
        if series_len == 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if n_combos == 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if max_wlen == 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "max_wlen must be positive".into(),
            ));
        }
        if d_periods.len() != n_combos
            || d_warm_indices.len() != n_combos
            || d_inv_norms.len() != n_combos
        {
            return Err(CudaSrwmaError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_weights_flat.len() != n_combos * max_wlen {
            return Err(CudaSrwmaError::InvalidInput(
                "weights buffer must be combos * max_wlen".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaSrwmaError::InvalidInput(
                "prices buffer length must equal series_len".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaSrwmaError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_weights_flat,
            d_periods,
            d_warm_indices,
            d_inv_norms,
            series_len,
            max_wlen,
            n_combos,
            d_out,
        )
    }

    pub fn srwma_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SrwmaParams,
    ) -> Result<DeviceArrayF32Srwma, CudaSrwmaError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;
        // VRAM estimate (prices + first_valids + weights + out) with checked arithmetic
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prices_bytes = num_series
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(sz_f32))
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let fv_bytes = num_series
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let weights_bytes = prepared
            .weights
            .len()
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = num_series
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(sz_f32))
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(fv_bytes)
            .and_then(|x| x.checked_add(weights_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaSrwmaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 32 * 1024 * 1024; // 32MB
        if !Self::will_fit(required, headroom) {
            let (free, _) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaSrwmaError::OutOfMemory { required, free, headroom });
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }
            .map_err(CudaSrwmaError::from)?;
        let d_first_valids =
            unsafe { DeviceBuffer::from_slice_async(&prepared.first_valids, &self.stream) }
                .map_err(CudaSrwmaError::from)?;
        let mut d_weights =
            unsafe { DeviceBuffer::from_slice_async(&prepared.weights, &self.stream) }
                .map_err(CudaSrwmaError::from)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(num_series * series_len, &self.stream)
                .map_err(CudaSrwmaError::from)?
        };

        // If constant-memory weights are available in the compiled module, upload once.
        if let Ok(mut sym) = self
            .module
            .get_global::<[f32; 4096]>(&CString::new("srwma_const_w").unwrap())
        {
            let mut buf = [0.0f32; 4096];
            let wlen = prepared.weights.len().min(4096);
            buf[..wlen].copy_from_slice(&prepared.weights[..wlen]);
            sym.copy_from(&buf).map_err(CudaSrwmaError::from)?;
            // Kernel ignores the weights pointer in const-weight variant; safe to keep passing it.
        }

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            &d_weights,
            prepared.period,
            prepared.inv_norm,
            num_series,
            series_len,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32Srwma {
            buf: d_out,
            rows: series_len,
            cols: num_series,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn srwma_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        inv_norm: f32,
        num_series: i32,
        series_len: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSrwmaError> {
        if period <= 1 {
            return Err(CudaSrwmaError::InvalidInput("period must be >= 2".into()));
        }
        if num_series <= 0 || series_len <= 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if d_weights.len() != (period as usize - 1) {
            return Err(CudaSrwmaError::InvalidInput(
                "weights length must equal period - 1".into(),
            ));
        }
        if d_first_valids.len() != num_series as usize {
            return Err(CudaSrwmaError::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }
        if d_prices_tm.len() != (num_series as usize * series_len as usize)
            || d_out_tm.len() != (num_series as usize * series_len as usize)
        {
            return Err(CudaSrwmaError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            d_weights,
            period as usize,
            inv_norm,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn srwma_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SrwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSrwmaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaSrwmaError::InvalidInput(
                "output slice wrong length".into(),
            ));
        }
        let handle = self.srwma_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        // Ensure prior async work is complete before host copy
        self.stream
            .synchronize()
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        handle
            .buf
            .copy_to(out_tm)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights_flat: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warm_indices: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        series_len: usize,
        max_wlen: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSrwmaError> {
        // Policy: Plain only (no tiled variants for SRWMA). Allow block size override.
        let mut block_x: u32 = 128;
        if let BatchKernelPolicy::Plain { block_x: bx } = self.policy.batch {
            block_x = bx.max(1);
        }

        let mut func = match self.module.get_function("srwma_batch_f32") {
            Ok(f) => f,
            Err(_) => return Err(CudaSrwmaError::MissingKernelSymbol { name: "srwma_batch_f32" }),
        };

        // Decide block size: Auto uses occupancy with dynamic smem footprint
        let mut block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
            BatchKernelPolicy::Auto => {
                Self::pick_block_x_auto(&func, &|bx| Self::dyn_smem_bytes_batch(bx, max_wlen))
            }
            _ => 256,
        };

        // Compute/fit dynamic shared memory; opt-in for >48KiB
        let mut shared_bytes = Self::dyn_smem_bytes_batch(block_x, max_wlen);
        if let Ok(dev) = Device::get_device(self.device_id) {
            if let Ok(max_optin) = dev.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock) {
                while (shared_bytes as i32) > max_optin && block_x > 32 {
                    block_x /= 2;
                    shared_bytes = Self::dyn_smem_bytes_batch(block_x, max_wlen);
                }
            }
        }
        Self::opt_in_dynamic_smem(&func, shared_bytes)?;
        Self::prefer_shared(&mut func)?;

        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        // Validate against device attributes
        if let Ok(dev) = Device::get_device(self.device_id) {
            if let (Ok(max_b), Ok(max_gx), Ok(max_gy), Ok(max_gz)) = (
                dev.get_attribute(DeviceAttribute::MaxThreadsPerBlock),
                dev.get_attribute(DeviceAttribute::MaxGridDimX),
                dev.get_attribute(DeviceAttribute::MaxGridDimY),
                dev.get_attribute(DeviceAttribute::MaxGridDimZ),
            ) {
                if block_x as i32 > max_b
                    || grid_x as i32 > max_gx
                    || 1 > max_gy
                    || 1 > max_gz
                {
                    return Err(CudaSrwmaError::LaunchConfigTooLarge { gx: grid_x.max(1), gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
                }
            }
        }

        for (start, len) in Self::grid_y_chunks(n_combos) {
            let shared_bytes = Self::dyn_smem_bytes_batch(block_x, max_wlen);
            let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut weights_ptr = d_weights_flat
                    .as_device_ptr()
                    .add(start * max_wlen)
                    .as_raw();
                let mut weights_ptr = d_weights_flat
                    .as_device_ptr()
                    .add(start * max_wlen)
                    .as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut warm_ptr = d_warm_indices.as_device_ptr().add(start).as_raw();
                let mut inv_ptr = d_inv_norms.as_device_ptr().add(start).as_raw();
                let mut max_wlen_i = max_wlen as i32;
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = len as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let mut args: [*mut c_void; 9] = [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut warm_ptr as *mut _ as *mut c_void,
                    &mut inv_ptr as *mut _ as *mut c_void,
                    &mut max_wlen_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, shared_bytes, &mut args).map_err(CudaSrwmaError::from)?;
            }
        }

        unsafe {
            (*(self as *const _ as *mut CudaSrwma)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        self.stream.synchronize().map_err(CudaSrwmaError::from)
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        inv_norm: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSrwmaError> {
        // OneD policy only (no tiled 2D variant provided for SRWMA)
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let shared_bytes = ((period - 1) * std::mem::size_of::<f32>()) as u32;

        let mut func = match self.module.get_function("srwma_many_series_one_param_f32") {
            Ok(f) => f,
            Err(_) => {
                return Err(CudaSrwmaError::MissingKernelSymbol {
                    name: "srwma_many_series_one_param_f32",
                })
            }
        };
        let wlen = period.saturating_sub(1);
        let mut block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
            ManySeriesKernelPolicy::Auto => {
                Self::pick_block_x_auto(&func, &|bx| Self::dyn_smem_bytes_many(bx, wlen))
            }
            _ => 256,
        };
        let mut shared_bytes = Self::dyn_smem_bytes_many(block_x, wlen);
        if let Ok(dev) = Device::get_device(self.device_id) {
            if let Ok(max_optin) = dev.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock) {
                while (shared_bytes as i32) > max_optin && block_x > 32 {
                    block_x /= 2;
                    shared_bytes = Self::dyn_smem_bytes_many(block_x, wlen);
                }
            }
        }
        Self::opt_in_dynamic_smem(&func, shared_bytes)?;
        Self::prefer_shared(&mut func)?;
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        // Validate against device attributes
        if let Ok(dev) = Device::get_device(self.device_id) {
            if let (Ok(max_b), Ok(max_gx), Ok(max_gy), Ok(max_gz)) = (
                dev.get_attribute(DeviceAttribute::MaxThreadsPerBlock),
                dev.get_attribute(DeviceAttribute::MaxGridDimX),
                dev.get_attribute(DeviceAttribute::MaxGridDimY),
                dev.get_attribute(DeviceAttribute::MaxGridDimZ),
            ) {
                if block_x as i32 > max_b
                    || grid_x as i32 > max_gx
                    || (num_series as i32) > max_gy
                    || 1 > max_gz
                {
                    return Err(CudaSrwmaError::LaunchConfigTooLarge { gx: grid_x.max(1), gy: num_series as u32, gz: 1, bx: block_x, by: 1, bz: 1 });
                }
            }
        }

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv = inv_norm;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 8] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.stream
                .launch(&func, grid, block, shared_bytes, &mut args)
                .map_err(CudaSrwmaError::from)?;
        }

        unsafe {
            (*(self as *const _ as *mut CudaSrwma)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaSrwma)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        self.stream
            .synchronize()
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SrwmaBatchRange,
    ) -> Result<PreparedSrwmaBatch, CudaSrwmaError> {
        if data_f32.is_empty() {
            return Err(CudaSrwmaError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaSrwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaSrwmaError::InvalidInput("all values are NaN".into()))?;

        let mut max_wlen = 0usize;
        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaSrwmaError::InvalidInput(format!(
                    "invalid period {} (must be >= 2)",
                    period
                )));
            }
            if series_len - first_valid < period + 1 {
                return Err(CudaSrwmaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period + 1,
                    series_len - first_valid
                )));
            }
            max_wlen = max_wlen.max(period - 1);
        }

        let n_combos = combos.len();
        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warm_indices = Vec::with_capacity(n_combos);
        let mut inv_norms = Vec::with_capacity(n_combos);
        let mut weights_flat = vec![0f32; n_combos * max_wlen];

        for (idx, params) in combos.iter().enumerate() {
            let period = params.period.unwrap();
            let wlen = period - 1;
            let mut norm = 0f32;
            for k in 0..wlen {
                let weight = ((period - k) as f32).sqrt();
                weights_flat[idx * max_wlen + k] = weight;
                norm += weight;
            }
            if norm <= 0.0 {
                return Err(CudaSrwmaError::InvalidInput(format!(
                    "period {} produced non-positive norm",
                    period
                )));
            }
            periods_i32.push(period as i32);
            warm_indices.push((first_valid + period + 1) as i32);
            inv_norms.push(1.0f32 / norm);
        }

        Ok(PreparedSrwmaBatch {
            combos,
            first_valid,
            series_len,
            max_wlen,
            periods_i32,
            warm_indices,
            inv_norms,
            weights_flat,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SrwmaParams,
    ) -> Result<PreparedSrwmaManySeries, CudaSrwmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaSrwmaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        let period = params.period.unwrap_or(14);
        if period < 2 {
            return Err(CudaSrwmaError::InvalidInput(format!(
                "invalid period {} (must be >= 2)",
                period
            )));
        }
        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + series];
                if v.is_finite() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaSrwmaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if series_len - fv < period + 1 {
                return Err(CudaSrwmaError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    period + 1,
                    series_len - fv
                )));
            }
            first_valids.push(fv as i32);
        }

        let wlen = period - 1;
        let mut weights = Vec::with_capacity(wlen);
        let mut norm = 0f32;
        for k in 0..wlen {
            let weight = ((period - k) as f32).sqrt();
            weights.push(weight);
            norm += weight;
        }
        if norm <= 0.0 {
            return Err(CudaSrwmaError::InvalidInput(
                "computed weight norm <= 0".into(),
            ));
        }
        let inv_norm = 1.0f32 / norm;

        Ok(PreparedSrwmaManySeries {
            first_valids,
            period,
            weights,
            inv_norm,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        srwma_benches,
        CudaSrwma,
        crate::indicators::moving_averages::srwma::SrwmaBatchRange,
        crate::indicators::moving_averages::srwma::SrwmaParams,
        srwma_batch_dev,
        srwma_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::srwma::SrwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::srwma::SrwmaParams { period: Some(64) },
        "srwma",
        "srwma"
    );
    pub use srwma_benches::bench_profiles;
}

fn expand_grid(range: &SrwmaBatchRange) -> Vec<SrwmaParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 { return vec![start]; }
        if start == end { return vec![start]; }
        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(step) { Some(nx) if nx > x => x = nx, _ => break }
            }
        } else {
            let mut x = start;
            while x >= end {
                v.push(x);
                match x.checked_sub(step) { Some(nx) if nx < x => x = nx, _ => break }
                if x == 0 { break; }
            }
        }
        v
    }
    let periods = axis(range.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SrwmaParams { period: Some(p) });
    }
    out
}
