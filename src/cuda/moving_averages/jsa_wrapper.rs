//! CUDA support for the Jump Step Average (JSA).
//!
//! Mirrors the ALMA CUDA surface for user experience consistency while staying
//! lean: a single plain kernel path is sufficient for JSA, but we still expose
//! policy enums, VRAM pre-flight checks, grid chunking, BENCH_DEBUG logging, and
//! robust PTX JIT loading (DetermineTargetFromContext + O2 → fallback).
//!
//! Provided device entry points (match ALMA naming/patterns):
//! - `jsa_batch_f32`                        — one-series × many-params (period sweep)
//! - `jsa_many_series_one_param_f32`       — time‑major many-series × one param
//!
//! Warmup/NaN policy is identical to the scalar implementation: write NaN for
//! indices < warm, and compute `(x + y) * 0.5` in that exact order for parity.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::jsa::{JsaBatchRange, JsaParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::error::CudaError;
use std::convert::TryFrom;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// -------- Kernel selection policy (mirrors ALMA; only Plain used here) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput {
    One,
    Two,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain {
        block_x: u32,
    },
    Tiled {
        tile: u32,
        per_thread: BatchThreadsPerOutput,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaJsaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaJsaPolicy {
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
pub enum CudaJsaError {
    Cuda(CudaError),
    InvalidInput(String),
    NotImplemented,
    OutOfMemory { required: usize, free: usize, headroom: usize },
    MissingKernelSymbol { name: &'static str },
    InvalidPolicy(&'static str),
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    DeviceMismatch { buf: u32, current: u32 },
}

impl From<CudaError> for CudaJsaError {
    fn from(e: CudaError) -> Self { CudaJsaError::Cuda(e) }
}

impl fmt::Display for CudaJsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaJsaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaJsaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaJsaError::NotImplemented => write!(f, "Not implemented"),
            CudaJsaError::OutOfMemory { required, free, headroom } => write!(
                f,
                "insufficient device memory: required={} free={} headroom={}",
                required, free, headroom
            ),
            CudaJsaError::MissingKernelSymbol { name } => write!(f, "missing kernel symbol: {}", name),
            CudaJsaError::InvalidPolicy(p) => write!(f, "invalid policy: {}", p),
            CudaJsaError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz } => write!(
                f,
                "launch config exceeds device limits: grid=({}, {}, {}), block=({}, {}, {})",
                gx, gy, gz, bx, by, bz
            ),
            CudaJsaError::DeviceMismatch { buf, current } => write!(
                f,
                "device mismatch: buffer on device {} but current is {}",
                buf, current
            ),
        }
    }
}

impl std::error::Error for CudaJsaError {}

/// Minimal VRAM handle for Python interop (keeps context/device alive)
#[cfg(all(feature = "python", feature = "cuda"))]
pub struct JsaDeviceHandle {
    pub(crate) buf: DeviceBuffer<f32>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

pub struct CudaJsa {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaJsaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    // Device capability cache for portable grid chunking.
    max_grid_x: usize,
}

struct PreparedJsaBatch {
    combos: Vec<JsaParams>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    warm_indices: Vec<i32>,
}

struct PreparedJsaManySeries {
    first_valids: Vec<i32>,
    warm_indices: Vec<i32>,
    period: i32,
}

impl CudaJsa {
    pub fn new(device_id: usize) -> Result<Self, CudaJsaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/jsa_kernel.ptx"));
        // Adopt ALMA's robust JIT loading strategy. Prefer O4, then relax.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => { Module::from_ptx(ptx, &[])? }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Query device limit (portable grid chunking)
        let max_grid_x = device.get_attribute(DeviceAttribute::MaxGridDimX)? as usize;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaJsaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            max_grid_x,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaJsaPolicy) -> Result<Self, CudaJsaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    #[inline]
    pub fn set_policy(&mut self, policy: CudaJsaPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaJsaPolicy {
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
    pub fn synchronize(&self) -> Result<(), CudaJsaError> {
        self.stream.synchronize().map_err(CudaJsaError::from)
    }

    // ---------- VRAM helpers ----------
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
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    // ---------- Debug logging (once per scenario when BENCH_DEBUG=1) ----------
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
                    eprintln!("[DEBUG] JSA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaJsa)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] JSA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaJsa)).debug_many_logged = true;
                }
            }
        }
    }

    // Chunk combos across grid.x to avoid exceeding device grid limits.
    #[inline]
    fn grid_x_chunks(&self, n: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let cap = self.max_grid_x.max(1);
        (0..n).step_by(cap).map(move |start| {
            let len = (n - start).min(cap);
            (start, len)
        })
    }

    /// Best-effort: request an L2 persisting cache window for read-mostly spans.
    /// Enabled by default; opt-out via JSA_L2_PERSIST=0.
    fn try_enable_persisting_l2(&self, base_dev_ptr: u64, bytes: usize) {
        if std::env::var("JSA_L2_PERSIST").ok().as_deref() == Some("0") {
            return;
        }
        unsafe {
            use cust::sys::{
                cuCtxSetLimit, cuDeviceGetAttribute, cuStreamSetAttribute,
                CUaccessPolicyWindow_v1 as CUaccessPolicyWindow,
                CUaccessProperty_enum as AccessProp, CUdevice_attribute_enum as DevAttr,
                CUlimit_enum as CULimit, CUstreamAttrID_enum as StreamAttrId,
                CUstreamAttrValue_v1 as CUstreamAttrValue,
            };

            // Max window size supported by device
            let mut max_window_bytes_i32: i32 = 0;
            let _ = cuDeviceGetAttribute(
                &mut max_window_bytes_i32 as *mut _,
                DevAttr::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                self.device_id as i32,
            );
            let max_window_bytes = (max_window_bytes_i32.max(0) as usize).min(bytes);
            if max_window_bytes == 0 {
                return;
            }

            // Best-effort set-aside for L2 persistence
            let _ = cuCtxSetLimit(CULimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, max_window_bytes);

            // Configure the stream access policy window
            let mut val: CUstreamAttrValue = std::mem::zeroed();
            val.accessPolicyWindow = CUaccessPolicyWindow {
                base_ptr: base_dev_ptr as *mut std::ffi::c_void,
                num_bytes: max_window_bytes,
                hitRatio: 0.6f32,
                hitProp: AccessProp::CU_ACCESS_PROPERTY_PERSISTING,
                missProp: AccessProp::CU_ACCESS_PROPERTY_STREAMING,
            };
            let _ = cuStreamSetAttribute(
                self.stream.as_inner(),
                StreamAttrId::CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &mut val as *mut _,
            );
        }
    }

    // ---------------- Python-friendly variants returning a handle with context ----------------
    #[cfg(all(feature = "python", feature = "cuda"))]
    pub fn jsa_batch_dev_handle(
        &self,
        data_f32: &[f32],
        sweep: &JsaBatchRange,
    ) -> Result<JsaDeviceHandle, CudaJsaError> {
        use core::mem::ManuallyDrop;
        let arr = self.jsa_batch_dev(data_f32, sweep)?;
        let arr = ManuallyDrop::new(arr);
        // SAFETY: prevent dropping `arr` so moving out the buffer is safe
        let buf = unsafe { std::ptr::read(&arr.buf) };
        Ok(JsaDeviceHandle {
            buf,
            rows: arr.rows,
            cols: arr.cols,
            _ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    #[cfg(all(feature = "python", feature = "cuda"))]
    pub fn jsa_many_series_one_param_time_major_dev_handle(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &JsaParams,
    ) -> Result<JsaDeviceHandle, CudaJsaError> {
        use core::mem::ManuallyDrop;
        let arr = self.jsa_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        let arr = ManuallyDrop::new(arr);
        let buf = unsafe { std::ptr::read(&arr.buf) };
        Ok(JsaDeviceHandle {
            buf,
            rows: arr.rows,
            cols: arr.cols,
            _ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn jsa_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &JsaBatchRange,
    ) -> Result<DeviceArrayF32, CudaJsaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        // VRAM estimate + headroom (~64MB)
        let n_combos = prepared.combos.len();
        let prices_bytes = prepared
            .series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let warm_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let out_elems = prepared
            .series_len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let required = prices_bytes
            .checked_add(periods_bytes)
            .and_then(|x| x.checked_add(warm_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaJsaError::OutOfMemory { required, free, headroom });
        }

        // Async allocations and copies on NON_BLOCKING stream; sync at method end.
        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;
        let d_periods =
            unsafe { DeviceBuffer::from_slice_async(&prepared.periods_i32, &self.stream) }?;
        let d_warm =
            unsafe { DeviceBuffer::from_slice_async(&prepared.warm_indices, &self.stream) }?;
        let total_elems = prepared
            .series_len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total_elems, &self.stream)? };

        // Best-effort L2 persisting hint for shared prices vector
        self.try_enable_persisting_l2(d_prices.as_device_ptr().as_raw() as u64, prices_bytes);

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_warm,
            prepared.series_len,
            prepared.first_valid,
            n_combos,
            &mut d_out,
        )?;

        // Maintain synchronous semantics at API boundary
        self.stream.synchronize().map_err(CudaJsaError::from)?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn jsa_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJsaError> {
        if series_len == 0 {
            return Err(CudaJsaError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaJsaError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if n_combos == 0 {
            return Err(CudaJsaError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaJsaError::InvalidInput(
                "prices buffer length mismatch".into(),
            ));
        }
        if d_periods.len() != n_combos || d_warm.len() != n_combos {
            return Err(CudaJsaError::InvalidInput(
                "period or warm buffer length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaJsaError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_warm,
            series_len,
            first_valid,
            n_combos,
            d_out,
        )
    }

    pub fn jsa_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &JsaBatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaJsaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaJsaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.jsa_batch_dev(data_f32, sweep)?;
        handle.buf.copy_to(out_flat).map_err(CudaJsaError::from)
    }

    pub fn jsa_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &JsaParams,
    ) -> Result<DeviceArrayF32, CudaJsaError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        // VRAM estimate with ~64MB headroom (mirror ALMA approach)
        let elems = num_series
            .checked_mul(series_len)
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let input_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let out_bytes = input_bytes;
        let idx_bytes = num_series
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|x| x.checked_mul(2)) // first_valids + warm_indices
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let required = input_bytes
            .checked_add(out_bytes)
            .and_then(|x| x.checked_add(idx_bytes))
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaJsaError::OutOfMemory { required, free, headroom });
        }

        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }?;
        // Hint: persist the input price window in L2 when supported (default on)
        self.try_enable_persisting_l2(d_prices_tm.as_device_ptr().as_raw() as u64, input_bytes);
        let d_first =
            unsafe { DeviceBuffer::from_slice_async(&prepared.first_valids, &self.stream) }?;
        let d_warm =
            unsafe { DeviceBuffer::from_slice_async(&prepared.warm_indices, &self.stream) }?;
        let total_elems = num_series
            .checked_mul(series_len)
            .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total_elems, &self.stream)? };

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_first,
            &d_warm,
            prepared.period,
            num_series,
            series_len,
            &mut d_out_tm,
        )?;

        // Maintain synchronous semantics before handing back the buffer
        self.stream.synchronize().map_err(CudaJsaError::from)?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows: series_len,
            cols: num_series,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn jsa_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        d_warm_indices: &DeviceBuffer<i32>,
        period: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJsaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaJsaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if period <= 0 {
            return Err(CudaJsaError::InvalidInput("period must be positive".into()));
        }
        let expected = num_series * series_len;
        if d_prices_tm.len() != expected || d_out_tm.len() != expected {
            return Err(CudaJsaError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series || d_warm_indices.len() != num_series {
            return Err(CudaJsaError::InvalidInput(
                "first_valid or warm index buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            d_warm_indices,
            period,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    pub fn jsa_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &JsaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaJsaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaJsaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.jsa_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle.buf.copy_to(out_tm).map_err(CudaJsaError::from)
    }

    /// Return the batch sweep into pinned (page-locked) host memory for faster D2H.
    pub fn jsa_batch_into_pinned_f32(
        &self,
        data_f32: &[f32],
        sweep: &JsaBatchRange,
    ) -> Result<LockedBuffer<f32>, CudaJsaError> {
        let dev = self.jsa_batch_dev(data_f32, sweep)?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(dev.rows * dev.cols)? };
        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaJsaError::from)?;
        }
        self.stream.synchronize().map_err(CudaJsaError::from)?;
        Ok(pinned)
    }

    /// Return many-series time-major result into pinned (page-locked) host memory.
    pub fn jsa_many_series_one_param_time_major_into_pinned_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &JsaParams,
    ) -> Result<LockedBuffer<f32>, CudaJsaError> {
        let dev = self.jsa_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(num_series * series_len)? };
        unsafe {
            dev.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaJsaError::from)?;
        }
        self.stream.synchronize().map_err(CudaJsaError::from)?;
        Ok(pinned)
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJsaError> {
        // Resolve policy using CUDA occupancy when Auto.
        let func = self
            .module
            .get_function("jsa_batch_f32")
            .map_err(|_| CudaJsaError::MissingKernelSymbol { name: "jsa_batch_f32" })?;
        let (suggested_block_x, _min_grid) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .map_err(CudaJsaError::from)?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            _ => suggested_block_x.max(128).min(1024),
        };
        let block: BlockSize = (block_x, 1, 1).into();

        for (start, len) in self.grid_x_chunks(n_combos) {
            let grid: GridSize = (len as u32, 1, 1).into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut warm_ptr = d_warm.as_device_ptr().add(start).as_raw();
                let mut first_valid_i = first_valid as i32;
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let mut args: [*mut c_void; 7] = [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut warm_ptr as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, &mut args)
                    .map_err(CudaJsaError::from)?;
            }
        }

        // Record selection and optionally log for benches.
        unsafe {
            let this = self as *const _ as *mut CudaJsa;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        self.stream.synchronize().map_err(CudaJsaError::from)
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        d_warm: &DeviceBuffer<i32>,
        period: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJsaError> {
        // Heuristic: use coalesced 2D mapping when wide in series and deep enough in time,
        // or when policy explicitly requests Tiled2D.
        const TIME_TILE: u32 = 64; // must match JSA_TIME_TILE in PTX
        let prefer_coalesced =
            matches!(
                self.policy.many_series,
                ManySeriesKernelPolicy::Tiled2D { .. }
            ) || (matches!(self.policy.many_series, ManySeriesKernelPolicy::Auto)
                && num_series >= 128
                && series_len >= 512);

        // Coalesced 2D variant exists in PTX but is disabled here for
        // simplicity and stability. Always use the 1D mapping below.

        // Fallback: original 1D mapping (one block per series)
        let func = self
            .module
            .get_function("jsa_many_series_one_param_f32")
            .map_err(|_| CudaJsaError::MissingKernelSymbol { name: "jsa_many_series_one_param_f32" })?;
        let (suggested_block_x, _min_grid) = func
            .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
            .map_err(CudaJsaError::from)?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
            _ => suggested_block_x.max(128).min(1024),
        };
        let block: BlockSize = (block_x, 1, 1).into();
        let grid_x_u32 = num_series as u32;
        if (num_series as usize) > self.max_grid_x {
            return Err(CudaJsaError::LaunchConfigTooLarge { gx: grid_x_u32, gy: 1, gz: 1, bx: block_x, by: 1, bz: 1 });
        }
        let grid: GridSize = (grid_x_u32, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut warm_ptr = d_warm.as_device_ptr().as_raw();
            let mut period_i = period;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 7] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args).map_err(CudaJsaError::from)?;
        }
        unsafe {
            let this = self as *const _ as *mut CudaJsa;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        self.stream.synchronize().map_err(CudaJsaError::from)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &JsaBatchRange,
    ) -> Result<PreparedJsaBatch, CudaJsaError> {
        if data_f32.is_empty() {
            return Err(CudaJsaError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_periods(sweep)?;

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaJsaError::InvalidInput("all values are NaN".into()))?;

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut warm_indices = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaJsaError::InvalidInput("period must be positive".into()));
            }
            if series_len - first_valid < period {
                return Err(CudaJsaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, have {}",
                    period,
                    series_len - first_valid
                )));
            }
            let period_i32 = i32::try_from(period)
                .map_err(|_| CudaJsaError::InvalidInput("period exceeds i32".into()))?;
            let warm = first_valid
                .checked_add(period)
                .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
            periods_i32.push(period_i32);
            warm_indices.push(warm as i32);
        }

        Ok(PreparedJsaBatch {
            combos,
            first_valid,
            series_len,
            periods_i32,
            warm_indices,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &JsaParams,
    ) -> Result<PreparedJsaManySeries, CudaJsaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaJsaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaJsaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaJsaError::InvalidInput("period must be positive".into()));
        }
        if period > series_len {
            return Err(CudaJsaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, series_len
            )));
        }

        let period_i32 = i32::try_from(period)
            .map_err(|_| CudaJsaError::InvalidInput("period exceeds i32".into()))?;

        let mut first_valids = Vec::with_capacity(num_series);
        let mut warm_indices = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let value = data_tm_f32[t * num_series + series];
                if value.is_finite() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaJsaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if series_len - fv < period {
                return Err(CudaJsaError::InvalidInput(format!(
                    "series {} does not have enough valid data (needed >= {}, have {})",
                    series,
                    period,
                    series_len - fv
                )));
            }
            first_valids.push(fv as i32);
            let warm = (fv as usize)
                .checked_add(period as usize)
                .ok_or_else(|| CudaJsaError::InvalidInput("size overflow".into()))?;
            warm_indices.push(warm as i32);
        }

        Ok(PreparedJsaManySeries {
            first_valids,
            warm_indices,
            period: period_i32,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        jsa_benches,
        CudaJsa,
        crate::indicators::moving_averages::jsa::JsaBatchRange,
        crate::indicators::moving_averages::jsa::JsaParams,
        jsa_batch_dev,
        jsa_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::jsa::JsaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::jsa::JsaParams { period: Some(64) },
        "jsa",
        "jsa"
    );
    pub use jsa_benches::bench_profiles;
}

fn expand_periods(range: &JsaBatchRange) -> Result<Vec<JsaParams>, CudaJsaError> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return Ok(vec![JsaParams { period: Some(start) }]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut v = start;
        while v <= end {
            out.push(JsaParams { period: Some(v) });
            match v.checked_add(step) { Some(n) => v = n, None => break }
        }
    } else {
        let mut v = start;
        while v >= end {
            out.push(JsaParams { period: Some(v) });
            if v < end + step { break; }
            v -= step;
            if v == usize::MAX { break; }
        }
    }
    if out.is_empty() {
        Err(CudaJsaError::InvalidInput(format!(
            "invalid range expansion: start={}, end={}, step={}",
            start, end, step
        )))
    } else {
        Ok(out)
    }
}
