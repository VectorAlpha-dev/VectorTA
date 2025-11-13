//! CUDA scaffolding for the DMA (Dickson Moving Average) kernels.
//!
//! Mirrors ALMA’s CUDA wrapper architecture: explicit policy selection,
//! introspection, VRAM checks, chunked launches, and deterministic benches.
//! For DMA (recursive/IIR), each thread walks time sequentially for its
//! (series,param) while we parallelize across series/params.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::dma::{DmaBatchRange, DmaParams};
use cust::context::Context;
use cust::error::CudaError;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::mem::{size_of, zeroed};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

// -------- Kernel selection policy (mirrors ALMA shape) --------

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
    // For DMA, tiled means 1D tiling across parameter combos; `tile` = threads/block
    Tiled {
        tile: u32,
        per_thread: BatchThreadsPerOutput,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    // 2D tiled across series; tx is kept for API parity (unused), ty = threads on series
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaDmaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

// -------- Introspection --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    Tiled1d { tx: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Debug, Error)]
pub enum CudaDmaError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("not implemented")]
    NotImplemented,
    #[error("out of device memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("output slice length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}), block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer on device {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
}

pub struct CudaDma {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    policy: CudaDmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaDma {
    /// Create a new `CudaDma` on `device_id` and load the PTX module.
    pub fn new(device_id: usize) -> Result<Self, CudaDmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/dma_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O3),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            ctx: Arc::new(context),
            device_id: device_id as u32,
            policy: CudaDmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create using an explicit policy (tests/benches).
    pub fn new_with_policy(device_id: usize, policy: CudaDmaPolicy) -> Result<Self, CudaDmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaDmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaDmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    /// Synchronize the stream (deterministic timings in benches/tests).
    pub fn synchronize(&self) -> Result<(), CudaDmaError> {
        self.stream.synchronize()?;
        Ok(())
    }

    /// Clone of the underlying CUDA context to keep it alive alongside VRAM handles.
    #[inline]
    pub fn context(&self) -> Arc<Context> { self.ctx.clone() }

    /// Device id for interop (__dlpack_device__).
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn maybe_enable_l2_persist_for_prices(
        &self,
        d_prices_bytes: usize,
        d_prices_ptr: u64,
    ) -> Result<(), CudaDmaError> {
        // Default-on best-effort: if unsupported, quietly no-op.
        unsafe {
            // Get current device from context
            let mut dev: cu::CUdevice = 0;
            let rc_dev = cu::cuCtxGetDevice(&mut dev as *mut _);
            if rc_dev != cu::CUresult::CUDA_SUCCESS {
                return Ok(());
            }

            // Query device caps
            let mut max_persist_bytes: i32 = 0;
            let _ = cu::cuDeviceGetAttribute(
                &mut max_persist_bytes as *mut i32,
                cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
                dev,
            );
            if max_persist_bytes <= 0 {
                return Ok(());
            }

            let mut max_window: i32 = 0;
            let _ = cu::cuDeviceGetAttribute(
                &mut max_window as *mut i32,
                cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                dev,
            );
            if max_window <= 0 {
                return Ok(());
            }

            // Set context limit (set-aside)
            let set_aside = d_prices_bytes.min(max_persist_bytes as usize);
            let _ = cu::cuCtxSetLimit(cu::CUlimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, set_aside);

            // Configure stream access policy window
            let mut apw: cu::CUaccessPolicyWindow = zeroed();
            apw.base_ptr = d_prices_ptr as usize as *mut c_void;
            apw.num_bytes = d_prices_bytes.min(max_window as usize);
            apw.hitRatio = 1.0;
            apw.hitProp = cu::CUaccessProperty::CU_ACCESS_PROPERTY_PERSISTING;
            apw.missProp = cu::CUaccessProperty::CU_ACCESS_PROPERTY_NORMAL;

            let mut attr: cu::CUstreamAttrValue = zeroed();
            attr.accessPolicyWindow = apw;

            let hstream = self.stream.as_inner();
            let _ = cu::cuStreamSetAttribute(
                hstream,
                cu::CUstreamAttrID::CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &mut attr as *mut _,
            );
        }
        Ok(())
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
                    eprintln!("[DEBUG] DMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] DMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaDma)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------- Utilities ----------

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn ensure_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaDmaError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.checked_add(headroom_bytes).unwrap_or(usize::MAX) <= free {
                Ok(())
            } else {
                Err(CudaDmaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }
    #[inline]
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535;
        (0..n).step_by(MAX).map(move |s| {
            let l = (n - s).min(MAX);
            (s, l)
        })
    }

    // ---------- Public API: one-series × many-params ----------

    /// Host input → VRAM output; batches parameter combos. Parallelizes across combos.
    pub fn dma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &DmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let inputs = Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_with_prices_host(data_f32, &inputs)
    }

    /// Device prices → VRAM output. Avoids H2D copy when prices already resident.
    pub fn dma_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: &DmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let (combos, max_sqrt_len) =
            Self::prepare_batch_inputs_device(series_len, first_valid, sweep)?;
        self.run_batch_with_prices_device(d_prices, series_len, first_valid, &combos, max_sqrt_len)
    }

    /// Convenience to copy GPU output back to host and return metadata.
    pub fn dma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &DmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<DmaParams>), CudaDmaError> {
        let inputs = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = inputs
            .series_len
            .checked_mul(inputs.hull_lengths.len())
            .ok_or_else(|| CudaDmaError::InvalidInput("rows*cols overflow".into()))?;
        if out.len() != expected {
            return Err(CudaDmaError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
        let arr = self.run_batch_with_prices_host(data_f32, &inputs)?;
        unsafe { arr.buf.async_copy_to(out, &self.stream) }?;
        self.stream.synchronize()?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    /// Lower-level device path used by benches/tests.
    pub fn dma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_hulls: &DeviceBuffer<i32>,
        d_emas: &DeviceBuffer<i32>,
        d_gain_limits: &DeviceBuffer<i32>,
        d_types: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_sqrt_len: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaDmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize {
            return Err(CudaDmaError::InvalidInput(
                "series too long for kernel argument width".into(),
            ));
        }
        self.launch_batch_kernels(
            d_prices,
            d_hulls,
            d_emas,
            d_gain_limits,
            d_types,
            series_len,
            n_combos,
            first_valid,
            max_sqrt_len,
            d_out,
        )
    }

    // ---------- Public API: many-series × one-param (time-major) ----------

    /// Device path for many-series one-param.
    pub fn dma_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        hull_length: i32,
        ema_length: i32,
        ema_gain_limit: i32,
        hull_type: i32,
        series_len: usize,
        num_series: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmaError> {
        if hull_length <= 0 || ema_length <= 0 {
            return Err(CudaDmaError::InvalidInput(
                "hull_length and ema_length must be positive".into(),
            ));
        }
        if series_len == 0 || num_series == 0 {
            return Err(CudaDmaError::InvalidInput(
                "series_len and num_series must be positive".into(),
            ));
        }
        if ema_gain_limit < 0 {
            return Err(CudaDmaError::InvalidInput(
                "ema_gain_limit must be non-negative".into(),
            ));
        }
        if hull_type != 0 && hull_type != 1 {
            return Err(CudaDmaError::InvalidInput(
                "hull_type must be 0 (WMA) or 1 (EMA)".into(),
            ));
        }
        let sqrt_len = ((hull_length as f64).sqrt().round() as usize).max(1);
        self.launch_many_series_kernels(
            d_prices_tm,
            hull_length as usize,
            ema_length as usize,
            ema_gain_limit as usize,
            hull_type,
            series_len,
            num_series,
            d_first_valids,
            sqrt_len,
            d_out_tm,
        )
    }

    /// Host path for many-series one-param (time-major).
    pub fn dma_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &DmaParams,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let (first_valids, hull_length, ema_length, ema_gain_limit, hull_type, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;
        self.run_many_series_kernel(
            data_tm_f32,
            num_series,
            series_len,
            &first_valids,
            hull_length,
            ema_length,
            ema_gain_limit,
            hull_type,
            sqrt_len,
        )
    }

    pub fn dma_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &DmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaDmaError> {
        if out_tm.len() != data_tm_f32.len() {
            return Err(CudaDmaError::OutputLengthMismatch {
                expected: data_tm_f32.len(),
                got: out_tm.len(),
            });
        }
        let (first_valids, hull_length, ema_length, ema_gain_limit, hull_type, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;
        let arr = self.run_many_series_kernel(
            data_tm_f32,
            num_series,
            series_len,
            &first_valids,
            hull_length,
            ema_length,
            ema_gain_limit,
            hull_type,
            sqrt_len,
        )?;
        unsafe { arr.buf.async_copy_to(out_tm, &self.stream) }?;
        self.stream.synchronize()?;
        Ok(())
    }

    // ---------- Internal runners ----------

    fn run_batch_with_prices_host(
        &self,
        data_f32: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let n_combos = inputs.hull_lengths.len();
        let series_len = inputs.series_len;
        let first_valid = inputs.first_valid;
        let max_sqrt_len = inputs.max_sqrt_len;

        let prices_bytes = series_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| CudaDmaError::InvalidInput("series_len bytes overflow".into()))?;
        let hull_bytes = n_combos
            .checked_mul(size_of::<i32>())
            .ok_or_else(|| CudaDmaError::InvalidInput("param bytes overflow".into()))?;
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaDmaError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| CudaDmaError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(hull_bytes.checked_mul(3).ok_or_else(|| CudaDmaError::InvalidInput("param bytes overflow".into()))?)
            .and_then(|v| v.checked_add(out_bytes))
            .and_then(|v| v.checked_add(64 * 1024 * 1024))
            .ok_or_else(|| CudaDmaError::InvalidInput("required bytes overflow".into()))?;
        Self::ensure_fit(required, 0)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream)? };
        // Enable L2 persisting cache hint for prices (best-effort)
        let _ = self.maybe_enable_l2_persist_for_prices(
            series_len * size_of::<f32>(),
            d_prices.as_device_ptr().as_raw(),
        );
        let d_hulls = unsafe { DeviceBuffer::from_slice_async(&inputs.hull_lengths, &self.stream)? };
        let d_emas = unsafe { DeviceBuffer::from_slice_async(&inputs.ema_lengths, &self.stream)? };
        let d_gains = unsafe { DeviceBuffer::from_slice_async(&inputs.ema_gain_limits, &self.stream)? };
        let d_types = unsafe { DeviceBuffer::from_slice_async(&inputs.hull_types, &self.stream)? };
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream)? };

        self.launch_batch_kernels(
            &d_prices,
            &d_hulls,
            &d_emas,
            &d_gains,
            &d_types,
            series_len,
            n_combos,
            first_valid,
            max_sqrt_len,
            &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_batch_with_prices_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        combos: &[DmaParams],
        max_sqrt_len: usize,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let n_combos = combos.len();
        // VRAM check: output + parameter vectors + headroom (input already resident)
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaDmaError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| CudaDmaError::InvalidInput("output bytes overflow".into()))?;
        let param_bytes = 4usize
            .checked_mul(n_combos)
            .and_then(|v| v.checked_mul(size_of::<i32>()))
            .ok_or_else(|| CudaDmaError::InvalidInput("param bytes overflow".into()))?;
        let required = out_bytes
            .checked_add(param_bytes)
            .and_then(|v| v.checked_add(64 * 1024 * 1024))
            .ok_or_else(|| CudaDmaError::InvalidInput("required bytes overflow".into()))?;
        Self::ensure_fit(required, 0)?;
        let mut hulls = Vec::with_capacity(n_combos);
        let mut emas = Vec::with_capacity(n_combos);
        let mut gains = Vec::with_capacity(n_combos);
        let mut types = Vec::with_capacity(n_combos);
        for prm in combos {
            hulls.push(prm.hull_length.unwrap_or(7) as i32);
            emas.push(prm.ema_length.unwrap_or(20) as i32);
            gains.push(prm.ema_gain_limit.unwrap_or(50) as i32);
            let tag = prm
                .hull_ma_type
                .as_deref()
                .unwrap_or("WMA")
                .to_ascii_uppercase();
            types.push(match tag.as_str() {
                "WMA" => 0,
                "EMA" => 1,
                other => {
                    return Err(CudaDmaError::InvalidInput(format!(
                        "unsupported hull_ma_type {}",
                        other
                    )))
                }
            });
        }
        let d_hulls = unsafe { DeviceBuffer::from_slice_async(&hulls, &self.stream) }?;
        let d_emas = unsafe { DeviceBuffer::from_slice_async(&emas, &self.stream) }?;
        let d_gains = unsafe { DeviceBuffer::from_slice_async(&gains, &self.stream) }?;
        let d_types = unsafe { DeviceBuffer::from_slice_async(&types, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream) }?;
        // Hint L2 persistence for device-resident prices
        let _ = self.maybe_enable_l2_persist_for_prices(
            series_len * size_of::<f32>(),
            d_prices.as_device_ptr().as_raw(),
        );
        self.launch_batch_kernels(
            d_prices,
            &d_hulls,
            &d_emas,
            &d_gains,
            &d_types,
            series_len,
            n_combos,
            first_valid,
            max_sqrt_len,
            &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn launch_batch_kernels(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_hulls: &DeviceBuffer<i32>,
        d_emas: &DeviceBuffer<i32>,
        d_gains: &DeviceBuffer<i32>,
        d_types: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_sqrt_len: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmaError> {
        // Prefer 1D tiled across combos when available and combos are large
        let has_tx128 = self
            .module
            .get_function("dma_batch_tiled_f32_tx128")
            .is_ok();
        let has_tx64 = self.module.get_function("dma_batch_tiled_f32_tx64").is_ok();
        let prefer_tiled = match self.policy.batch {
            BatchKernelPolicy::Tiled { .. } => true,
            BatchKernelPolicy::Plain { .. } => false,
            BatchKernelPolicy::Auto => n_combos >= 32,
        };

        if prefer_tiled && (has_tx128 || has_tx64) {
            let mut tx: u32 = match self.policy.batch {
                BatchKernelPolicy::Tiled { tile, .. } => tile,
                _ => std::env::var("DMA_BATCH_TX")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .filter(|&v| v == 64 || v == 128)
                    .unwrap_or(128),
            };
            if tx == 128 && !has_tx128 {
                tx = 64;
            }
            let func_name = if tx == 128 {
                "dma_batch_tiled_f32_tx128"
            } else {
                "dma_batch_tiled_f32_tx64"
            };
            let func = self
                .module
                .get_function(func_name)
                .map_err(|_| CudaDmaError::MissingKernelSymbol { name: func_name })?;
            let block: BlockSize = (tx, 1, 1).into();
            let mut shared_bytes = (max_sqrt_len * tx as usize * size_of::<f32>()) as u32;
            shared_bytes = (shared_bytes + 255) & !255;

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut hull_ptr = d_hulls.as_device_ptr().as_raw();
                let mut ema_ptr = d_emas.as_device_ptr().as_raw();
                let mut gain_ptr = d_gains.as_device_ptr().as_raw();
                let mut type_ptr = d_types.as_device_ptr().as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = n_combos as i32;
                let mut first_valid_i = first_valid as i32;
                let mut sqrt_stride_i = max_sqrt_len as i32;
                let mut out_ptr = d_out.as_device_ptr().as_raw();

                for (start, len) in Self::grid_y_chunks(n_combos) {
                    let mut combo_start_i = start as i32;
                    let grid_x = ((len as u32) + tx - 1) / tx;
                    let grid: GridSize = (grid_x, 1, 1).into();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut hull_ptr as *mut _ as *mut c_void,
                        &mut ema_ptr as *mut _ as *mut c_void,
                        &mut gain_ptr as *mut _ as *mut c_void,
                        &mut type_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut combo_start_i as *mut _ as *mut c_void,
                        &mut sqrt_stride_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(&func, grid, block, shared_bytes, args)
                        .map_err(|e| CudaDmaError::Cuda(e))?;
                }
            }

            unsafe {
                (*(self as *const _ as *mut CudaDma)).last_batch =
                    Some(BatchKernelSelected::Tiled1d { tx });
            }
            self.maybe_log_batch_debug();
        } else {
            let func = self
                .module
                .get_function("dma_batch_f32")
                .map_err(|_| CudaDmaError::MissingKernelSymbol { name: "dma_batch_f32" })?;
            let block_x = match self.policy.batch {
                BatchKernelPolicy::Plain { block_x } => block_x,
                _ => 1,
            };
            let block: BlockSize = (block_x, 1, 1).into();
            let mut shared_bytes = (max_sqrt_len * size_of::<f32>()) as u32;
            shared_bytes = (shared_bytes + 255) & !255;
            for (start, len) in Self::grid_y_chunks(n_combos) {
                let grid: GridSize = (len as u32, 1, 1).into();
                let out_ptr = unsafe { d_out.as_device_ptr() };
                let stream = &self.stream;
                unsafe {
                    launch!(
                        func<<<grid, block, shared_bytes, stream>>>(
                            d_prices.as_device_ptr(),
                            d_hulls.as_device_ptr(),
                            d_emas.as_device_ptr(),
                            d_gains.as_device_ptr(),
                            d_types.as_device_ptr(),
                            series_len as i32,
                            n_combos as i32,
                            first_valid as i32,
                            out_ptr
                        )
                    )
                    .map_err(|e| CudaDmaError::Cuda(e))?;
                }
            }
            unsafe {
                (*(self as *const _ as *mut CudaDma)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x });
            }
            self.maybe_log_batch_debug();
        }
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        first_valids: &[i32],
        hull_length: usize,
        ema_length: usize,
        ema_gain_limit: usize,
        hull_type: i32,
        sqrt_len: usize,
    ) -> Result<DeviceArrayF32, CudaDmaError> {
        let elems = num_series * series_len;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = in_bytes;
        if let Some((free, _)) = mem_get_info().ok() {
            if in_bytes + out_bytes + 64 * 1024 * 1024 > free {
                return Err(CudaDmaError::OutOfMemory { required: in_bytes + out_bytes + 64 * 1024 * 1024, free, headroom: 0 });
            }
        }

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)
                .map_err(|e| CudaDmaError::Cuda(e))?
        };
        // Hint L2 persist for entire slab (time-major)
        let _ = self.maybe_enable_l2_persist_for_prices(
            elems * size_of::<f32>(),
            d_prices.as_device_ptr().as_raw(),
        );
        let d_first = unsafe {
            DeviceBuffer::from_slice_async(first_valids, &self.stream)
                .map_err(|e| CudaDmaError::Cuda(e))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(elems, &self.stream)
                .map_err(|e| CudaDmaError::Cuda(e))?
        };
        self.launch_many_series_kernels(
            &d_prices,
            hull_length,
            ema_length,
            ema_gain_limit,
            hull_type,
            series_len,
            num_series,
            &d_first,
            sqrt_len,
            &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    fn launch_many_series_kernels(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        hull_length: usize,
        ema_length: usize,
        ema_gain_limit: usize,
        hull_type: i32,
        series_len: usize,
        num_series: usize,
        d_first_valids: &DeviceBuffer<i32>,
        sqrt_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaDmaError> {
        let has_2d_ty4 = self
            .module
            .get_function("dma_ms1p_tiled_f32_tx1_ty4")
            .is_ok();
        let has_2d_ty2 = self
            .module
            .get_function("dma_ms1p_tiled_f32_tx1_ty2")
            .is_ok();
        let prefer_2d = match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { .. } => true,
            ManySeriesKernelPolicy::OneD { .. } => false,
            ManySeriesKernelPolicy::Auto => num_series >= 4,
        };

        if prefer_2d && (has_2d_ty4 || has_2d_ty2) {
            let mut ty = match self.policy.many_series {
                ManySeriesKernelPolicy::Tiled2D { ty, .. } => ty,
                _ => 4,
            };
            if ty == 4 && !has_2d_ty4 {
                ty = 2;
            }
            let func_name = if ty == 4 {
                "dma_ms1p_tiled_f32_tx1_ty4"
            } else {
                "dma_ms1p_tiled_f32_tx1_ty2"
            };
            let func = self
                .module
                .get_function(func_name)
                .map_err(|_| CudaDmaError::MissingKernelSymbol { name: func_name })?;
            let block: BlockSize = (1, ty, 1).into();
            let mut shared_bytes = (sqrt_len * ty as usize * size_of::<f32>()) as u32;
            shared_bytes = (shared_bytes + 255) & !255;
            let grid_x = ((num_series as u32) + ty - 1) / ty;
            let grid: GridSize = (grid_x, 1, 1).into();
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shared_bytes, stream>>>(
                        d_prices_tm.as_device_ptr(),
                        hull_length as i32,
                        ema_length as i32,
                        ema_gain_limit as i32,
                        hull_type,
                        series_len as i32,
                        num_series as i32,
                        d_first_valids.as_device_ptr(),
                        sqrt_len as i32,
                        d_out_tm.as_device_ptr()
                    )
                )?;
            }
            unsafe {
                (*(self as *const _ as *mut CudaDma)).last_many =
                    Some(ManySeriesKernelSelected::Tiled2D { tx: 1, ty });
            }
            self.maybe_log_many_debug();
        } else {
            let func = self
                .module
                .get_function("dma_many_series_one_param_f32")
                .map_err(|_| CudaDmaError::MissingKernelSymbol { name: "dma_many_series_one_param_f32" })?;
            let block_x = match self.policy.many_series {
                ManySeriesKernelPolicy::OneD { block_x } => block_x,
                _ => 1,
            };
            let block: BlockSize = (block_x, 1, 1).into();
            let grid: GridSize = (num_series as u32, 1, 1).into();
            let shared_bytes = (sqrt_len * std::mem::size_of::<f32>()) as u32;
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shared_bytes, stream>>>(
                        d_prices_tm.as_device_ptr(),
                        hull_length as i32,
                        ema_length as i32,
                        ema_gain_limit as i32,
                        hull_type,
                        series_len as i32,
                        num_series as i32,
                        d_first_valids.as_device_ptr(),
                        sqrt_len as i32,
                        d_out_tm.as_device_ptr()
                    )
                )?;
            }
            unsafe {
                (*(self as *const _ as *mut CudaDma)).last_many =
                    Some(ManySeriesKernelSelected::OneD { block_x });
            }
            self.maybe_log_many_debug();
        }
        Ok(())
    }

    // ---------- Input preparation ----------

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &DmaBatchRange,
    ) -> Result<BatchInputs, CudaDmaError> {
        if data_f32.is_empty() {
            return Err(CudaDmaError::InvalidInput("empty data".into()));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaDmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        if series_len > i32::MAX as usize {
            return Err(CudaDmaError::InvalidInput(
                "series too long for kernel argument width".into(),
            ));
        }

        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaDmaError::InvalidInput("all values are NaN".into()))?;
        let valid = series_len - first_valid;

        let mut hull_lengths = Vec::with_capacity(combos.len());
        let mut ema_lengths = Vec::with_capacity(combos.len());
        let mut ema_gain_limits = Vec::with_capacity(combos.len());
        let mut hull_types = Vec::with_capacity(combos.len());
        let mut max_sqrt_len = 0usize;

        for prm in &combos {
            let hull_len = prm.hull_length.unwrap_or(0);
            let ema_len = prm.ema_length.unwrap_or(0);
            let gain_limit = prm.ema_gain_limit.unwrap_or(0);
            let hull_ma_type = prm
                .hull_ma_type
                .as_deref()
                .unwrap_or("WMA")
                .to_ascii_uppercase();

            if hull_len == 0 || hull_len > series_len {
                return Err(CudaDmaError::InvalidInput(format!(
                    "invalid hull length {} for data len {}",
                    hull_len, series_len
                )));
            }
            if ema_len == 0 || ema_len > series_len {
                return Err(CudaDmaError::InvalidInput(format!(
                    "invalid ema length {} for data len {}",
                    ema_len, series_len
                )));
            }
            let sqrt_len = ((hull_len as f64).sqrt().round()) as usize;
            let needed = hull_len.max(ema_len) + sqrt_len;
            if valid < needed {
                return Err(CudaDmaError::InvalidInput(format!(
                    "not enough valid data (needed >= {}, valid = {})",
                    needed, valid
                )));
            }

            let hull_tag = match hull_ma_type.as_str() {
                "WMA" => 0,
                "EMA" => 1,
                other => {
                    return Err(CudaDmaError::InvalidInput(format!(
                        "unsupported hull_ma_type {}",
                        other
                    )))
                }
            };

            if hull_len > i32::MAX as usize || ema_len > i32::MAX as usize {
                return Err(CudaDmaError::InvalidInput(
                    "parameter length exceeds kernel limits".into(),
                ));
            }
            if gain_limit > i32::MAX as usize {
                return Err(CudaDmaError::InvalidInput(
                    "ema_gain_limit exceeds kernel limits".into(),
                ));
            }

            hull_lengths.push(hull_len as i32);
            ema_lengths.push(ema_len as i32);
            ema_gain_limits.push(gain_limit as i32);
            hull_types.push(hull_tag);
            max_sqrt_len = max_sqrt_len.max(sqrt_len.max(1));
        }

        Ok(BatchInputs {
            combos,
            hull_lengths,
            ema_lengths,
            ema_gain_limits,
            hull_types,
            first_valid,
            series_len,
            max_sqrt_len,
        })
    }

    fn prepare_batch_inputs_device(
        series_len: usize,
        first_valid: usize,
        sweep: &DmaBatchRange,
    ) -> Result<(Vec<DmaParams>, usize), CudaDmaError> {
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaDmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let mut max_sqrt_len = 0usize;
        for prm in &combos {
            let hull_len = prm.hull_length.unwrap_or(0);
            let ema_len = prm.ema_length.unwrap_or(0);
            if hull_len == 0 || ema_len == 0 || hull_len > series_len || ema_len > series_len {
                return Err(CudaDmaError::InvalidInput(
                    "invalid params vs series length".into(),
                ));
            }
            let sqrt_len = ((hull_len as f64).sqrt().round()) as usize;
            let needed = hull_len.max(ema_len) + sqrt_len;
            let valid = series_len - first_valid;
            if valid < needed {
                return Err(CudaDmaError::InvalidInput("not enough valid data".into()));
            }
            max_sqrt_len = max_sqrt_len.max(sqrt_len.max(1));
        }
        Ok((combos, max_sqrt_len))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &DmaParams,
    ) -> Result<(Vec<i32>, usize, usize, usize, i32, usize), CudaDmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaDmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaDmaError::InvalidInput(format!(
                "data length {} != num_series * series_len {}",
                data_tm_f32.len(),
                num_series * series_len
            )));
        }

        let hull_length = params.hull_length.unwrap_or(7);
        let ema_length = params.ema_length.unwrap_or(20);
        let ema_gain_limit = params.ema_gain_limit.unwrap_or(50);
        if hull_length == 0 || ema_length == 0 {
            return Err(CudaDmaError::InvalidInput(
                "hull_length and ema_length must be positive".into(),
            ));
        }
        let hull_ma_type = params
            .hull_ma_type
            .as_deref()
            .unwrap_or("WMA")
            .to_ascii_uppercase();
        let hull_type_tag = match hull_ma_type.as_str() {
            "WMA" => 0,
            "EMA" => 1,
            other => {
                return Err(CudaDmaError::InvalidInput(format!(
                    "unsupported hull_ma_type {}",
                    other
                )))
            }
        };

        if hull_length > i32::MAX as usize
            || ema_length > i32::MAX as usize
            || ema_gain_limit > i32::MAX as usize
        {
            return Err(CudaDmaError::InvalidInput(
                "parameter exceeds kernel argument width".into(),
            ));
        }

        let sqrt_len = ((hull_length as f64).sqrt().round() as usize).max(1);
        let needed = hull_length.max(ema_length) + sqrt_len;

        let mut first_valids = vec![0i32; num_series];
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + series];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let first = fv.ok_or_else(|| {
                CudaDmaError::InvalidInput(format!("series {} all values are NaN", series))
            })?;
            if series_len - first < needed {
                return Err(CudaDmaError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    needed,
                    series_len - first
                )));
            }
            first_valids[series] = first as i32;
        }

        Ok((
            first_valids,
            hull_length,
            ema_length,
            ema_gain_limit,
            hull_type_tag,
            sqrt_len,
        ))
    }
}

fn axis_values((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    if start < end {
        return (start..=end).step_by(step).collect();
    }
    let mut v: Vec<usize> = (end..=start).step_by(step).collect();
    v.reverse();
    v
}

fn expand_grid(range: &DmaBatchRange) -> Vec<DmaParams> {
    let hull_lengths = axis_values(range.hull_length);
    let ema_lengths = axis_values(range.ema_length);
    let ema_gain_limits = axis_values(range.ema_gain_limit);

    let mut combos = Vec::new();
    for &h in &hull_lengths {
        for &e in &ema_lengths {
            for &g in &ema_gain_limits {
                combos.push(DmaParams {
                    hull_length: Some(h),
                    ema_length: Some(e),
                    ema_gain_limit: Some(g),
                    hull_ma_type: Some(range.hull_ma_type.clone()),
                });
            }
        }
    }
    combos
}

struct BatchInputs {
    combos: Vec<DmaParams>,
    hull_lengths: Vec<i32>,
    ema_lengths: Vec<i32>,
    ema_gain_limits: Vec<i32>,
    hull_types: Vec<i32>,
    first_valid: usize,
    series_len: usize,
    max_sqrt_len: usize,
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        dma_benches,
        CudaDma,
        crate::indicators::moving_averages::dma::DmaBatchRange,
        crate::indicators::moving_averages::dma::DmaParams,
        dma_batch_dev,
        dma_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::dma::DmaBatchRange {
            hull_length: (7, 7 + PARAM_SWEEP - 1, 1),
            ema_length: (20, 20, 0),
            ema_gain_limit: (50, 50, 0),
            hull_ma_type: "WMA".to_string(),
        },
        crate::indicators::moving_averages::dma::DmaParams {
            hull_length: Some(64),
            ema_length: Some(20),
            ema_gain_limit: Some(50),
            hull_ma_type: Some("WMA".to_string()),
        },
        "dma",
        "dma"
    );
    pub use dma_benches::bench_profiles;
}
