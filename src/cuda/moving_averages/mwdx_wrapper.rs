//! CUDA support for the Midway Weighted Exponential (MWDX) indicator.
//!
//! Aligns with ALMA's CUDA API and policy surface while respecting MWDX's
//! recurrence nature (sequential over time). Exposes device entry points for
//! both one-series × many-params ("batch") and many-series × one-param
//! scenarios and mirrors warmup/NaN behavior of the scalar reference.

#![cfg(feature = "cuda")]

use super::cwma_wrapper::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::indicators::moving_averages::mwdx::{expand_grid_mwdx, MwdxBatchRange, MwdxParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::AsyncCopyDestination; // for async_copy_to on DeviceBuffer
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu; // raw CUDA driver API
use std::ffi::c_void;
use std::fmt;
use thiserror::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Error)]
pub enum CudaMwdxError {
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
    #[error(
        "launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})"
    )]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

/// VRAM-backed array handle for MWDX with context guard and device id
pub struct DeviceArrayF32Mwdx {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Mwdx {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

pub struct CudaMwdx {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaMwdxPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

struct PreparedMwdxBatch {
    combos: Vec<MwdxParams>,
    first_valid: usize,
    series_len: usize,
    factors_f32: Vec<f32>,
}

struct PreparedMwdxManySeries {
    first_valids: Vec<i32>,
    factor: f32,
}

impl CudaMwdx {
    /// Best-effort: request L2 persisting set-aside and configure the stream's
    /// Access Policy Window (APW) so that `d_prices` is treated as persisting in L2.
    ///
    /// Safe to call repeatedly; failures are ignored and do not affect correctness.
    unsafe fn try_enable_persisting_l2_for_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
    ) {
        use cu::{
            cuCtxSetLimit, cuDeviceGetAttribute, cuStreamSetAttribute,
            CUaccessPolicyWindow_v1 as CUaccessPolicyWindow, CUaccessProperty_enum as AccessProp,
            CUdevice_attribute_enum as DevAttr, CUlimit_enum as CULimit,
            CUstreamAttrID_enum as StreamAttrId, CUstreamAttrValue_v1 as CUstreamAttrValue,
        };
        use cust::device::Device as CuDevice;

        let window_bytes = series_len.saturating_mul(std::mem::size_of::<f32>());

        // Query device max supported APW size and clamp the request
        let mut max_window_bytes_i32: i32 = 0;
        if let Ok(dev) = CuDevice::get_device(self.device_id) {
            let _ = cuDeviceGetAttribute(
                &mut max_window_bytes_i32 as *mut _,
                DevAttr::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                dev.as_raw(),
            );
        }
        let max_window_bytes = (max_window_bytes_i32.max(0) as usize).min(window_bytes);

        // Best-effort L2 persisting set-aside
        let _ = cuCtxSetLimit(CULimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, max_window_bytes);

        // Configure stream APW: persisting on hit, streaming on miss
        let mut val: CUstreamAttrValue = std::mem::zeroed();
        val.accessPolicyWindow = CUaccessPolicyWindow {
            base_ptr: d_prices.as_device_ptr().as_raw() as *mut c_void,
            num_bytes: max_window_bytes,
            hitRatio: 0.90f32,
            hitProp: AccessProp::CU_ACCESS_PROPERTY_PERSISTING,
            missProp: AccessProp::CU_ACCESS_PROPERTY_STREAMING,
        };
        let _ = cuStreamSetAttribute(
            self.stream.as_inner(),
            StreamAttrId::CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &mut val as *mut _,
        );
    }
    pub fn new(device_id: usize) -> Result<Self, CudaMwdxError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/mwdx_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
            ModuleJitOption::MaxRegisters(64),
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
            policy: CudaMwdxPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaMwdxPolicy,
    ) -> Result<Self, CudaMwdxError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaMwdxPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaMwdxPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaMwdxError> {
        self.stream.synchronize().map_err(CudaMwdxError::from)
    }

    pub fn mwdx_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MwdxBatchRange,
    ) -> Result<DeviceArrayF32Mwdx, CudaMwdxError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        // VRAM estimate (+64MB headroom)
        let bytes_prices = prepared.series_len * std::mem::size_of::<f32>();
        let bytes_factors = n_combos * std::mem::size_of::<f32>();
        let bytes_out = prepared.series_len * n_combos * std::mem::size_of::<f32>();
        let required = bytes_prices
            .checked_add(bytes_factors)
            .and_then(|x| x.checked_add(bytes_out))
            .ok_or_else(|| CudaMwdxError::InvalidInput("byte size overflow".into()))?;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaMwdxError::OutOfMemory {
                    required,
                    free,
                    headroom: 64 * 1024 * 1024,
                });
            } else {
                return Err(CudaMwdxError::InvalidInput(
                    "insufficient VRAM (mem_get_info unavailable)".into(),
                ));
            }
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let d_factors = DeviceBuffer::from_slice(&prepared.factors_f32)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_factors,
            prepared.series_len,
            prepared.first_valid,
            n_combos,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32Mwdx {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn mwdx_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_factors: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMwdxError> {
        if series_len == 0 {
            return Err(CudaMwdxError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if n_combos == 0 {
            return Err(CudaMwdxError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaMwdxError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if d_prices.len() != series_len {
            return Err(CudaMwdxError::InvalidInput(
                "prices buffer length mismatch".into(),
            ));
        }
        if d_factors.len() != n_combos {
            return Err(CudaMwdxError::InvalidInput(
                "factors buffer length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_factors,
            series_len,
            first_valid,
            n_combos,
            d_out,
        )
    }

    pub fn mwdx_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &MwdxBatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaMwdxError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaMwdxError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.mwdx_batch_dev(data_f32, sweep)?;

        // Use page-locked host buffer and async copy for better throughput.
        let mut host_locked: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(out_flat.len())
                .map_err(CudaMwdxError::from)?
        };
        unsafe {
            handle
                .buf
                .async_copy_to(&mut host_locked, &self.stream)
                .map_err(CudaMwdxError::from)?;
        }
        self.stream
            .synchronize()
            .map_err(CudaMwdxError::from)?;
        out_flat.copy_from_slice(host_locked.as_slice());
        Ok(())
    }

    pub fn mwdx_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &MwdxParams,
    ) -> Result<DeviceArrayF32Mwdx, CudaMwdxError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        // VRAM estimate (+64MB headroom)
        let bytes_prices = num_series * series_len * std::mem::size_of::<f32>();
        let bytes_first = num_series * std::mem::size_of::<i32>();
        let bytes_out = num_series * series_len * std::mem::size_of::<f32>();
        let required = bytes_prices
            .checked_add(bytes_first)
            .and_then(|x| x.checked_add(bytes_out))
            .ok_or_else(|| CudaMwdxError::InvalidInput("byte size overflow".into()))?;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaMwdxError::OutOfMemory {
                    required,
                    free,
                    headroom: 64 * 1024 * 1024,
                });
            } else {
                return Err(CudaMwdxError::InvalidInput(
                    "insufficient VRAM (mem_get_info unavailable)".into(),
                ));
            }
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            prepared.factor,
            num_series,
            series_len,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32Mwdx {
            buf: d_out,
            rows: series_len,
            cols: num_series,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn mwdx_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        factor: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMwdxError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaMwdxError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if d_prices_tm.len() != num_series * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "time-major prices length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaMwdxError::InvalidInput(
                "first_valids length must equal num_series".into(),
            ));
        }
        if d_out_tm.len() != num_series * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }
        if !factor.is_finite() || factor <= 0.0 {
            return Err(CudaMwdxError::InvalidInput(
                "factor must be positive and finite".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            factor,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    pub fn mwdx_many_series_one_param_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &MwdxParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaMwdxError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.mwdx_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;

        let mut host_locked: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(out_tm.len())? };
        unsafe {
            handle
                .buf
                .async_copy_to(&mut host_locked, &self.stream)
                .map_err(CudaMwdxError::from)?;
        }
        self.stream.synchronize().map_err(CudaMwdxError::from)?;
        out_tm.copy_from_slice(host_locked.as_slice());
        Ok(())
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_factors: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMwdxError> {
        if n_combos == 0 {
            return Ok(());
        }

        // Policy: only Plain makes sense for a sequential recurrence; choose block_x
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
            BatchKernelPolicy::Tiled { .. } => 256, // tiled not applicable; treat as plain
        };

        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaMwdx;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        let func = self
            .module
            .get_function("mwdx_batch_f32")
            .map_err(|_| CudaMwdxError::MissingKernelSymbol { name: "mwdx_batch_f32" })?;

        // Hint driver to persist `prices` in L2 across combo CTAs.
        unsafe {
            self.try_enable_persisting_l2_for_prices(d_prices, series_len);
        }

        // Chunk grid.y to avoid 65k limit; shift pointers per-chunk
        for (start, len) in Self::grid_y_chunks(n_combos) {
            let grid: GridSize = (1, len as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let prices_ptr = d_prices.as_device_ptr().add(0);
                let facs_ptr = d_factors.as_device_ptr().add(start);
                let out_ptr = d_out.as_device_ptr().add(start * series_len);
                let mut prices_raw = prices_ptr.as_raw();
                let mut facs_raw = facs_ptr.as_raw();
                let mut out_raw = out_ptr.as_raw();
                let mut series_len_i = series_len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut combos_i = len as i32;
                let mut args: [*mut c_void; 6] = [
                    &mut prices_raw as *mut _ as *mut c_void,
                    &mut facs_raw as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut out_raw as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, &mut args)?;
            }
        }

        self.stream.synchronize().map_err(CudaMwdxError::from)
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        factor: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMwdxError> {
        // Pick 1D vs 2D tiled kernel based on policy and shape
        let (use_tiled, tx, ty) = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {
                if series_len >= 4096 && num_series >= 2 {
                    (true, 128u32, if num_series % 4 == 0 { 4 } else { 2 })
                } else {
                    (false, 256, 1)
                }
            }
            ManySeriesKernelPolicy::OneD { block_x } => (false, block_x.max(64).min(1024), 1),
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => (true, tx, ty),
        };

        if use_tiled {
            let func_name = match (tx, ty) {
                (128, 2) => "mwdx_many_series_one_param_tiled2d_f32_tx128_ty2",
                (128, 4) => "mwdx_many_series_one_param_tiled2d_f32_tx128_ty4",
                _ => "mwdx_many_series_one_param_tiled2d_f32_tx128_ty2",
            };
            let func = self
                .module
                .get_function(func_name)
                .map_err(|_| CudaMwdxError::MissingKernelSymbol { name: func_name })?;

            // Introspection
            unsafe {
                let this = self as *const _ as *mut CudaMwdx;
                (*this).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
            }
            self.maybe_log_many_debug();

            // Do not tile across time for a recurrence; keep grid.x = 1.
            let grid_y = ((num_series as u32) + ty - 1) / ty;
            let grid: GridSize = (1, grid_y, 1).into();
            let block: BlockSize = (tx, ty, 1).into();
            unsafe {
                let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
                let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut factor_f32 = factor;
                let mut num_series_i = num_series as i32;
                let mut series_len_i = series_len as i32;
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 6] = [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut factor_f32 as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, &mut args)?;
            }
        } else {
            let func = self
                .module
                .get_function("mwdx_many_series_one_param_f32")
                .map_err(|_| CudaMwdxError::MissingKernelSymbol { name: "mwdx_many_series_one_param_f32" })?;

            // Introspection
            unsafe {
                let this = self as *const _ as *mut CudaMwdx;
                (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x: tx });
            }
            self.maybe_log_many_debug();

            let grid: GridSize = (num_series as u32, 1, 1).into();
            let block: BlockSize = (tx, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
                let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut factor_f32 = factor;
                let mut num_series_i = num_series as i32;
                let mut series_len_i = series_len as i32;
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let mut args: [*mut c_void; 6] = [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut factor_f32 as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, &mut args)?;
            }
        }

        self.stream.synchronize().map_err(CudaMwdxError::from)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &MwdxBatchRange,
    ) -> Result<PreparedMwdxBatch, CudaMwdxError> {
        if data_f32.is_empty() {
            return Err(CudaMwdxError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_grid_mwdx(sweep);
        if combos.is_empty() {
            return Err(CudaMwdxError::InvalidInput(
                "no factor combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaMwdxError::InvalidInput("all values are NaN".into()))?;
        if first_valid >= series_len {
            return Err(CudaMwdxError::InvalidInput(
                "first_valid index equals series length".into(),
            ));
        }

        let mut factors_f32 = Vec::with_capacity(combos.len());
        for params in &combos {
            let factor = params.factor.unwrap_or(0.2);
            if !factor.is_finite() || factor <= 0.0 {
                return Err(CudaMwdxError::InvalidInput(format!(
                    "factor must be positive and finite (got {})",
                    factor
                )));
            }
            let val2 = (2.0 / factor) - 1.0;
            if val2 + 1.0 <= 0.0 {
                return Err(CudaMwdxError::InvalidInput(format!(
                    "invalid denominator for factor {}",
                    factor
                )));
            }
            factors_f32.push(factor as f32);
        }

        Ok(PreparedMwdxBatch {
            combos,
            first_valid,
            series_len,
            factors_f32,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &MwdxParams,
    ) -> Result<PreparedMwdxManySeries, CudaMwdxError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaMwdxError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }

        let factor = params.factor.unwrap_or(0.2);
        if !factor.is_finite() || factor <= 0.0 {
            return Err(CudaMwdxError::InvalidInput(
                "factor must be positive and finite".into(),
            ));
        }
        let val2 = (2.0 / factor) - 1.0;
        if val2 + 1.0 <= 0.0 {
            return Err(CudaMwdxError::InvalidInput(
                "invalid denominator derived from factor".into(),
            ));
        }

        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut first = None;
            for t in 0..series_len {
                let value = data_tm_f32[t * num_series + series];
                if value.is_finite() {
                    first = Some(t as i32);
                    break;
                }
            }
            let idx = first.ok_or_else(|| {
                CudaMwdxError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            first_valids.push(idx);
        }

        Ok(PreparedMwdxManySeries {
            first_valids,
            factor: factor as f32,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::mwdx::{MwdxBatchRange, MwdxParams};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct MwdxBatchDevState {
        cuda: CudaMwdx,
        d_prices: DeviceBuffer<f32>,
        d_factors: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MwdxBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .mwdx_batch_device(
                    &self.d_prices,
                    &self.d_factors,
                    self.series_len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("mwdx batch kernel");
            self.cuda.stream.synchronize().expect("mwdx sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMwdx::new(0).expect("cuda mwdx");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = MwdxBatchRange {
            factor: (0.05, 0.05 + (PARAM_SWEEP as f64 - 1.0) * 0.001, 0.001),
        };
        let prepared = CudaMwdx::prepare_batch_inputs(&price, &sweep).expect("mwdx prepare batch");
        let n_combos = prepared.combos.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_factors = DeviceBuffer::from_slice(&prepared.factors_f32).expect("d_factors");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prepared.series_len * n_combos) }.expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(MwdxBatchDevState {
            cuda,
            d_prices,
            d_factors,
            series_len: prepared.series_len,
            n_combos,
            first_valid: prepared.first_valid,
            d_out,
        })
    }

    struct MwdxManyDevState {
        cuda: CudaMwdx,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        factor: f32,
        cols: usize,
        rows: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MwdxManyDevState {
        fn launch(&mut self) {
            self.cuda
                .mwdx_many_series_one_param_device(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.factor,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("mwdx many-series kernel");
            self.cuda.stream.synchronize().expect("mwdx sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaMwdx::new(0).expect("cuda mwdx");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = MwdxParams { factor: Some(0.2) };
        let prepared =
            CudaMwdx::prepare_many_series_inputs(&data_tm, cols, rows, &params).expect("mwdx prep");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(MwdxManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            factor: prepared.factor,
            cols,
            rows,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "mwdx",
                "one_series_many_params",
                "mwdx_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "mwdx",
                "many_series_one_param",
                "mwdx_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

// ---------------- Policy, debugging, and utilities (ALMA-style) ----------------

#[derive(Clone, Copy, Debug)]
pub struct CudaMwdxPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaMwdxPolicy {
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
    Tiled2D { tx: u32, ty: u32 },
}

impl CudaMwdx {
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
    #[inline]
    fn grid_y_chunks(n_combos: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_Y: usize = 65_535;
        (0..n_combos).step_by(MAX_GRID_Y).map(move |start| {
            let len = (n_combos - start).min(MAX_GRID_Y);
            (start, len)
        })
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
                    eprintln!("[DEBUG] MWDX batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMwdx)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] MWDX many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMwdx)).debug_many_logged = true;
                }
            }
        }
    }
}
