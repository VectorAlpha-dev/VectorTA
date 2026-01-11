//! CUDA support for VIDYA (Variable Index Dynamic Average).
//!
//! Mirrors the ALMA/EMA wrapper patterns:
//! - PTX load via OUT_DIR with DetermineTargetFromContext and OptLevel::O2, with fallbacks
//! - NON_BLOCKING stream
//! - VRAM estimation with ~64MB headroom and grid chunking
//! - Policy enums for batch and many-series variants
//!
//! Math pattern: recurrence/IIR with adaptive factor k = alpha * (short_std / long_std).
//! Warmup/writes match the scalar path in src/indicators/vidya.rs:
//! - warmup index = first_valid + long_period - 2
//! - out[warm-2] = price[warm-2]; out[warm-1] uses initial k

#![cfg(feature = "cuda")]

use super::alma_wrapper::{DeviceArrayF32, CudaAlmaError};
use crate::indicators::vidya::{VidyaBatchRange, VidyaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(thiserror::Error, Debug)]
pub enum CudaVidyaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

// -------- Policy (kept simple; one block per combo/series) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
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

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaVidyaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaVidya {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>,
    device_id: u32,
    policy: CudaVidyaPolicy,
    // Introspection
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    // Device caps
    max_grid_x: usize,
}

impl CudaVidya {
    pub fn new(device_id: usize) -> Result<Self, CudaVidyaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vidya_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
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

        let max_grid_x =
            device.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)? as usize;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaVidyaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            max_grid_x,
        })
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaVidyaError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    pub fn context_arc_clone(&self) -> std::sync::Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    // ---------- Batch (one-series × many-params) ----------

    pub fn vidya_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &VidyaBatchRange,
    ) -> Result<DeviceArrayF32, CudaVidyaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();
        let use_prefix = self.module.get_function("vidya_batch_prefix_f32").is_ok();

        let prices_bytes = prepared.series_len * std::mem::size_of::<f32>();
        let params_bytes =
            (prepared.short_i32.len() + prepared.long_i32.len()) * std::mem::size_of::<i32>()
                + prepared.alpha_f32.len() * std::mem::size_of::<f32>();
        let prefix_bytes = if use_prefix {
            let elems = prepared
                .series_len
                .checked_add(1)
                .ok_or_else(|| CudaVidyaError::InvalidInput("series_len+1 overflow".into()))?;
            elems
                .checked_mul(2)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
                .ok_or_else(|| CudaVidyaError::InvalidInput("prefix bytes overflow".into()))?
        } else {
            0
        };
        let out_elems = n_combos
            .checked_mul(prepared.series_len)
            .ok_or_else(|| CudaVidyaError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaVidyaError::InvalidInput("rows*cols*sizeof(f32) overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .and_then(|x| x.checked_add(prefix_bytes))
            .ok_or_else(|| CudaVidyaError::InvalidInput("VRAM size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaVidyaError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaVidyaError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)?
        };
        let d_short = unsafe {
            DeviceBuffer::from_slice_async(&prepared.short_i32, &self.stream)?
        };
        let d_long = unsafe {
            DeviceBuffer::from_slice_async(&prepared.long_i32, &self.stream)?
        };
        let d_alpha = unsafe {
            DeviceBuffer::from_slice_async(&prepared.alpha_f32, &self.stream)?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(n_combos * prepared.series_len, &self.stream)?
        };

        let mut d_prefix_sum: Option<DeviceBuffer<f64>> = None;
        let mut d_prefix_sum2: Option<DeviceBuffer<f64>> = None;
        if use_prefix {
            // Host prefix sums of x and x^2 (NaN treated as 0) to enable O(1) rolling stats.
            let mut prefix_sum: Vec<f64> = Vec::with_capacity(prepared.series_len + 1);
            let mut prefix_sum2: Vec<f64> = Vec::with_capacity(prepared.series_len + 1);
            prefix_sum.push(0.0f64);
            prefix_sum2.push(0.0f64);
            let mut acc = 0.0f64;
            let mut acc2 = 0.0f64;
            for &v in data_f32 {
                let x = if v.is_nan() { 0.0f64 } else { v as f64 };
                acc += x;
                acc2 += x * x;
                prefix_sum.push(acc);
                prefix_sum2.push(acc2);
            }

            d_prefix_sum = Some(unsafe {
                DeviceBuffer::from_slice_async(&prefix_sum, &self.stream)?
            });
            d_prefix_sum2 = Some(unsafe {
                DeviceBuffer::from_slice_async(&prefix_sum2, &self.stream)?
            });
            let d_prefix_sum_ref = d_prefix_sum.as_ref().ok_or_else(|| {
                CudaVidyaError::InvalidInput("failed to allocate prefix_sum buffer".into())
            })?;
            let d_prefix_sum2_ref = d_prefix_sum2.as_ref().ok_or_else(|| {
                CudaVidyaError::InvalidInput("failed to allocate prefix_sum2 buffer".into())
            })?;

            self.launch_batch_prefix_kernel(
                &d_prices,
                d_prefix_sum_ref,
                d_prefix_sum2_ref,
                &d_short,
                &d_long,
                &d_alpha,
                prepared.series_len,
                prepared.first_valid,
                n_combos,
                &mut d_out,
            )?;
        } else {
            self.launch_batch_kernel(
                &d_prices,
                &d_short,
                &d_long,
                &d_alpha,
                prepared.series_len,
                prepared.first_valid,
                n_combos,
                &mut d_out,
            )?;
        }

        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    // ---------- Many-series (time-major, one param) ----------

    pub fn vidya_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &VidyaParams,
    ) -> Result<DeviceArrayF32, CudaVidyaError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let elems = num_series
            .checked_mul(series_len)
            .ok_or_else(|| CudaVidyaError::InvalidInput("rows*cols overflow".into()))?;
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaVidyaError::InvalidInput("rows*cols*sizeof(f32) overflow".into()))?;
        let params_bytes = prepared
            .first_valids
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaVidyaError::InvalidInput("first_valids bytes overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaVidyaError::InvalidInput("rows*cols*sizeof(f32) overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaVidyaError::InvalidInput("VRAM size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaVidyaError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaVidyaError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)?
        };
        let d_first = unsafe {
            DeviceBuffer::from_slice_async(&prepared.first_valids, &self.stream)?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(num_series * series_len, &self.stream)?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.short as i32,
            prepared.long as i32,
            prepared.alpha as f32,
            num_series,
            series_len,
            &mut d_out,
        )?;

        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    // ---------- Internal launches ----------

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_short: &DeviceBuffer<i32>,
        d_long: &DeviceBuffer<i32>,
        d_alpha: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVidyaError> {
        if n_combos == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("vidya_batch_f32")
            .map_err(|_| CudaVidyaError::MissingKernelSymbol { name: "vidya_batch_f32" })?;

        let mut block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => env::var("VIDYA_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(256),
        };
        if block_x == 0 {
            block_x = 256;
        }
        unsafe {
            (*(self as *const _ as *mut CudaVidya)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Chunk by device grid.x cap
        let cap = self.max_grid_x.max(1).min(usize::MAX / 2);
        for (start, len) in Self::grid_chunks(n_combos, cap) {
            let gx = len as u32;
            let gy = 1u32;
            let gz = 1u32;
            let bx = block_x;
            let by = 1u32;
            let bz = 1u32;
            if gx == 0 || bx == 0 {
                return Err(CudaVidyaError::LaunchConfigTooLarge {
                    gx,
                    gy,
                    gz,
                    bx,
                    by,
                    bz,
                });
            }
            let grid: GridSize = (gx, gy, gz).into();
            let block: BlockSize = (bx, by, bz).into();
            let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };
            let short_ptr = unsafe { d_short.as_device_ptr().add(start) };
            let long_ptr = unsafe { d_long.as_device_ptr().add(start) };
            let alpha_ptr = unsafe { d_alpha.as_device_ptr().add(start) };
            let series_len_i = series_len as i32;
            let first_valid_i = first_valid as i32;
            let n_combos_i = len as i32;
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_prices.as_device_ptr(),
                        short_ptr,
                        long_ptr,
                        alpha_ptr,
                        series_len_i,
                        first_valid_i,
                        n_combos_i,
                        out_ptr
                    )
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_prefix_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_prefix_sum: &DeviceBuffer<f64>,
        d_prefix_sum2: &DeviceBuffer<f64>,
        d_short: &DeviceBuffer<i32>,
        d_long: &DeviceBuffer<i32>,
        d_alpha: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVidyaError> {
        if n_combos == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("vidya_batch_prefix_f32")
            .map_err(|_| CudaVidyaError::MissingKernelSymbol { name: "vidya_batch_prefix_f32" })?;

        // Warp-scan kernel: fixed one-warp block for best utilization.
        const BLOCK_X: u32 = 32;
        unsafe {
            (*(self as *const _ as *mut CudaVidya)).last_batch =
                Some(BatchKernelSelected::Plain { block_x: BLOCK_X });
        }
        self.maybe_log_batch_debug();

        let cap = self.max_grid_x.max(1).min(usize::MAX / 2);
        for (start, len) in Self::grid_chunks(n_combos, cap) {
            let grid: GridSize = (len as u32, 1u32, 1u32).into();
            let block: BlockSize = (BLOCK_X, 1u32, 1u32).into();

            let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };
            let short_ptr = unsafe { d_short.as_device_ptr().add(start) };
            let long_ptr = unsafe { d_long.as_device_ptr().add(start) };
            let alpha_ptr = unsafe { d_alpha.as_device_ptr().add(start) };

            let series_len_i = series_len as i32;
            let first_valid_i = first_valid as i32;
            let n_combos_i = len as i32;
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_prices.as_device_ptr(),
                        d_prefix_sum.as_device_ptr(),
                        d_prefix_sum2.as_device_ptr(),
                        short_ptr,
                        long_ptr,
                        alpha_ptr,
                        series_len_i,
                        first_valid_i,
                        n_combos_i,
                        out_ptr
                    )
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        short_period: i32,
        long_period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVidyaError> {
        if num_series == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("vidya_many_series_one_param_f32")
            .map_err(|_| CudaVidyaError::MissingKernelSymbol {
                name: "vidya_many_series_one_param_f32",
            })?;

        let mut block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => env::var("VIDYA_MS_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(128),
        };
        if block_x == 0 {
            block_x = 128;
        }
        unsafe {
            (*(self as *const _ as *mut CudaVidya)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        // One block per series (compat 1D launch)
        let gx = num_series as u32;
        let gy = 1u32;
        let gz = 1u32;
        let bx = block_x;
        let by = 1u32;
        let bz = 1u32;
        if gx == 0 || bx == 0 {
            return Err(CudaVidyaError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        let grid: GridSize = (gx, gy, gz).into();
        let block: BlockSize = (bx, by, bz).into();

        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_prices_tm.as_device_ptr(),
                    d_first_valids.as_device_ptr(),
                    short_period,
                    long_period,
                    alpha,
                    num_series as i32,
                    series_len as i32,
                    d_out_tm.as_device_ptr()
                )
            )?;
        }
        Ok(())
    }

    // ---------- Prep helpers ----------

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &VidyaBatchRange,
    ) -> Result<PreparedVidyaBatch, CudaVidyaError> {
        if data_f32.is_empty() {
            return Err(CudaVidyaError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaVidyaError::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }
        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaVidyaError::InvalidInput("all values are NaN".into()))?;
        let mut short_i32 = Vec::with_capacity(combos.len());
        let mut long_i32 = Vec::with_capacity(combos.len());
        let mut alpha_f32 = Vec::with_capacity(combos.len());
        for p in &combos {
            let sp = p.short_period.unwrap_or(0);
            let lp = p.long_period.unwrap_or(0);
            let a = p.alpha.unwrap_or(-1.0);
            if sp < 2 || lp < sp || lp < 2 || !(0.0..=1.0).contains(&a) {
                return Err(CudaVidyaError::InvalidInput(format!(
                    "invalid params: short={}, long={}, alpha={}",
                    sp, lp, a
                )));
            }
            if series_len - first_valid < lp {
                return Err(CudaVidyaError::InvalidInput(format!(
                    "not enough valid data: need {} valid samples, have {}",
                    lp,
                    series_len - first_valid
                )));
            }
            short_i32.push(sp as i32);
            long_i32.push(lp as i32);
            alpha_f32.push(a as f32);
        }
        Ok(PreparedVidyaBatch {
            combos,
            first_valid,
            series_len,
            short_i32,
            long_i32,
            alpha_f32,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &VidyaParams,
    ) -> Result<PreparedVidyaManySeries, CudaVidyaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaVidyaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len()
            != num_series
                .checked_mul(series_len)
                .ok_or_else(|| CudaVidyaError::InvalidInput("rows*cols overflow".into()))?
        {
            return Err(CudaVidyaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        let sp = params.short_period.unwrap_or(0);
        let lp = params.long_period.unwrap_or(0);
        let a = params.alpha.unwrap_or(-1.0);
        if sp < 2 || lp < sp || lp < 2 || !(0.0..=1.0).contains(&a) {
            return Err(CudaVidyaError::InvalidInput(format!(
                "invalid params: short={}, long={}, alpha={}",
                sp, lp, a
            )));
        }
        let mut first_valids = Vec::with_capacity(num_series);
        for s in 0..num_series {
            let mut fv: Option<usize> = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + s];
                if v.is_finite() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaVidyaError::InvalidInput(format!("series {} contains only NaNs", s))
            })?;
            let remain = series_len - fv;
            if remain < lp {
                return Err(CudaVidyaError::InvalidInput(format!(
                    "series {} does not have enough valid data: need {} valid samples, have {}",
                    s, lp, remain
                )));
            }
            first_valids.push(fv as i32);
        }
        Ok(PreparedVidyaManySeries {
            first_valids,
            short: sp,
            long: lp,
            alpha: a,
        })
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
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            return required_bytes + headroom <= free as usize;
        }
        true
    }
    #[inline]
    fn grid_chunks(total: usize, cap_x: usize) -> impl Iterator<Item = (usize, usize)> {
        struct It {
            total: usize,
            cap: usize,
            start: usize,
        }
        impl Iterator for It {
            type Item = (usize, usize);
            fn next(&mut self) -> Option<Self::Item> {
                if self.start >= self.total {
                    return None;
                }
                let remain = self.total - self.start;
                let len = remain.min(self.cap);
                let s = self.start;
                self.start += len;
                Some((s, len))
            }
        }
        It {
            total,
            cap: cap_x.max(1),
            start: 0,
        }
    }
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(BatchKernelSelected::Plain { block_x }) = self.last_batch {
                eprintln!("[VIDYA] batch kernel: Plain block_x={}", block_x);
            }
        }
        unsafe {
            (*(self as *const _ as *mut CudaVidya)).debug_batch_logged = true;
        }
    }
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(ManySeriesKernelSelected::OneD { block_x }) = self.last_many {
                eprintln!("[VIDYA] many-series kernel: OneD block_x={}", block_x);
            }
        }
        unsafe {
            (*(self as *const _ as *mut CudaVidya)).debug_many_logged = true;
        }
    }
}

// ---------- Prep structs ----------

struct PreparedVidyaBatch {
    combos: Vec<VidyaParams>,
    first_valid: usize,
    series_len: usize,
    short_i32: Vec<i32>,
    long_i32: Vec<i32>,
    alpha_f32: Vec<f32>,
}
struct PreparedVidyaManySeries {
    first_valids: Vec<i32>,
    short: usize,
    long: usize,
    alpha: f64,
}

fn expand_grid(r: &VidyaBatchRange) -> Vec<VidyaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let sp = axis_usize(r.short_period);
    let lp = axis_usize(r.long_period);
    let al = axis_f64(r.alpha);
    let mut out = Vec::with_capacity(sp.len() * lp.len() * al.len());
    for &s in &sp {
        for &l in &lp {
            for &a in &al {
                out.push(VidyaParams {
                    short_period: Some(s),
                    long_period: Some(l),
                    alpha: Some(a),
                });
            }
        }
    }
    out
}

// ---------- Benches ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * 4;
        let prefix_bytes = (ONE_SERIES_LEN + 1) * 2 * 8; // two f64 prefix buffers
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * 4;
        in_bytes + prefix_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * 4;
        let out_bytes = elems * 4;
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct VidyaBatchState {
        cuda: CudaVidya,
        d_prices: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        d_prefix_sum: DeviceBuffer<f64>,
        d_prefix_sum2: DeviceBuffer<f64>,
        d_short: DeviceBuffer<i32>,
        d_long: DeviceBuffer<i32>,
        d_alpha: DeviceBuffer<f32>,
        first_valid: usize,
        len: usize,
        combos: usize,
        warmed: bool,
    }
    impl CudaBenchState for VidyaBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_prefix_kernel(
                    &self.d_prices,
                    &self.d_prefix_sum,
                    &self.d_prefix_sum2,
                    &self.d_short,
                    &self.d_long,
                    &self.d_alpha,
                    self.len,
                    self.first_valid,
                    self.combos,
                    &mut self.d_out,
                )
                .expect("vidya batch launch");
            self.cuda.synchronize().expect("sync");
            if !self.warmed {
                self.warmed = true;
            }
        }
    }
    fn prep_vidya_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaVidya::new(0).expect("cuda vidya");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = VidyaBatchRange {
            short_period: (2, 2, 0),
            long_period: (10, 10 + PARAM_SWEEP - 1, 1),
            alpha: (0.2, 0.2, 0.0),
        };
        let combos = super::expand_grid(&sweep);
        let first_valid = price.iter().position(|&x| !x.is_nan()).unwrap_or(0);
        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");

        // Precompute prefix sums of x and x^2 once for the benchmark state.
        let mut prefix_sum: Vec<f64> = Vec::with_capacity(ONE_SERIES_LEN + 1);
        let mut prefix_sum2: Vec<f64> = Vec::with_capacity(ONE_SERIES_LEN + 1);
        prefix_sum.push(0.0f64);
        prefix_sum2.push(0.0f64);
        let mut acc = 0.0f64;
        let mut acc2 = 0.0f64;
        for &v in &price {
            let x = if v.is_nan() { 0.0f64 } else { v as f64 };
            acc += x;
            acc2 += x * x;
            prefix_sum.push(acc);
            prefix_sum2.push(acc2);
        }
        let d_prefix_sum = DeviceBuffer::from_slice(&prefix_sum).expect("d_prefix_sum");
        let d_prefix_sum2 = DeviceBuffer::from_slice(&prefix_sum2).expect("d_prefix_sum2");
        let short_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.short_period.unwrap() as i32)
            .collect();
        let long_i32: Vec<i32> = combos
            .iter()
            .map(|p| p.long_period.unwrap() as i32)
            .collect();
        let alpha_f32: Vec<f32> = combos.iter().map(|p| p.alpha.unwrap() as f32).collect();
        let d_short = DeviceBuffer::from_slice(&short_i32).expect("d_short");
        let d_long = DeviceBuffer::from_slice(&long_i32).expect("d_long");
        let d_alpha = DeviceBuffer::from_slice(&alpha_f32).expect("d_alpha");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(ONE_SERIES_LEN * combos.len()) }.expect("d_out");
        Box::new(VidyaBatchState {
            cuda,
            d_prices,
            d_out,
            d_prefix_sum,
            d_prefix_sum2,
            d_short,
            d_long,
            d_alpha,
            first_valid,
            len: ONE_SERIES_LEN,
            combos: combos.len(),
            warmed: false,
        })
    }

    struct VidyaManyState {
        cuda: CudaVidya,
        d_prices_tm: DeviceBuffer<f32>,
        d_first: DeviceBuffer<i32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        sp: i32,
        lp: i32,
        alpha: f32,
        warmed: bool,
    }
    impl CudaBenchState for VidyaManyState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first,
                    self.sp,
                    self.lp,
                    self.alpha,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("vidya many launch");
            self.cuda.synchronize().expect("sync");
            if !self.warmed {
                self.warmed = true;
            }
        }
    }
    fn prep_vidya_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaVidya::new(0).expect("cuda vidya");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let prices_tm = gen_time_major_prices(cols, rows);
        let sp = 2;
        let lp = 64;
        let alpha = 0.2f32;
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                if prices_tm[t * cols + s].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
        let d_first = DeviceBuffer::from_slice(&first_valids).expect("d_first");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        Box::new(VidyaManyState {
            cuda,
            d_prices_tm,
            d_first,
            d_out_tm,
            cols,
            rows,
            sp,
            lp,
            alpha,
            warmed: false,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "vidya",
                "one_series_many_params",
                "vidya_cuda_batch_dev",
                "1m_x_250",
                prep_vidya_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "vidya",
                "many_series_one_param",
                "vidya_cuda_many_series_one_param_dev",
                "256x1m",
                prep_vidya_many_series_one_param,
            )
            .with_sample_size(6)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
