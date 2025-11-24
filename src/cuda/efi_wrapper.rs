//! CUDA support for Elder's Force Index (EFI)
//!
//! Matches ALMA wrapper parity: policy enums, NON_BLOCKING stream, PTX
//! load with DetermineTargetFromContext + OptLevel O2 (with fallbacks),
//! VRAM checks, and simple batch/many-series entry points.
//!
//! Math pattern: recurrence/IIR (EMA over price-diff × volume). We compute
//! per-step `diff = (p[t] - p[t-1]) * v[t]` and apply `prev = prev + alpha*(diff-prev)`.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32; // Reuse common VRAM handle
use crate::indicators::efi::{EfiBatchRange, EfiParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaEfiError {
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
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

// -------- Policies (knobs kept simple; IIR scan is sequential) --------

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain {
        block_x: u32,
    },
}
#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaEfiPolicy {
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

pub struct CudaEfi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaEfiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEfi {
    pub fn new(device_id: usize) -> Result<Self, CudaEfiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/efi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaEfiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn new_with_policy(device_id: usize, policy: CudaEfiPolicy) -> Result<Self, CudaEfiError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaEfiError> { self.stream.synchronize().map_err(Into::into) }

    #[inline]
    fn validate_launch(grid: GridSize, block: BlockSize) -> Result<(), CudaEfiError> {
        let (gx, gy, gz) = (grid.x, grid.y, grid.z);
        let (bx, by, bz) = (block.x, block.y, block.z);
        let threads = (bx as u64) * (by as u64) * (bz as u64);
        if threads > 1024 { // conservative default limit
            return Err(CudaEfiError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    // ---------- Public device entry points ----------

    pub fn efi_batch_dev(
        &self,
        prices_f32: &[f32],
        volumes_f32: &[f32],
        sweep: &EfiBatchRange,
    ) -> Result<DeviceArrayF32, CudaEfiError> {
        let mut prepared = Self::prepare_batch_inputs(prices_f32, volumes_f32, sweep)?;

        // Build diffs directly into pinned host memory (faster HtoD)
        let mut h_diffs = unsafe { LockedBuffer::<f32>::uninitialized(prepared.series_len) }
            .map_err(CudaEfiError::Cuda)?;
        {
            let diffs = unsafe { h_diffs.as_mut_slice() };
            diffs.fill(f32::NAN);
            for t in prepared.warm..prepared.series_len {
                let pc = prices_f32[t];
                let pp = prices_f32[t - 1];
                let vc = volumes_f32[t];
                if pc.is_finite() && pp.is_finite() && vc.is_finite() {
                    diffs[t] = (pc - pp) * vc;
                }
            }
        }

        // VRAM estimate + async copies (checked to avoid overflow)
        let prices_bytes = prepared
            .series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("prices_bytes overflow".into()))?;
        let params_bytes_periods = prepared
            .periods_i32
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("periods_bytes overflow".into()))?;
        let params_bytes_alphas = prepared
            .alphas_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("alphas_bytes overflow".into()))?;
        let params_bytes = params_bytes_periods
            .checked_add(params_bytes_alphas)
            .ok_or_else(|| CudaEfiError::InvalidInput("params_bytes overflow".into()))?;
        let out_elems = prepared
            .series_len
            .checked_mul(prepared.combos.len())
            .ok_or_else(|| CudaEfiError::InvalidInput("output elements overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaEfiError::InvalidInput("total VRAM size overflow".into()))?;
        Self::ensure_fit(required, 64 * 1024 * 1024)?;

        let mut d_diffs: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(prepared.series_len, &self.stream)
                .map_err(CudaEfiError::Cuda)?
        };
        unsafe {
            d_diffs
                .async_copy_from(h_diffs.as_slice(), &self.stream)
                .map_err(CudaEfiError::Cuda)?;
        }
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&prepared.periods_i32, &self.stream).map_err(CudaEfiError::Cuda)? };
        let d_alphas = unsafe { DeviceBuffer::from_slice_async(&prepared.alphas_f32, &self.stream).map_err(CudaEfiError::Cuda)? };
        let out_len = prepared
            .combos
            .len()
            .checked_mul(prepared.series_len)
            .ok_or_else(|| CudaEfiError::InvalidInput("output elements overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(out_len, &self.stream)
                .map_err(CudaEfiError::Cuda)?
        };

        self.launch_batch_kernel_with_diffs(
            &d_diffs,
            &d_periods,
            &d_alphas,
            prepared.series_len,
            prepared.warm,
            prepared.combos.len(),
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows: prepared.combos.len(), cols: prepared.series_len })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn efi_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        warm: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEfiError> {
        if prices_vol_ok(d_prices, d_volumes, series_len).is_err() {
            return Err(CudaEfiError::InvalidInput(
                "prices/volumes length mismatch".into(),
            ));
        }
        if d_periods.len() != n_combos || d_alphas.len() != n_combos {
            return Err(CudaEfiError::InvalidInput(
                "period/alpha buffers must match n_combos".into(),
            ));
        }
        if let Some(exp) = n_combos.checked_mul(series_len) { if d_out.len() != exp { return Err(CudaEfiError::InvalidInput("output length mismatch".into())); } } else { return Err(CudaEfiError::InvalidInput("rows*cols overflow".into())); }
        // Ensure current device matches wrapper device
        unsafe { let mut cur: i32 = 0; let _ = cust::sys::cuCtxGetDevice(&mut cur); if cur as u32 != self.device_id { return Err(CudaEfiError::DeviceMismatch { buf: self.device_id, current: cur as u32 }); } }
        self.launch_batch_kernel(
            d_prices, d_volumes, d_periods, d_alphas, series_len, warm, n_combos, d_out,
        )
    }

    pub fn efi_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EfiParams,
    ) -> Result<DeviceArrayF32, CudaEfiError> {
        let prepared = Self::prepare_many_series_inputs(
            prices_tm_f32,
            volumes_tm_f32,
            num_series,
            series_len,
            params,
        )?;

        let elems_series = num_series
            .checked_mul(series_len)
            .ok_or_else(|| CudaEfiError::InvalidInput("series elements overflow".into()))?;
        let prices_bytes = elems_series
            .checked_mul(std::mem::size_of::<f32>() * 2)
            .ok_or_else(|| CudaEfiError::InvalidInput("prices_bytes overflow".into()))?;
        let params_bytes = prepared
            .first_valids_diff
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("params_bytes overflow".into()))?;
        let out_bytes = elems_series
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaEfiError::InvalidInput("total VRAM size overflow".into()))?;
        Self::ensure_fit(required, 64 * 1024 * 1024)?;

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(prices_tm_f32, &self.stream)
                .map_err(CudaEfiError::Cuda)?
        };
        let d_volumes = unsafe {
            DeviceBuffer::from_slice_async(volumes_tm_f32, &self.stream)
                .map_err(CudaEfiError::Cuda)?
        };
        let d_first = unsafe {
            DeviceBuffer::from_slice_async(&prepared.first_valids_diff, &self.stream)
                .map_err(CudaEfiError::Cuda)?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(elems_series, &self.stream)
                .map_err(CudaEfiError::Cuda)?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_volumes,
            &d_first,
            prepared.period,
            prepared.alpha,
            num_series,
            series_len,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows: series_len, cols: num_series })
    }

    // ---------- Kernel launches ----------

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_volumes: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        warm: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEfiError> {
        let func = self
            .module
            .get_function("efi_batch_f32")
            .map_err(|_| CudaEfiError::MissingKernelSymbol { name: "efi_batch_f32" })?;

        // Policy: 1 block per combo; thread 0 does the scan
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
            BatchKernelPolicy::Auto => 128,
        };
        let grid = GridSize::x(n_combos as u32);
        let block = BlockSize::x(block_x);
        Self::validate_launch(grid, block)?;

        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_prices.as_device_ptr(),
                    d_volumes.as_device_ptr(),
                    d_periods.as_device_ptr(),
                    d_alphas.as_device_ptr(),
                    series_len as i32,
                    warm as i32,
                    n_combos as i32,
                    d_out.as_device_ptr()
                )
            )
            .map_err(CudaEfiError::Cuda)?;
        }

        unsafe {
            (*(self as *const _ as *mut CudaEfi)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    /// Fast path for one price series × many params with time‑major output.
    /// Returns DeviceArrayF32 with rows=series_len, cols=n_combos.
    pub fn efi_batch_time_major_dev(
        &self,
        prices_f32: &[f32],
        volumes_f32: &[f32],
        sweep: &EfiBatchRange,
    ) -> Result<DeviceArrayF32, CudaEfiError> {
        let mut prepared = Self::prepare_batch_inputs(prices_f32, volumes_f32, sweep)?;
        let n = prepared.series_len;

        // Build diffs into pinned host buffer
        let mut h_diffs = unsafe { LockedBuffer::<f32>::uninitialized(n) }.map_err(CudaEfiError::Cuda)?;
        {
            let diffs = unsafe { h_diffs.as_mut_slice() };
            diffs.fill(f32::NAN);
            for t in prepared.warm..n {
                let pc = prices_f32[t];
                let pp = prices_f32[t - 1];
                let vc = volumes_f32[t];
                if pc.is_finite() && pp.is_finite() && vc.is_finite() {
                    diffs[t] = (pc - pp) * vc;
                }
            }
        }

        // VRAM estimate (checked)
        let params_bytes_periods = prepared
            .periods_i32
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("periods_bytes overflow".into()))?;
        let params_bytes_alphas = prepared
            .alphas_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("alphas_bytes overflow".into()))?;
        let params_bytes = params_bytes_periods
            .checked_add(params_bytes_alphas)
            .ok_or_else(|| CudaEfiError::InvalidInput("params_bytes overflow".into()))?;
        let series_bytes = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("series bytes overflow".into()))?;
        let out_elems = n
            .checked_mul(prepared.combos.len())
            .ok_or_else(|| CudaEfiError::InvalidInput("output elements overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaEfiError::InvalidInput("output bytes overflow".into()))?;
        let required = series_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaEfiError::InvalidInput("total VRAM size overflow".into()))?;
        Self::ensure_fit(required, 64 * 1024 * 1024)?;

        // Upload
        let mut d_diffs: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(n, &self.stream).map_err(CudaEfiError::Cuda)?
        };
        unsafe {
            d_diffs.async_copy_from(h_diffs.as_slice(), &self.stream).map_err(CudaEfiError::Cuda)?;
        }
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&prepared.periods_i32, &self.stream).map_err(CudaEfiError::Cuda)? };
        let d_alphas = unsafe {
            DeviceBuffer::from_slice_async(&prepared.alphas_f32, &self.stream)
                .map_err(CudaEfiError::Cuda)?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(out_elems, &self.stream)
                .map_err(CudaEfiError::Cuda)?
        };

        self.launch_batch_kernel_time_major_from_diffs(
            &d_diffs,
            &d_periods,
            &d_alphas,
            n,
            prepared.warm,
            prepared.combos.len(),
            &mut d_out,
        )?;

        self.stream.synchronize().map_err(CudaEfiError::Cuda)?;

        Ok(DeviceArrayF32 { buf: d_out, rows: n, cols: prepared.combos.len() })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel_time_major_from_diffs(
        &self,
        d_diffs: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        warm: usize,
        n_combos: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEfiError> {
        let func = self
            .module
            .get_function("efi_one_series_many_params_from_diff_tm_f32")
            .or_else(|_| self.module.get_function("efi_one_series_many_params_from_diff_rm_f32"))
            .or_else(|_| self.module.get_function("efi_batch_from_diff_f32"))
            .map_err(|_| CudaEfiError::MissingKernelSymbol { name: "efi_batch_from_diff_f32" })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
            BatchKernelPolicy::Auto => 256,
        };
        let grid_x = ((n_combos + block_x as usize - 1) / block_x as usize) as u32;
        let grid = GridSize::x(grid_x);
        let block = BlockSize::x(block_x);

        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_diffs.as_device_ptr(),
                    d_periods.as_device_ptr(),
                    d_alphas.as_device_ptr(),
                    series_len as i32,
                    warm as i32,
                    n_combos as i32,
                    d_out_tm.as_device_ptr()
                )
            )
            .map_err(CudaEfiError::Cuda)?;
        }
        unsafe { (*(self as *const _ as *mut CudaEfi)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
        self.maybe_log_batch_debug();
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel_with_diffs(
        &self,
        d_diffs: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        warm: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEfiError> {
        // Prefer optimized row-major kernel; fall back to legacy symbol
        let func = self
            .module
            .get_function("efi_one_series_many_params_from_diff_rm_f32")
            .or_else(|_| self.module.get_function("efi_batch_from_diff_f32"))
            .map_err(|_| CudaEfiError::MissingKernelSymbol { name: "efi_batch_from_diff_f32" })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
            BatchKernelPolicy::Auto => 256,
        };
        let grid_x = ((n_combos + block_x as usize - 1) / block_x as usize) as u32;
        let grid = GridSize::x(grid_x);
        let block = BlockSize::x(block_x);

        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_diffs.as_device_ptr(),
                    d_periods.as_device_ptr(),
                    d_alphas.as_device_ptr(),
                    series_len as i32,
                    warm as i32,
                    n_combos as i32,
                    d_out.as_device_ptr()
                )
            )
            .map_err(CudaEfiError::Cuda)?;
        }

        unsafe {
            (*(self as *const _ as *mut CudaEfi)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_volumes_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEfiError> {
        let func = self
            .module
            .get_function("efi_many_series_one_param_f32")
            .map_err(|_| CudaEfiError::MissingKernelSymbol { name: "efi_many_series_one_param_f32" })?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
            ManySeriesKernelPolicy::Auto => 256,
        };
        // One thread per series across grid
        let grid_x = ((num_series + block_x as usize - 1) / block_x as usize) as u32;
        let grid = GridSize::x(grid_x);
        let block = BlockSize::x(block_x);

        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_prices_tm.as_device_ptr(),
                    d_volumes_tm.as_device_ptr(),
                    d_first_valids.as_device_ptr(),
                    period as i32,
                    alpha as f32,
                    num_series as i32,
                    series_len as i32,
                    d_out_tm.as_device_ptr()
                )
            )
            .map_err(CudaEfiError::Cuda)?;
        }

        unsafe {
            (*(self as *const _ as *mut CudaEfi)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    // ---------- Prep helpers ----------

    fn prepare_batch_inputs(
        prices_f32: &[f32],
        volumes_f32: &[f32],
        sweep: &EfiBatchRange,
    ) -> Result<PreparedEfiBatch, CudaEfiError> {
        if prices_f32.len() != volumes_f32.len() || prices_f32.is_empty() {
            return Err(CudaEfiError::InvalidInput(
                "prices and volumes must have same non-zero length".into(),
            ));
        }
        let series_len = prices_f32.len();
        let mut warm = None;
        // find first index t >= 1 where p[t], p[t-1], v[t] are finite
        for t in 1..series_len {
            if prices_f32[t].is_finite()
                && prices_f32[t - 1].is_finite()
                && volumes_f32[t].is_finite()
            {
                warm = Some(t);
                break;
            }
        }
        let warm = warm.ok_or_else(|| CudaEfiError::InvalidInput("all values NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaEfiError::InvalidInput("empty period sweep".into()));
        }
        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut alphas_f32 = Vec::with_capacity(combos.len());
        for prm in &combos {
            let p = prm.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaEfiError::InvalidInput("period must be positive".into()));
            }
            periods_i32.push(p as i32);
            alphas_f32.push(2.0f32 / (p as f32 + 1.0f32));
        }

        Ok(PreparedEfiBatch {
            combos,
            series_len,
            warm,
            periods_i32,
            alphas_f32,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        volumes_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EfiParams,
    ) -> Result<PreparedEfiManySeries, CudaEfiError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaEfiError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        let expected = num_series
            .checked_mul(series_len)
            .ok_or_else(|| CudaEfiError::InvalidInput("num_series*series_len overflow".into()))?;
        if prices_tm_f32.len() != volumes_tm_f32.len()
            || prices_tm_f32.len() != expected
        {
            return Err(CudaEfiError::InvalidInput(
                "time-major price/volume length mismatch".into(),
            ));
        }

        let period = params.period.unwrap_or(13) as i32;
        if period <= 0 {
            return Err(CudaEfiError::InvalidInput("period must be positive".into()));
        }
        let alpha = 2.0f32 / (period as f32 + 1.0f32);

        // Build first_valid_diff per series
        let mut first_valids_diff = vec![0i32; num_series];
        for s in 0..num_series {
            let mut found = None;
            for t in 1..series_len {
                let pc = prices_tm_f32[t * num_series + s];
                let pp = prices_tm_f32[(t - 1) * num_series + s];
                let vc = volumes_tm_f32[t * num_series + s];
                if pc.is_finite() && pp.is_finite() && vc.is_finite() {
                    found = Some(t as i32);
                    break;
                }
            }
            first_valids_diff[s] = found.ok_or_else(|| {
                CudaEfiError::InvalidInput(format!("series {} contains no valid diff", s))
            })?;
        }

        Ok(PreparedEfiManySeries {
            first_valids_diff,
            period,
            alpha,
            num_series,
            series_len,
        })
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
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn ensure_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaEfiError> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaEfiError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else { Ok(()) }
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged || std::env::var("BENCH_DEBUG").ok().as_deref() != Some("1") {
            return;
        }
        if let Some(sel) = self.last_batch {
            eprintln!("[DEBUG] EFI batch selected kernel: {:?}", sel);
            unsafe {
                (*(self as *const _ as *mut CudaEfi)).debug_batch_logged = true;
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged || std::env::var("BENCH_DEBUG").ok().as_deref() != Some("1") {
            return;
        }
        if let Some(sel) = self.last_many {
            eprintln!("[DEBUG] EFI many-series selected kernel: {:?}", sel);
            unsafe {
                (*(self as *const _ as *mut CudaEfi)).debug_many_logged = true;
            }
        }
    }
}

fn prices_vol_ok(
    d_prices: &DeviceBuffer<f32>,
    d_volumes: &DeviceBuffer<f32>,
    series_len: usize,
) -> Result<(), ()> {
    if d_prices.len() != series_len || d_volumes.len() != series_len {
        return Err(());
    }
    Ok(())
}

struct PreparedEfiBatch {
    combos: Vec<EfiParams>,
    series_len: usize,
    warm: usize,
    periods_i32: Vec<i32>,
    alphas_f32: Vec<f32>,
}
struct PreparedEfiManySeries {
    first_valids_diff: Vec<i32>,
    period: i32,
    alpha: f32,
    num_series: usize,
    series_len: usize,
}

fn expand_grid(r: &EfiBatchRange) -> Vec<EfiParams> {
    fn axis_u((s, e, st): (usize, usize, usize)) -> Vec<usize> {
        if st == 0 || s == e { return vec![s]; }
        let mut v = Vec::new();
        if s < e {
            let mut x = s; while x <= e { v.push(x); match x.checked_add(st) { Some(n) => { if n == x { break; } x = n; }, None => break } }
        } else {
            let mut x = s; loop { v.push(x); if x <= e { break; } let n = x.saturating_sub(st); if n == x { break; } x = n; }
        }
        v
    }
    axis_u(r.period)
        .into_iter()
        .map(|p| EfiParams { period: Some(p) })
        .collect()
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        // Batch: one-series × many-params
        v.push(CudaBenchScenario::new(
            "efi",
            "one_series_many_params",
            "efi_cuda_batch_dev",
            "100k_x_64",
            || {
                struct State {
                    cuda: CudaEfi,
                    prices: Vec<f32>,
                    volumes: Vec<f32>,
                    sweep: EfiBatchRange,
                }
                impl CudaBenchState for State {
                    fn launch(&mut self) {
                        let _ = self
                            .cuda
                            .efi_batch_dev(&self.prices, &self.volumes, &self.sweep);
                    }
                }
                let n = 100_000usize;
                let mut p = vec![f32::NAN; n];
                let mut vv = vec![f32::NAN; n];
                for i in 1..n {
                    let x = i as f32;
                    p[i] = (x * 0.00123).sin() + 0.00017 * x;
                    vv[i] = (x * 0.00077).cos().abs() + 0.5;
                }
                let sweep = EfiBatchRange {
                    period: (8, 8 + 63, 1),
                };
                let cuda = CudaEfi::new(0).unwrap();
                Box::new(State {
                    cuda,
                    prices: p,
                    volumes: vv,
                    sweep,
                })
            },
        ));
        // Many-series × one-param
        v.push(CudaBenchScenario::new(
            "efi",
            "many_series_one_param",
            "efi_cuda_many_series_one_param_dev",
            "64x4096",
            || {
                struct State {
                    cuda: CudaEfi,
                    tm_p: Vec<f32>,
                    tm_v: Vec<f32>,
                    cols: usize,
                    rows: usize,
                    prm: EfiParams,
                }
                impl CudaBenchState for State {
                    fn launch(&mut self) {
                        let _ = self.cuda.efi_many_series_one_param_time_major_dev(
                            &self.tm_p, &self.tm_v, self.cols, self.rows, &self.prm,
                        );
                    }
                }
                let cols = 64usize;
                let rows = 4096usize;
                let mut tm_p = vec![f32::NAN; rows * cols];
                let mut tm_v = vec![f32::NAN; rows * cols];
                for s in 0..cols {
                    for t in 1..rows {
                        let x = (t as f32) + (s as f32) * 0.3;
                        tm_p[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
                        tm_v[t * cols + s] = (x * 0.001).cos().abs() + 0.4;
                    }
                }
                let prm = EfiParams { period: Some(13) };
                let cuda = CudaEfi::new(0).unwrap();
                Box::new(State {
                    cuda,
                    tm_p,
                    tm_v,
                    cols,
                    rows,
                    prm,
                })
            },
        ));
        v
    }
}
