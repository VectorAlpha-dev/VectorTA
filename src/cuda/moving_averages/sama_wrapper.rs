//! CUDA support for the Slope Adaptive Moving Average (SAMA).
//!
//! Mirrors the ALMA/CWMA CUDA API surface by exposing zero‑copy device entry
//! points for both the one‑series × many‑parameter sweep and the time‑major
//! many‑series × one‑parameter scenario. Kernels operate fully in FP32 and
//! reuse host‑prepared alpha coefficients to keep GPU‑side work focused on the
//! adaptive recurrence itself. Wrapper adds ALMA‑parity features: policy enums,
//! JIT options (DetermineTargetFromContext + O2 with fallbacks), NON_BLOCKING
//! stream, VRAM checks with ~64MB headroom, chunked launches, and BENCH_DEBUG
//! logging of selected kernels.

#![cfg(feature = "cuda")]

// Reuse shared policy enums exported by CWMA for ALMA‑parity API shape.
use super::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::indicators::moving_averages::sama::{SamaBatchRange, SamaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use thiserror::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Error)]
pub enum CudaSamaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch config too large: gx={gx} gy={gy} gz={gz} bx={bx} by={by} bz={bz}")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaSama {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    // Policy + introspection for parity with ALMA/CWMA wrappers
    policy: CudaSamaPolicy,
    last_batch: Option<SamaBatchKernelSelected>,
    last_many: Option<SamaManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

/// VRAM-backed array handle for SAMA with context guard and device id
pub struct DeviceArrayF32Sama {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Sama {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

// -------- Kernel selection policy (explicit for tests; Auto for production) --------

#[derive(Clone, Copy, Debug)]
pub struct CudaSamaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaSamaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

// -------- Introspection (selected kernel) --------

#[derive(Clone, Copy, Debug)]
pub enum SamaBatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum SamaManySeriesKernelSelected {
    OneD { block_x: u32 },
}

struct PreparedSamaBatch {
    combos: Vec<SamaParams>,
    first_valid: usize,
    series_len: usize,
    lengths_i32: Vec<i32>,
    min_alphas: Vec<f32>,
    maj_alphas: Vec<f32>,
    first_valids: Vec<i32>,
}

struct PreparedSamaManySeries {
    first_valids: Vec<i32>,
    length: i32,
    min_alpha: f32,
    maj_alpha: f32,
}

impl CudaSama {
    pub fn new(device_id: usize) -> Result<Self, CudaSamaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/sama_kernel.ptx"));
        // Match ALMA/CWMA JIT policy for broad driver support and perf
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
            _context: context,
            device_id: device_id as u32,
            policy: CudaSamaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create with an explicit policy (mirrors ALMA/CWMA convenience).
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaSamaPolicy,
    ) -> Result<Self, CudaSamaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaSamaPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaSamaPolicy {
        &self.policy
    }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<SamaBatchKernelSelected> {
        self.last_batch
    }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<SamaManySeriesKernelSelected> {
        self.last_many
    }

    pub fn sama_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SamaBatchRange,
    ) -> Result<DeviceArrayF32Sama, CudaSamaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        // VRAM estimate (prices + params + output). Note: first_valids filled on device.
        let prices_bytes = prepared
            .series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSamaError::InvalidInput("size overflow".into()))?;
        let params_each = std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>();
        let params_bytes = n_combos
            .checked_mul(params_each)
            .ok_or_else(|| CudaSamaError::InvalidInput("size overflow".into()))?;
        let out_bytes = n_combos
            .checked_mul(prepared.series_len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaSamaError::InvalidInput("size overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaSamaError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(required, headroom) {
            let (free, _) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaSamaError::OutOfMemory { required, free, headroom });
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let d_lengths = DeviceBuffer::from_slice(&prepared.lengths_i32)?;
        let d_min = DeviceBuffer::from_slice(&prepared.min_alphas)?;
        let d_maj = DeviceBuffer::from_slice(&prepared.maj_alphas)?;

        // Allocate device first_valids and fill it on-device (avoid H->D copy)
        let mut d_first: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(n_combos)? };
        memset_i32_async(&self.stream, &mut d_first, prepared.first_valid as i32)?;

        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(prepared.series_len * n_combos)? };

        // Compute max_window for this run; we will still compute per-slice values.
        let max_window_total: i32 = *prepared.lengths_i32.iter().max().unwrap_or(&0);

        self.launch_batch_kernel_sliced_opt(
            &d_prices,
            &d_lengths,
            &d_min,
            &d_maj,
            &d_first,
            prepared.series_len,
            n_combos,
            Some(&prepared.lengths_i32),
            max_window_total,
            &mut d_out,
        )?;

        // Synchronize producing stream so Python CAI may omit 'stream'
        self.stream.synchronize()?;

        Ok(DeviceArrayF32Sama {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
            ctx: Arc::clone(&self._context),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_min_alphas: &DeviceBuffer<f32>,
        d_maj_alphas: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSamaError> {
        if series_len == 0 {
            return Err(CudaSamaError::InvalidInput("series_len must be positive".into()));
        }
        if n_combos == 0 {
            return Err(CudaSamaError::InvalidInput("n_combos must be positive".into()));
        }
        if d_lengths.len() != n_combos
            || d_min_alphas.len() != n_combos
            || d_maj_alphas.len() != n_combos
            || d_first_valids.len() != n_combos
        {
            return Err(CudaSamaError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaSamaError::InvalidInput(
                "prices length must equal series_len".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaSamaError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        // Derive max_window with one small D->H copy to enable per-slice smem sizing
        let mut h_lengths = vec![0i32; n_combos];
        d_lengths.copy_to(&mut h_lengths)?;
        let max_window_total: i32 = *h_lengths.iter().max().unwrap_or(&0);

        self.launch_batch_kernel_sliced_opt(
            d_prices,
            d_lengths,
            d_min_alphas,
            d_maj_alphas,
            d_first_valids,
            series_len,
            n_combos,
            Some(&h_lengths),
            max_window_total,
            d_out,
        )
    }

    pub fn sama_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &SamaBatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaSamaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaSamaError::InvalidInput("output slice length mismatch".into()));
        }
        let handle = self.sama_batch_dev(data_f32, sweep)?;
        handle.buf.copy_to(out_flat).map_err(Into::into)
    }

    pub fn sama_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SamaParams,
    ) -> Result<DeviceArrayF32Sama, CudaSamaError> {
        let prepared = Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        // VRAM estimate (prices + first_valids + out)
        let prices_bytes = num_series
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaSamaError::InvalidInput("size overflow".into()))?;
        let fv_bytes = num_series
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaSamaError::InvalidInput("size overflow".into()))?;
        let out_bytes = num_series
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaSamaError::InvalidInput("size overflow".into()))?;
        let required = prices_bytes
            .checked_add(fv_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaSamaError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let (free, _) = Self::device_mem_info().unwrap_or((0, 0));
            return Err(CudaSamaError::OutOfMemory { required, free, headroom });
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&prepared.first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(num_series * series_len)? };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.length,
            prepared.min_alpha,
            prepared.maj_alpha,
            num_series,
            series_len,
            &mut d_out,
        )?;

        // Synchronize producing stream so Python CAI may omit 'stream'
        self.stream.synchronize()?;

        Ok(DeviceArrayF32Sama {
            buf: d_out,
            rows: series_len,
            cols: num_series,
            ctx: Arc::clone(&self._context),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sama_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        length: i32,
        min_alpha: f32,
        maj_alpha: f32,
        num_series: i32,
        series_len: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSamaError> {
        if num_series <= 0 || series_len <= 0 {
            return Err(CudaSamaError::InvalidInput("num_series and series_len must be positive".into()));
        }
        if length <= 0 {
            return Err(CudaSamaError::InvalidInput("length must be positive".into()));
        }
        if d_first_valids.len() != num_series as usize {
            return Err(CudaSamaError::InvalidInput("first_valids buffer length mismatch".into()));
        }
        let total = num_series as usize * series_len as usize;
        if d_prices_tm.len() != total || d_out_tm.len() != total {
            return Err(CudaSamaError::InvalidInput("time-major buffer length mismatch".into()));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            length,
            min_alpha,
            maj_alpha,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn sama_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SamaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSamaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaSamaError::InvalidInput("output slice length mismatch".into()));
        }
        let handle = self.sama_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle.buf.copy_to(out_tm).map_err(CudaSamaError::from)
    }

    // Launch helper that chunks across parameter combos to respect grid limits
    // and supports large sweeps. Each chunk updates param/output pointers.
    fn launch_batch_kernel_sliced_opt(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_min_alphas: &DeviceBuffer<f32>,
        d_maj_alphas: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        host_lengths_opt: Option<&[i32]>,
        max_window_total: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSamaError> {
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto
            | BatchKernelPolicy::Plain { .. }
            | BatchKernelPolicy::Tiled { .. } => 256u32,
        };
        let block: BlockSize = (block_x, 1, 1).into();

        let func = self
            .module
            .get_function("sama_batch_f32")
            .map_err(|_| CudaSamaError::MissingKernelSymbol { name: "sama_batch_f32" })?;

        // Select kernel once for introspection/debugging
        unsafe {
            (*(self as *const _ as *mut CudaSama)).last_batch =
                Some(SamaBatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Chunk across combos; modern gridDim.x permits very large x
        const MAX_SLICE: usize = 2_147_483_647; // modern x-dim limit
        let mut start = 0usize;
        while start < n_combos {
            let len = (n_combos - start).min(MAX_SLICE);
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut lengths_ptr = d_lengths.as_device_ptr().add(start).as_raw();
                let mut min_ptr = d_min_alphas.as_device_ptr().add(start).as_raw();
                let mut maj_ptr = d_maj_alphas.as_device_ptr().add(start).as_raw();
                let mut first_ptr = d_first_valids.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                // Choose per-slice max_window when host lengths present, else total
                let slice_max_window: i32 = if let Some(host_lengths) = host_lengths_opt {
                    *host_lengths[start..start + len].iter().max().unwrap_or(&0)
                } else {
                    max_window_total
                };
                let mut max_window_i = slice_max_window;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let mut args: [*mut c_void; 9] = [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut lengths_ptr as *mut _ as *mut c_void,
                    &mut min_ptr as *mut _ as *mut c_void,
                    &mut maj_ptr as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut max_window_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let grid: GridSize = (len as u32, 1, 1).into();
                // Dynamic shared memory for two deques of indices
                let shmem_bytes: u32 =
                    (2 * ((slice_max_window as usize) + 1) * std::mem::size_of::<i32>()) as u32;
                self.stream
                    .launch(&func, grid, block, shmem_bytes, &mut args)
                    .map_err(CudaSamaError::from)?;
            }
            start += len;
        }

        self.stream.synchronize().map_err(Into::into)
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        length: i32,
        min_alpha: f32,
        maj_alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSamaError> {
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto
            | ManySeriesKernelPolicy::OneD { .. }
            | ManySeriesKernelPolicy::Tiled2D { .. } => 256u32,
        };
        let block: BlockSize = (block_x, 1, 1).into();
        let grid: GridSize = (num_series as u32, 1, 1).into();

        let func = self
            .module
            .get_function("sama_many_series_one_param_f32")
            .map_err(|_| CudaSamaError::MissingKernelSymbol { name: "sama_many_series_one_param_f32" })?;

        // Introspection for benches/debug
        unsafe {
            (*(self as *const _ as *mut CudaSama)).last_many =
                Some(SamaManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        // Dynamic shared memory for deques (cap = length + 1)
        let max_window = length.max(0);
        let shmem_bytes: u32 =
            (2 * ((max_window as usize) + 1) * std::mem::size_of::<i32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut length_i = length;
            let mut min_a = min_alpha;
            let mut maj_a = maj_alpha;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut max_window_i = max_window;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 9] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut length_i as *mut _ as *mut c_void,
                &mut min_a as *mut _ as *mut c_void,
                &mut maj_a as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut max_window_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shmem_bytes, &mut args)
                .map_err(CudaSamaError::from)?;
        }

        self.stream.synchronize().map_err(CudaSamaError::from)
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SamaBatchRange,
    ) -> Result<PreparedSamaBatch, CudaSamaError> {
        if data_f32.is_empty() {
            return Err(CudaSamaError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_grid(sweep)?;

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaSamaError::InvalidInput("all values are NaN".into()))?;
        if series_len - first_valid < 1 {
            return Err(CudaSamaError::InvalidInput(
                "not enough valid data to start computation".into(),
            ));
        }

        let mut lengths_i32 = Vec::with_capacity(combos.len());
        let mut min_alphas = Vec::with_capacity(combos.len());
        let mut maj_alphas = Vec::with_capacity(combos.len());
        let mut first_valids = Vec::with_capacity(combos.len());

        for params in &combos {
            let length = params.length.unwrap_or(200);
            let maj_length = params.maj_length.unwrap_or(14);
            let min_length = params.min_length.unwrap_or(6);

            if length == 0 || maj_length == 0 || min_length == 0 {
                return Err(CudaSamaError::InvalidInput(
                    "length, maj_length, and min_length must be positive".into(),
                ));
            }
            if length + 1 > series_len {
                return Err(CudaSamaError::InvalidInput(format!(
                    "length {} exceeds available data {}",
                    length + 1,
                    series_len
                )));
            }

            let min_alpha = 2.0f32 / (min_length as f32 + 1.0f32);
            let maj_alpha = 2.0f32 / (maj_length as f32 + 1.0f32);

            lengths_i32.push(length as i32);
            min_alphas.push(min_alpha);
            maj_alphas.push(maj_alpha);
            first_valids.push(first_valid as i32);
        }

        Ok(PreparedSamaBatch {
            combos,
            first_valid,
            series_len,
            lengths_i32,
            min_alphas,
            maj_alphas,
            first_valids,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SamaParams,
    ) -> Result<PreparedSamaManySeries, CudaSamaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaSamaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaSamaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }

        let length = params.length.unwrap_or(200) as i32;
        let maj_length = params.maj_length.unwrap_or(14);
        let min_length = params.min_length.unwrap_or(6);
        if length <= 0 || maj_length == 0 || min_length == 0 {
            return Err(CudaSamaError::InvalidInput(
                "length, maj_length, and min_length must be positive".into(),
            ));
        }
        if (length as usize) + 1 > series_len {
            return Err(CudaSamaError::InvalidInput(format!(
                "length {} exceeds available data {}",
                length as usize + 1,
                series_len
            )));
        }

        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + series];
                if v.is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaSamaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if series_len - (fv as usize) < 1 {
                return Err(CudaSamaError::InvalidInput(format!(
                    "series {} does not have enough valid data",
                    series
                )));
            }
            first_valids.push(fv);
        }

        let min_alpha = 2.0f32 / (min_length as f32 + 1.0f32);
        let maj_alpha = 2.0f32 / (maj_length as f32 + 1.0f32);

        Ok(PreparedSamaManySeries {
            first_valids,
            length,
            min_alpha,
            maj_alpha,
        })
    }
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
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
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
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SAMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSama)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] SAMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSama)).debug_many_logged = true;
                }
            }
        }
    }
}

// --- utility: async memset to fill i32 buffers on device ---
#[inline]
fn memset_i32_async(
    stream: &Stream,
    buf: &mut DeviceBuffer<i32>,
    value: i32,
) -> Result<(), CudaSamaError> {
    unsafe {
        let ptr: cu::CUdeviceptr = buf.as_device_ptr().as_raw();
        let n: usize = buf.len();
        let st: cu::CUstream = stream.as_inner();
        let res = cu::cuMemsetD32Async(ptr, value as u32, n, st);
        match res {
            cu::CUresult::CUDA_SUCCESS => Ok(()),
            e => Err(CudaSamaError::InvalidInput(format!("cuMemsetD32Async failed: {:?}", e))),
        }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        sama_benches,
        CudaSama,
        crate::indicators::moving_averages::sama::SamaBatchRange,
        crate::indicators::moving_averages::sama::SamaParams,
        sama_batch_dev,
        sama_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::sama::SamaBatchRange {
            length: (64, 64 + PARAM_SWEEP - 1, 1),
            maj_length: (14, 14, 0),
            min_length: (6, 6, 0)
        },
        crate::indicators::moving_averages::sama::SamaParams {
            length: Some(64),
            maj_length: Some(14),
            min_length: Some(6)
        },
        "sama",
        "sama"
    );
    pub use sama_benches::bench_profiles;
}

fn expand_grid(range: &SamaBatchRange) -> Result<Vec<SamaParams>, CudaSamaError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaSamaError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            while x <= end {
                v.push(x);
                x = x
                    .checked_add(step)
                    .ok_or_else(|| CudaSamaError::InvalidInput("range overflow".into()))?;
            }
            if v.is_empty() {
                return Err(CudaSamaError::InvalidInput(
                    format!("invalid range: start={} end={} step={}", start, end, step),
                ));
            }
            Ok(v)
        } else {
            // reversed bounds supported
            let mut v = Vec::new();
            let mut x = start;
            loop {
                v.push(x);
                if x <= end { break; }
                // saturating sub is fine; we stop when we cross end
                x = x.saturating_sub(step);
                if x <= end { break; }
                // guard infinite loops on step==0 handled above
            }
            if v.is_empty() {
                return Err(CudaSamaError::InvalidInput(
                    format!("invalid range: start={} end={} step={}", start, end, step),
                ));
            }
            Ok(v)
        }
    }

    let lengths = axis(range.length)?;
    let maj_lengths = axis(range.maj_length)?;
    let min_lengths = axis(range.min_length)?;

    if lengths.is_empty() || maj_lengths.is_empty() || min_lengths.is_empty() {
        return Err(CudaSamaError::InvalidInput("no parameter combinations provided".into()));
    }

    let cap = lengths
        .len()
        .checked_mul(maj_lengths.len())
        .and_then(|x| x.checked_mul(min_lengths.len()))
        .ok_or_else(|| CudaSamaError::InvalidInput("size overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &len in &lengths {
        for &maj in &maj_lengths {
            for &min in &min_lengths {
                out.push(SamaParams {
                    length: Some(len),
                    maj_length: Some(maj),
                    min_length: Some(min),
                });
            }
        }
    }
    Ok(out)
}
