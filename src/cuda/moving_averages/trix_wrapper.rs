//! CUDA scaffolding for the TRIX (Triple Exponential Average Oscillator) kernels.
//!
//! ALMA-parity wrapper surface:
//! - PTX loaded with DetermineTargetFromContext and OptLevel O2 (with fallbacks)
//! - NON_BLOCKING stream, VRAM estimates via mem_get_info, ~64MB headroom
//! - Policy enums and one-shot BENCH_DEBUG logging of selected kernels
//! - Batch (one series × many params) and time-major many-series × one param
//! - Chunk extremely large grids when needed (grid.x based)
//!
//! Math category: recurrence/IIR. The batch kernel expects ln(price) to be
//! precomputed on host and passed as input to avoid redundant device logf
//! across parameter rows. The many-series kernel computes ln(price) on device
//! per series.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use super::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::indicators::trix::{TrixBatchRange, TrixParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

#[inline]
const fn qnan_u32() -> u32 {
    0x7fc0_0000
}

#[derive(thiserror::Error, Debug)]
pub enum CudaTrixError {
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

#[derive(Clone, Copy, Debug)]
pub struct CudaTrixPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaTrixPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    /// One block per combo; single-thread sequential scan
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    /// One block per series; single-thread sequential scan (time-major)
    OneD { block_x: u32 },
}

pub struct CudaTrix {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaTrixPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    max_grid_x: u32,
}

impl CudaTrix {
    pub fn new(device_id: usize) -> Result<Self, CudaTrixError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let max_grid_x = device.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/trix_kernel.ptx"));
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
            policy: CudaTrixPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            max_grid_x,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaTrixPolicy,
    ) -> Result<Self, CudaTrixError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaTrixPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaTrixPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaTrixError> {
        self.stream
            .synchronize()
            .map_err(Into::into)
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

    pub fn trix_batch_dev(
        &self,
        prices: &[f32],
        sweep: &TrixBatchRange,
    ) -> Result<DeviceArrayF32, CudaTrixError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(&inputs)
    }

    pub fn trix_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &TrixBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TrixParams>), CudaTrixError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs
            .series_len
            .checked_mul(inputs.combos.len())
            .ok_or_else(|| CudaTrixError::InvalidInput("rows*cols overflow".into()))?;
        if out.len() != expected {
            return Err(CudaTrixError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(&inputs)?;
        unsafe { arr.buf.async_copy_to(out, &self.stream) }?;
        self.synchronize()?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    pub fn trix_batch_device(
        &self,
        d_logs: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrixError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaTrixError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaTrixError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        self.launch_batch_kernel(d_logs, d_periods, series_len, n_combos, first_valid, d_out)
    }

    pub fn trix_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrixError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaTrixError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaTrixError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            period,
            num_series,
            series_len,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn trix_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaTrixError> {
        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)
    }

    pub fn trix_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTrixError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTrixError::InvalidInput("rows*cols overflow".into()))?;
        if out_tm.len() != expected {
            return Err(CudaTrixError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                expected
            )));
        }
        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)?;
        unsafe { arr.buf.async_copy_to(out_tm, &self.stream) }?;
        self.synchronize()?;
        Ok(())
    }

    fn run_batch_kernel(&self, inputs: &BatchInputs) -> Result<DeviceArrayF32, CudaTrixError> {
        // VRAM budget: logs + periods + outputs
        let logs_bytes = inputs
            .series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaTrixError::InvalidInput("logs byte size overflow".into()))?;
        let periods_bytes = inputs
            .periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaTrixError::InvalidInput("periods byte size overflow".into()))?;
        let out_elems = inputs
            .series_len
            .checked_mul(inputs.combos.len())
            .ok_or_else(|| CudaTrixError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaTrixError::InvalidInput("output byte size overflow".into()))?;
        let bytes = logs_bytes
            .checked_add(periods_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaTrixError::InvalidInput("VRAM requirement overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaTrixError::OutOfMemory {
                    required: bytes,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaTrixError::InvalidInput(
                    "insufficient device memory for TRIX batch".into(),
                ));
            }
        }

        // Host precompute: ln(prices) already prepared in inputs.logs
        let d_logs = self.htod_copy_f32(&inputs.logs)?;
        let d_periods = self.htod_copy_i32(&inputs.periods)?;
        let out_len = out_elems;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_len) }?;
        memset_f32_qnan_async(&self.stream, &mut d_out)
            .map_err(|e| CudaTrixError::InvalidInput(e))?;

        self.launch_batch_kernel(
            &d_logs,
            &d_periods,
            inputs.series_len,
            inputs.combos.len(),
            inputs.first_valid,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: inputs.combos.len(),
            cols: inputs.series_len,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_logs: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrixError> {
        let func = self
            .module
            .get_function("trix_batch_f32")
            .map_err(|_| CudaTrixError::MissingKernelSymbol { name: "trix_batch_f32" })?;
        // Chunk grid.x to device limit
        let mut launched = 0usize;
        while launched < n_combos {
            let chunk = (n_combos - launched).min(self.max_grid_x as usize);
            let grid: GridSize = (chunk as u32, 1, 1).into();
            let block: BlockSize = (1, 1, 1).into();
            unsafe {
                (*(self as *const _ as *mut CudaTrix)).last_batch =
                    Some(BatchKernelSelected::Plain { block_x: 1 });
            }
            self.maybe_log_batch_debug();

            unsafe {
                let mut logs_ptr = d_logs.as_device_ptr().as_raw() + 0u64; // logs always start at 0
                let period_offset_bytes = launched
                    .checked_mul(core::mem::size_of::<i32>())
                    .ok_or_else(|| CudaTrixError::InvalidInput("periods offset overflow".into()))?;
                let mut periods_ptr =
                    d_periods.as_device_ptr().as_raw() + period_offset_bytes as u64;
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = chunk as i32;
                let mut first_valid_i = first_valid as i32;
                let offset_elems = launched
                    .checked_mul(series_len)
                    .ok_or_else(|| CudaTrixError::InvalidInput("output offset overflow".into()))?;
                let offset_bytes = offset_elems
                    .checked_mul(core::mem::size_of::<f32>())
                    .ok_or_else(|| CudaTrixError::InvalidInput("output offset bytes overflow".into()))?;
                let mut out_ptr =
                    d_out.as_device_ptr().as_raw() + offset_bytes as u64;
                let args: &mut [*mut c_void] = &mut [
                    &mut logs_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
            launched += chunk;
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrixError> {
        let func = self
            .module
            .get_function("trix_many_series_one_param_f32")
            .map_err(|_| CudaTrixError::MissingKernelSymbol {
                name: "trix_many_series_one_param_f32",
            })?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaTrix)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x: 1 });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut fvs_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut fvs_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        inputs: &ManySeriesInputs,
    ) -> Result<DeviceArrayF32, CudaTrixError> {
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTrixError::InvalidInput("rows*cols overflow".into()))?;
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaTrixError::InvalidInput("prices byte size overflow".into()))?;
        let fvs_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaTrixError::InvalidInput("first_valids byte size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaTrixError::InvalidInput("output byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(fvs_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaTrixError::InvalidInput("VRAM requirement overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaTrixError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaTrixError::InvalidInput(
                    "insufficient device memory for TRIX many-series launch".into(),
                ));
            }
        }
        let d_prices_tm =
            unsafe { DeviceBuffer::from_slice_async(prices_tm_f32, &self.stream) }?;
        let d_first_valids =
            unsafe { DeviceBuffer::from_slice_async(&inputs.first_valids, &self.stream) }?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }?;
        memset_f32_qnan_async(&self.stream, &mut d_out_tm)
            .map_err(|e| CudaTrixError::InvalidInput(e))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &TrixBatchRange,
    ) -> Result<BatchInputs, CudaTrixError> {
        if prices.is_empty() {
            return Err(CudaTrixError::InvalidInput("empty prices".into()));
        }
        let combos = expand_grid_trix(sweep)?;
        if combos.is_empty() {
            return Err(CudaTrixError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // First valid index
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTrixError::InvalidInput("all values are NaN".into()))?;

        let series_len = prices.len();
        let mut periods = Vec::with_capacity(combos.len());
        let mut max_period = 0usize;
        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaTrixError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            if period > i32::MAX as usize {
                return Err(CudaTrixError::InvalidInput(
                    "period exceeds i32 kernel limit".into(),
                ));
            }
            periods.push(period as i32);
            max_period = max_period.max(period);
        }

        // Validate sufficient data for the largest period
        let needed = max_period
            .checked_sub(1)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| CudaTrixError::InvalidInput("period overflow when computing TRIX warmup length".into()))?;
        if series_len - first_valid < needed {
            return Err(CudaTrixError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                needed,
                series_len - first_valid
            )));
        }

        // Shared host precompute: ln(price)
        let mut logs = vec![0f32; series_len];
        for i in 0..first_valid {
            logs[i] = 0.0;
        }
        for i in first_valid..series_len {
            logs[i] = prices[i].ln();
        }

        Ok(BatchInputs {
            combos,
            periods,
            first_valid,
            series_len,
            logs,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<ManySeriesInputs, CudaTrixError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTrixError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTrixError::InvalidInput("rows*cols overflow".into()))?;
        if prices_tm_f32.len() != elems {
            return Err(CudaTrixError::InvalidInput("matrix shape mismatch".into()));
        }
        if period == 0 {
            return Err(CudaTrixError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        if period > i32::MAX as usize {
            return Err(CudaTrixError::InvalidInput(
                "period exceeds i32 kernel limit".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series_idx in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series_idx;
                let price = prices_tm_f32[idx];
                if !price.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let first = fv.ok_or_else(|| {
                CudaTrixError::InvalidInput(format!("series {} has all NaN values", series_idx))
            })?;
            let needed = period
                .checked_sub(1)
                .and_then(|v| v.checked_mul(3))
                .and_then(|v| v.checked_add(2))
                .ok_or_else(|| CudaTrixError::InvalidInput("period overflow when computing TRIX warmup length".into()))?;
            if rows - first < needed {
                return Err(CudaTrixError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series_idx,
                    needed,
                    rows - first
                )));
            }
            first_valids[series_idx] = first as i32;
        }
        Ok(ManySeriesInputs { first_valids })
    }

    #[inline]
    fn htod_copy_f32(&self, src: &[f32]) -> Result<DeviceBuffer<f32>, CudaTrixError> {
        match LockedBuffer::from_slice(src) {
            Ok(h_pinned) => unsafe {
                let mut dst =
                    DeviceBuffer::uninitialized_async(src.len(), &self.stream)?;
                dst.async_copy_from(&h_pinned, &self.stream)?;
                Ok(dst)
            },
            Err(_) => DeviceBuffer::from_slice(src).map_err(Into::into),
        }
    }
    #[inline]
    fn htod_copy_i32(&self, src: &[i32]) -> Result<DeviceBuffer<i32>, CudaTrixError> {
        match LockedBuffer::from_slice(src) {
            Ok(h_pinned) => unsafe {
                let mut dst =
                    DeviceBuffer::uninitialized_async(src.len(), &self.stream)?;
                dst.async_copy_from(&h_pinned, &self.stream)?;
                Ok(dst)
            },
            Err(_) => DeviceBuffer::from_slice(src).map_err(Into::into),
        }
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
                    eprintln!("[DEBUG] TRIX batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTrix)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] TRIX many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTrix)).debug_many_logged = true;
                }
            }
        }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches_batch_only;

    define_ma_period_benches_batch_only!(
        trix_benches,
        CudaTrix,
        crate::indicators::trix::TrixBatchRange,
        trix_batch_dev,
        crate::indicators::trix::TrixBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        "trix",
        "trix"
    );
    pub use trix_benches::bench_profiles;
}

fn expand_grid_trix(range: &TrixBatchRange) -> Result<Vec<TrixParams>, CudaTrixError> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return Ok(vec![TrixParams {
            period: Some(start),
        }]);
    }
    let mut vals = Vec::new();
    if start < end {
        let mut v = start;
        loop {
            vals.push(v);
            if v >= end {
                break;
            }
            let next = match v.checked_add(step) {
                Some(n) => n,
                None => break,
            };
            if next == v {
                break;
            }
            v = next;
        }
    } else {
        let mut v = start;
        loop {
            vals.push(v);
            if v <= end {
                break;
            }
            let next = v.saturating_sub(step);
            if next == v {
                break;
            }
            v = next;
        }
    }
    if vals.is_empty() {
        return Err(CudaTrixError::InvalidInput(format!(
            "invalid range: start={} end={} step={}",
            start, end, step
        )));
    }
    let out = vals
        .into_iter()
        .map(|p| TrixParams { period: Some(p) })
        .collect();
    Ok(out)
}

struct BatchInputs {
    combos: Vec<TrixParams>,
    periods: Vec<i32>,
    first_valid: usize,
    series_len: usize,
    logs: Vec<f32>,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
}

// --- utility: async memset to canonical quiet-NaN (0x7FC0_0000) ---
#[inline]
fn memset_f32_qnan_async(stream: &Stream, buf: &mut DeviceBuffer<f32>) -> Result<(), String> {
    unsafe {
        let ptr: cust::sys::CUdeviceptr = buf.as_device_ptr().as_raw();
        let n: usize = buf.len();
        let st: cust::sys::CUstream = stream.as_inner();
        match cust::sys::cuMemsetD32Async(ptr, qnan_u32(), n, st) {
            cust::sys::CUresult::CUDA_SUCCESS => Ok(()),
            e => Err(format!("cuMemsetD32Async failed: {:?}", e)),
        }
    }
}
