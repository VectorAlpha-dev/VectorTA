//! CUDA wrapper for SMMA (Smoothed Moving Average) kernels.
//!
//! Mirrors the ALMA/SWMA scaffolding: validate host inputs, upload FP32 data
//! once, and launch kernels that keep the dependency-aware recursion on the
//! device. Supports both the single-series × many-parameter sweep and the
//! many-series × one-parameter time-major path.

#![cfg(feature = "cuda")]

use std::sync::Arc;
use crate::indicators::moving_averages::smma::{expand_grid, SmmaBatchRange, SmmaParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaSmmaError {
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
    #[error(
        "launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})"
    )]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer on {buf}, current {current}")]
    DeviceMismatch { buf: i32, current: i32 },
    #[error("not implemented")] 
    NotImplemented,
}

// -------- Kernel selection policy (mirrors ALMA/CWMA shape) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaSmmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaSmmaPolicy {
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
    OneD { block_x: u32 },
    WarpScan { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

/// VRAM-backed array for SMMA results with context lifetime and device id.
pub struct DeviceArrayF32Smma {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Smma {
    #[inline]
    pub fn device_ptr(&self) -> u64 { self.buf.as_device_ptr().as_raw() as u64 }
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }
}

pub struct CudaSmma {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaSmmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaSmma {
    pub fn new(device_id: usize) -> Result<Self, CudaSmmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/smma_kernel.ptx"));
        // Prefer context-targeted JIT with highest optimization, fallback progressively
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
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
            policy: CudaSmmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaSmmaError> {
        if let Ok((free, _total)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaSmmaError::OutOfMemory { required, free, headroom });
            }
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
                    eprintln!("[DEBUG] SMMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSmma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] SMMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaSmma)).debug_many_logged = true;
                }
            }
        }
    }

    pub fn set_policy(&mut self, policy: CudaSmmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaSmmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SmmaBatchRange,
    ) -> Result<(Vec<SmmaParams>, usize, usize), CudaSmmaError> {
        if data_f32.is_empty() {
            return Err(CudaSmmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaSmmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaSmmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let len = data_f32.len();
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaSmmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaSmmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaSmmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaSmmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaSmmaError::InvalidInput(
                "series_len or n_combos exceed i32::MAX".into(),
            ));
        }
        if first_valid > i32::MAX as usize {
            return Err(CudaSmmaError::InvalidInput(
                "first_valid exceeds i32::MAX".into(),
            ));
        }

        if matches!(self.policy.batch, BatchKernelPolicy::Auto) {
            if let Ok(mut func) = self.module.get_function("smma_batch_warp_scan_f32") {
                let _ = func.set_cache_config(CacheConfig::PreferL1);

                let block_x = 32u32;
                unsafe {
                    (*(self as *const _ as *mut CudaSmma)).last_batch =
                        Some(BatchKernelSelected::WarpScan { block_x });
                }

                let grid: GridSize = (n_combos as u32, 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();

                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                    let mut warms_ptr = d_warms.as_device_ptr().as_raw();
                    let mut first_valid_i = first_valid as i32;
                    let mut series_len_i = series_len as i32;
                    let mut n_combos_i = n_combos as i32;
                    let mut out_ptr = d_out.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut warms_ptr as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(&func, grid, block, 0, args)
                        .map_err(CudaSmmaError::Cuda)?;
                }

                self.maybe_log_batch_debug();
                return Ok(());
            }
        }

        let mut func = self
            .module
            .get_function("smma_batch_f32")
            .map_err(|_| CudaSmmaError::MissingKernelSymbol { name: "smma_batch_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 128,
        };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaSmma)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warms_ptr = d_warms.as_device_ptr().as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warms_ptr as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args).map_err(CudaSmmaError::Cuda)?;
        }

        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn smma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmmaError> {
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_warms,
            first_valid,
            series_len,
            n_combos,
            d_out,
        )
    }

    pub fn smma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SmmaBatchRange,
    ) -> Result<DeviceArrayF32Smma, CudaSmmaError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // Basic VRAM check (prices + periods + warms + output) with ~64MB headroom, overflow-safe
        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaSmmaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaSmmaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let warms_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaSmmaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let out_bytes = n_combos
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaSmmaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let required = prices_bytes
            .checked_add(periods_bytes)
            .and_then(|x| x.checked_add(warms_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaSmmaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        Self::will_fit(required, 64usize * 1024 * 1024)?;

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warms_i32 = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            if period > i32::MAX as usize {
                return Err(CudaSmmaError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            let warm = first_valid + period - 1;
            if warm > i32::MAX as usize {
                return Err(CudaSmmaError::InvalidInput(
                    "warm index exceeds i32::MAX".into(),
                ));
            }
            periods_i32.push(period as i32);
            warms_i32.push(warm as i32);
        }

        let d_prices = DeviceBuffer::from_slice(data_f32).map_err(CudaSmmaError::Cuda)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32).map_err(CudaSmmaError::Cuda)?;
        let d_warms = DeviceBuffer::from_slice(&warms_i32).map_err(CudaSmmaError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
            .map_err(CudaSmmaError::Cuda)?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_warms,
            first_valid,
            series_len,
            n_combos,
            &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32Smma {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmmaParams,
    ) -> Result<(Vec<i32>, usize), CudaSmmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSmmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaSmmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let period = params.period.unwrap_or(7);
        if period == 0 {
            return Err(CudaSmmaError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaSmmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaSmmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period {
                return Err(CudaSmmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaSmmaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmmaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaSmmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaSmmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let mut func = self
            .module
            .get_function("smma_multi_series_one_param_f32")
            .map_err(|_| CudaSmmaError::MissingKernelSymbol { name: "smma_multi_series_one_param_f32" })?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 128,
        };
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaSmma)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args).map_err(CudaSmmaError::Cuda)?;
        }

        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn smma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSmmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaSmmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn smma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmmaParams,
    ) -> Result<DeviceArrayF32Smma, CudaSmmaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // Basic VRAM check (input + first_valids + output) with ~64MB headroom (overflow-safe)
        let input_bytes = cols
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaSmmaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let fv_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaSmmaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let out_bytes = cols
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaSmmaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        let required = input_bytes
            .checked_add(fv_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaSmmaError::InvalidInput("size overflow in VRAM estimate".into()))?;
        Self::will_fit(required, 64usize * 1024 * 1024)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32).map_err(CudaSmmaError::Cuda)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).map_err(CudaSmmaError::Cuda)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(CudaSmmaError::Cuda)?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream.synchronize().map_err(CudaSmmaError::Cuda)?;

        Ok(DeviceArrayF32Smma {
            buf: d_out_tm,
            rows,
            cols,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn smma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSmmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaSmmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32).map_err(CudaSmmaError::Cuda)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).map_err(CudaSmmaError::Cuda)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(CudaSmmaError::Cuda)?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream.synchronize().map_err(CudaSmmaError::Cuda)?;

        d_out_tm.copy_to(out_tm).map_err(CudaSmmaError::Cuda)?;
        Ok(())
    }

    /// Like `smma_multi_series_one_param_time_major_into_host_f32`, but returns
    /// a page-locked host buffer to avoid pageable staging on D2H.
    pub fn smma_multi_series_one_param_time_major_into_locked(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SmmaParams,
    ) -> Result<LockedBuffer<f32>, CudaSmmaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32).map_err(CudaSmmaError::Cuda)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).map_err(CudaSmmaError::Cuda)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(CudaSmmaError::Cuda)?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        // Allocate pinned host buffer and perform async D2H
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(cols * rows) }
            .map_err(CudaSmmaError::Cuda)?;
        unsafe {
            d_out_tm
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaSmmaError::Cuda)?;
        }
        self.stream.synchronize().map_err(CudaSmmaError::Cuda)?;
        Ok(pinned)
    }

    /// Variant that accepts time-major prices in page-locked memory and returns
    /// a page-locked result, enabling true async paths end-to-end.
    pub fn smma_multi_series_one_param_time_major_from_locked_to_locked(
        &self,
        data_tm_locked: &LockedBuffer<f32>,
        cols: usize,
        rows: usize,
        params: &SmmaParams,
    ) -> Result<LockedBuffer<f32>, CudaSmmaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_locked.as_slice(), cols, rows, params)?;

        // H2D from pinned host memory (async)
        let d_prices_tm = unsafe {
            DeviceBuffer::from_slice_async(data_tm_locked.as_slice(), &self.stream)
        }
        .map_err(CudaSmmaError::Cuda)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).map_err(CudaSmmaError::Cuda)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(CudaSmmaError::Cuda)?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(cols * rows) }
            .map_err(CudaSmmaError::Cuda)?;
        unsafe {
            d_out_tm
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(CudaSmmaError::Cuda)?;
        }
        self.stream.synchronize().map_err(CudaSmmaError::Cuda)?;
        Ok(pinned)
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::smma::SmmaParams;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let periods_bytes = PARAM_SWEEP * std::mem::size_of::<i32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + periods_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + first_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct SmmaBatchDevState {
        cuda: CudaSmma,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_warms: DeviceBuffer<i32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for SmmaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_warms,
                    self.first_valid,
                    self.series_len,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("smma batch kernel");
            self.cuda.stream.synchronize().expect("smma sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaSmma::new(0).expect("cuda smma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = SmmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (combos, first_valid, series_len) =
            CudaSmma::prepare_batch_inputs(&price, &sweep).expect("smma prepare batch inputs");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let warms_i32: Vec<i32> = combos
            .iter()
            .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
            .collect();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("d_warms");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(SmmaBatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_warms,
            first_valid,
            series_len,
            n_combos,
            d_out,
        })
    }

    struct SmmaManyDevState {
        cuda: CudaSmma,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for SmmaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    self.period,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("smma many-series kernel");
            self.cuda.stream.synchronize().expect("smma sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaSmma::new(0).expect("cuda smma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = SmmaParams { period: Some(64) };
        let (first_valids, period) =
            CudaSmma::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("smma prepare many-series inputs");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(SmmaManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "smma",
                "one_series_many_params",
                "smma_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "smma",
                "many_series_one_param",
                "smma_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
