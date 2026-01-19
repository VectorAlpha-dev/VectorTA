#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::tema::{TemaBatchRange, TemaParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const WARP: u32 = 32;
#[inline]
fn env_warps_per_block() -> u32 {
    std::env::var("TEMA_WPB")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&v| (1..=32).contains(&v))
        .unwrap_or(4)
}

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
pub struct CudaTemaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaTemaPolicy {
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
}

#[derive(Debug)]
pub enum CudaTemaError {
    Cuda(CudaError),
    InvalidInput(String),
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    MissingKernelSymbol {
        name: &'static str,
    },
    InvalidPolicy(&'static str),
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    DeviceMismatch {
        buf: u32,
        current: u32,
    },
    NotImplemented,
}

impl fmt::Display for CudaTemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTemaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTemaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaTemaError::OutOfMemory {
                required,
                free,
                headroom,
            } => write!(
                f,
                "Out of memory: required={} bytes (free={}, headroom={})",
                required, free, headroom
            ),
            CudaTemaError::MissingKernelSymbol { name } => {
                write!(f, "Missing kernel symbol: {}", name)
            }
            CudaTemaError::InvalidPolicy(s) => write!(f, "Invalid policy: {}", s),
            CudaTemaError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            } => write!(
                f,
                "Launch config too large: grid=({}, {}, {}), block=({}, {}, {})",
                gx, gy, gz, bx, by, bz
            ),
            CudaTemaError::DeviceMismatch { buf, current } => write!(
                f,
                "Device mismatch: buffer device {} vs current {}",
                buf, current
            ),
            CudaTemaError::NotImplemented => write!(f, "Not implemented"),
        }
    }
}

impl std::error::Error for CudaTemaError {}

impl From<CudaError> for CudaTemaError {
    #[inline]
    fn from(e: CudaError) -> Self {
        CudaTemaError::Cuda(e)
    }
}

pub struct CudaTema {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaTemaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,

    warps_per_block: u32,
    max_grid_x: u32,
}

impl CudaTema {
    pub fn new(device_id: usize) -> Result<Self, CudaTemaError> {
        cust::init(CudaFlags::empty()).map_err(CudaTemaError::Cuda)?;

        let device = Device::get_device(device_id as u32).map_err(CudaTemaError::Cuda)?;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(CudaTemaError::Cuda)? as u32;
        let context = Arc::new(Context::new(device).map_err(CudaTemaError::Cuda)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/tema_kernel.ptx"));

        let jit_opts = &[ModuleJitOption::DetermineTargetFromContext];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(CudaTemaError::Cuda)?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(CudaTemaError::Cuda)?;

        let warps_per_block = env_warps_per_block();

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaTemaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            warps_per_block,
            max_grid_x,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaTemaError> {
        self.stream.synchronize().map_err(CudaTemaError::Cuda)
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaTemaPolicy,
    ) -> Result<Self, CudaTemaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaTemaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaTemaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    #[inline]
    pub fn context_guard(&self) -> Arc<Context> {
        self._context.clone()
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    #[inline]
    pub fn stream_handle(&self) -> usize {
        self.stream.as_inner() as usize
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
                    eprintln!("[DEBUG] TEMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTema)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] TEMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTema)).debug_many_logged = true;
                }
            }
        }
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
    fn grid_x_chunks(_n: usize) -> impl Iterator<Item = (usize, usize)> {
        std::iter::empty()
    }

    #[inline]
    fn chunk_items_by_grid_x(&self, items: usize) -> impl Iterator<Item = (usize, usize)> {
        let per_block = self.warps_per_block as usize;
        let max_items_per_launch = (self.max_grid_x as usize).saturating_mul(per_block).max(1);
        (0..items).step_by(max_items_per_launch).map(move |start| {
            let len = (items - start).min(max_items_per_launch);
            (start, len)
        })
    }

    #[inline]
    fn batch_launch_dims(&self, n_combos: usize) -> (GridSize, BlockSize, u32) {
        let wpb = self.warps_per_block.max(1);
        let block_x = wpb * WARP;
        let blocks_x = (((n_combos as u32) + wpb - 1) / wpb).max(1);
        ((blocks_x, 1, 1).into(), (block_x, 1, 1).into(), block_x)
    }

    #[inline]
    fn warp_launch_dims(&self, items: usize) -> (GridSize, BlockSize, u32) {
        let wpb = self.warps_per_block;
        let block_x = wpb * WARP;
        let warps_needed = ((items as u32) + WARP - 1) / WARP;
        let blocks_x = ((warps_needed + wpb - 1) / wpb).max(1);
        ((blocks_x, 1, 1).into(), (block_x, 1, 1).into(), block_x)
    }

    pub fn tema_batch_dev(
        &self,
        prices: &[f32],
        sweep: &TemaBatchRange,
    ) -> Result<DeviceArrayF32, CudaTemaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn tema_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &TemaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TemaParams>), CudaTemaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaTemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;
        unsafe { arr.buf.async_copy_to(out, &self.stream) }.map_err(CudaTemaError::Cuda)?;
        self.stream.synchronize().map_err(CudaTemaError::Cuda)?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    pub fn tema_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTemaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaTemaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaTemaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn tema_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTemaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaTemaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaTemaError::InvalidInput(
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

    pub fn tema_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaTemaError> {
        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)
    }

    pub fn tema_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTemaError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTemaError::InvalidInput("cols * rows overflow".into()))?;
        if out_tm.len() != expected {
            return Err(CudaTemaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                expected
            )));
        }

        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)?;
        unsafe { arr.buf.async_copy_to(out_tm, &self.stream) }.map_err(CudaTemaError::Cuda)?;
        self.stream.synchronize().map_err(CudaTemaError::Cuda)?;
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaTemaError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        let total_elems = series_len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaTemaError::InvalidInput("rows * cols overflow".into()))?;

        let sz_f32 = core::mem::size_of::<f32>();
        let sz_i32 = core::mem::size_of::<i32>();
        let prices_bytes = series_len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaTemaError::InvalidInput("byte size overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaTemaError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = total_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaTemaError::InvalidInput("byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(periods_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaTemaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;

        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaTemaError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(prices, &self.stream) }
            .map_err(CudaTemaError::Cuda)?;
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&inputs.periods, &self.stream) }
            .map_err(CudaTemaError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total_elems) }.map_err(CudaTemaError::Cuda)?;

        self.try_enable_persisting_l2(d_prices.as_device_ptr().as_raw(), prices_bytes);

        for (start, len) in self.chunk_items_by_grid_x(n_combos) {
            let periods_ptr_raw =
                d_periods.as_device_ptr().as_raw() + (start * core::mem::size_of::<i32>()) as u64;
            let out_ptr_raw = d_out.as_device_ptr().as_raw()
                + (start * series_len * core::mem::size_of::<f32>()) as u64;

            self.launch_batch_kernel_chunk(
                &d_prices,
                periods_ptr_raw,
                series_len,
                len,
                inputs.first_valid,
                out_ptr_raw,
            )?;
        }

        self.stream.synchronize().map_err(CudaTemaError::Cuda)?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        prepared: &ManySeriesInputs,
    ) -> Result<DeviceArrayF32, CudaTemaError> {
        let num_series = cols;
        let series_len = rows;

        let sz_f32 = core::mem::size_of::<f32>();
        let sz_i32 = core::mem::size_of::<i32>();
        let prices_bytes = prices_tm_f32
            .len()
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaTemaError::InvalidInput("byte size overflow".into()))?;
        let first_valid_bytes = prepared
            .first_valids
            .len()
            .checked_mul(sz_i32)
            .ok_or_else(|| CudaTemaError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = prices_tm_f32
            .len()
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaTemaError::InvalidInput("byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(first_valid_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaTemaError::InvalidInput("byte size overflow".into()))?;
        let headroom = 32 * 1024 * 1024;

        if !Self::will_fit(required, headroom) {
            let free = Self::device_mem_info().map(|(f, _)| f).unwrap_or(0);
            return Err(CudaTemaError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32).map_err(CudaTemaError::Cuda)?;
        let d_first_valids =
            DeviceBuffer::from_slice(&prepared.first_valids).map_err(CudaTemaError::Cuda)?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prices_tm_f32.len()) }
                .map_err(CudaTemaError::Cuda)?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            num_series,
            series_len,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        self.stream.synchronize().map_err(CudaTemaError::Cuda)?;

        self.stream.synchronize().map_err(CudaTemaError::Cuda)?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows: series_len,
            cols: num_series,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTemaError> {
        self.launch_batch_kernel_chunk(
            d_prices,
            d_periods.as_device_ptr().as_raw(),
            series_len,
            n_combos,
            first_valid,
            d_out.as_device_ptr().as_raw(),
        )
    }

    fn launch_batch_kernel_chunk(
        &self,
        d_prices: &DeviceBuffer<f32>,
        periods_ptr_raw: u64,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        out_ptr_raw: u64,
    ) -> Result<(), CudaTemaError> {
        let func = self.module.get_function("tema_batch_f32").map_err(|_| {
            CudaTemaError::MissingKernelSymbol {
                name: "tema_batch_f32",
            }
        })?;

        let (grid, block, block_x) = self.batch_launch_dims(n_combos);
        unsafe {
            (*(self as *const _ as *mut CudaTema)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = periods_ptr_raw;
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = out_ptr_raw;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaTemaError::Cuda)?
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
    ) -> Result<(), CudaTemaError> {
        let func = self
            .module
            .get_function("tema_multi_series_one_param_f32")
            .map_err(|_| CudaTemaError::MissingKernelSymbol {
                name: "tema_multi_series_one_param_f32",
            })?;

        let (grid, block, block_x) = self.warp_launch_dims(num_series);
        unsafe {
            (*(self as *const _ as *mut CudaTema)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

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
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(CudaTemaError::Cuda)?
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &TemaBatchRange,
    ) -> Result<BatchInputs, CudaTemaError> {
        if prices.is_empty() {
            return Err(CudaTemaError::InvalidInput("empty prices".into()));
        }

        let combos = expand_grid_tema(sweep)?;
        if combos.is_empty() {
            return Err(CudaTemaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTemaError::InvalidInput("all values are NaN".into()))?;

        let series_len = prices.len();
        let mut periods = Vec::with_capacity(combos.len());
        let mut max_period = 0usize;
        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaTemaError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            if period > i32::MAX as usize {
                return Err(CudaTemaError::InvalidInput(
                    "period exceeds i32 kernel limit".into(),
                ));
            }
            periods.push(period as i32);
            max_period = max_period.max(period);
        }

        if series_len - first_valid < max_period {
            return Err(CudaTemaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }

        Ok(BatchInputs {
            combos,
            periods,
            first_valid,
            series_len,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<ManySeriesInputs, CudaTemaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTemaError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTemaError::InvalidInput("cols * rows overflow".into()))?;
        if prices_tm_f32.len() != expected {
            return Err(CudaTemaError::InvalidInput("matrix shape mismatch".into()));
        }
        if period == 0 {
            return Err(CudaTemaError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        if period > i32::MAX as usize {
            return Err(CudaTemaError::InvalidInput(
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
                CudaTemaError::InvalidInput(format!("series {} has all NaN values", series_idx))
            })?;
            if rows - first < period {
                return Err(CudaTemaError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series_idx,
                    period,
                    rows - first
                )));
            }
            first_valids[series_idx] = first as i32;
        }

        Ok(ManySeriesInputs { first_valids })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::tema::TemaBatchRange;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct TemaBatchDevState {
        cuda: CudaTema,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TemaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .tema_batch_device(
                    &self.d_prices,
                    &self.d_periods,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("tema batch kernel");
            self.cuda.stream.synchronize().expect("tema sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaTema::new(0).expect("cuda tema");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = TemaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };

        let inputs = CudaTema::prepare_batch_inputs(&price, &sweep).expect("tema prepare batch");
        let n_combos = inputs.combos.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&inputs.periods).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(inputs.series_len * n_combos) }.expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(TemaBatchDevState {
            cuda,
            d_prices,
            d_periods,
            series_len: inputs.series_len,
            n_combos,
            first_valid: inputs.first_valid,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "tema",
            "one_series_many_params",
            "tema_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}

impl CudaTema {
    fn try_enable_persisting_l2(&self, base_dev_ptr: u64, bytes: usize) {
        unsafe {
            use cust::device::Device as CuDevice;
            use cust::sys::{
                cuCtxSetLimit, cuDeviceGetAttribute, cuStreamSetAttribute,
                CUaccessPolicyWindow_v1 as CUaccessPolicyWindow,
                CUaccessProperty_enum as AccessProp, CUdevice_attribute_enum as DevAttr,
                CUlimit_enum as CULimit, CUstreamAttrID_enum as StreamAttrId,
                CUstreamAttrValue_v1 as CUstreamAttrValue,
            };

            let mut max_win_i32: i32 = 0;
            if let Ok(dev) = CuDevice::get_device(self.device_id) {
                let _ = cuDeviceGetAttribute(
                    &mut max_win_i32 as *mut _,
                    DevAttr::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                    dev.as_raw(),
                );
            }
            let max_bytes = (max_win_i32.max(0) as usize).min(bytes);
            if max_bytes == 0 {
                return;
            }

            let _ = cuCtxSetLimit(CULimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, max_bytes);

            let mut val: CUstreamAttrValue = std::mem::zeroed();
            val.accessPolicyWindow = CUaccessPolicyWindow {
                base_ptr: base_dev_ptr as *mut std::ffi::c_void,
                num_bytes: max_bytes,
                hitRatio: 0.9f32,
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
}

struct BatchInputs {
    combos: Vec<TemaParams>,
    periods: Vec<i32>,
    first_valid: usize,
    series_len: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
}

fn expand_grid_tema(range: &TemaBatchRange) -> Result<Vec<TemaParams>, CudaTemaError> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return Ok(vec![TemaParams {
            period: Some(start),
        }]);
    }
    let out: Vec<usize> = if start <= end {
        (start..=end).step_by(step).collect()
    } else {
        let mut v = Vec::new();
        let mut cur = start;
        while cur >= end {
            v.push(cur);
            if let Some(next) = cur.checked_sub(step) {
                cur = next;
            } else {
                break;
            }
            if cur < end {
                break;
            }
        }
        v
    };
    if out.is_empty() {
        return Err(CudaTemaError::InvalidInput("invalid period range".into()));
    }
    Ok(out
        .into_iter()
        .map(|p| TemaParams { period: Some(p) })
        .collect())
}
