#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::hma::{HmaBatchRange, HmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::device::DeviceAttribute;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

use super::cwma_wrapper::{BatchKernelPolicy, BatchThreadsPerOutput, ManySeriesKernelPolicy};

#[derive(Debug, Error)]
pub enum CudaHmaError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Out of memory on device: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("arithmetic overflow when computing {what}")]
    ArithmeticOverflow { what: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("device mismatch: buf on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
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

pub struct CudaHma {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    policy: CudaHmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaHmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaHmaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

impl CudaHma {
    #[inline]
    fn ring_in_shared() -> bool {
        true
    }
    #[inline]
    fn assume_out_prefilled() -> bool {
        false
    }
    pub fn new(device_id: usize) -> Result<Self, CudaHmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/hma_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
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
            ctx: Arc::new(context),
            device_id: device_id as u32,
            policy: CudaHmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaHmaPolicy) -> Result<Self, CudaHmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaHmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaHmaPolicy {
        &self.policy
    }
    #[inline]
    pub fn ctx(&self) -> Arc<Context> {
        Arc::clone(&self.ctx)
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaHmaError> {
        self.stream.synchronize().map_err(CudaHmaError::from)
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
                    eprintln!("[DEBUG] HMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] HMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHma)).debug_many_logged = true;
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
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaHmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        let (free, _total) = match Self::device_mem_info() {
            Some(v) => v,
            None => return Ok(()),
        };
        let need =
            required_bytes
                .checked_add(headroom_bytes)
                .ok_or(CudaHmaError::ArithmeticOverflow {
                    what: "required_bytes + headroom_bytes",
                })?;
        if need <= free {
            Ok(())
        } else {
            Err(CudaHmaError::OutOfMemory {
                required: required_bytes,
                free,
                headroom: headroom_bytes,
            })
        }
    }

    pub fn stream_handle_u64(&self) -> u64 {
        self.stream.as_inner() as u64
    }

    #[inline]
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_Y: usize = 65_535;
        (0..n).step_by(MAX_GRID_Y).map(move |start| {
            let len = (n - start).min(MAX_GRID_Y);
            (start, len)
        })
    }

    fn expand_range(range: &HmaBatchRange) -> Vec<HmaParams> {
        let (start, end, step) = range.period;

        if step == 0 || start == end {
            return vec![HmaParams {
                period: Some(start),
            }];
        }
        let s = step.max(1);

        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        let mut out = Vec::new();
        let mut x = lo;
        while x <= hi {
            out.push(HmaParams { period: Some(x) });
            match x.checked_add(s) {
                Some(nx) => x = nx,
                None => break,
            }
        }
        out
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &HmaBatchRange,
    ) -> Result<(Vec<HmaParams>, usize, usize, usize), CudaHmaError> {
        if data_f32.is_empty() {
            return Err(CudaHmaError::InvalidInput("empty data".into()));
        }

        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaHmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaHmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let tail_len = len - first_valid;
        let mut max_sqrt_len = 0usize;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaHmaError::InvalidInput(
                    "period must be at least 2".into(),
                ));
            }
            if period > len {
                return Err(CudaHmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            let half = period / 2;
            if half == 0 {
                return Err(CudaHmaError::InvalidInput(format!(
                    "period {} results in zero half-window",
                    period
                )));
            }
            let sqrt_len = ((period as f64).sqrt().floor() as usize).max(1);
            if tail_len < period + sqrt_len - 1 {
                return Err(CudaHmaError::InvalidInput(format!(
                    "not enough valid data for period {} (tail = {}, need >= {})",
                    period,
                    tail_len,
                    period + sqrt_len - 1
                )));
            }
            max_sqrt_len = max_sqrt_len.max(sqrt_len);
        }

        Ok((combos, first_valid, len, max_sqrt_len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods_ptr: u64,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_sqrt_len: usize,
        d_ring_ptr: u64,
        d_out_ptr: u64,
        block_x: u32,
        shared_bytes: usize,
    ) -> Result<(), CudaHmaError> {
        let func = self.module.get_function("hma_batch_f32").map_err(|_| {
            CudaHmaError::MissingKernelSymbol {
                name: "hma_batch_f32",
            }
        })?;

        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        {
            let dev = Device::get_device(self.device_id)?;
            let max_grid_x = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
            let max_tpb = dev.get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32;
            if grid_x == 0 || grid_x > max_grid_x || block_x == 0 || block_x > max_tpb {
                return Err(CudaHmaError::LaunchConfigTooLarge {
                    gx: grid_x,
                    gy: 1,
                    gz: 1,
                    bx: block_x,
                    by: 1,
                    bz: 1,
                });
            }
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods_ptr;
            let mut len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut max_sqrt_i = max_sqrt_len as i32;
            let mut ring_ptr = d_ring_ptr;
            let mut out_ptr = d_out_ptr;

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut max_sqrt_i as *mut _ as *mut c_void,
                &mut ring_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[HmaParams],
        first_valid: usize,
        len: usize,
        max_sqrt_len: usize,
    ) -> Result<DeviceArrayF32, CudaHmaError> {
        let n = combos.len();
        let sz_f32 = std::mem::size_of::<f32>();
        let prices_bytes = len
            .checked_mul(sz_f32)
            .ok_or(CudaHmaError::ArithmeticOverflow {
                what: "len * sizeof(f32)",
            })?;
        let periods_bytes =
            n.checked_mul(std::mem::size_of::<i32>())
                .ok_or(CudaHmaError::ArithmeticOverflow {
                    what: "n * sizeof(i32)",
                })?;
        let ring_elems = n
            .checked_mul(max_sqrt_len)
            .ok_or(CudaHmaError::ArithmeticOverflow {
                what: "n * max_sqrt_len",
            })?;
        let ring_bytes =
            ring_elems
                .checked_mul(sz_f32)
                .ok_or(CudaHmaError::ArithmeticOverflow {
                    what: "ring_elems * sizeof(f32)",
                })?;
        let out_elems = n
            .checked_mul(len)
            .ok_or(CudaHmaError::ArithmeticOverflow { what: "n * len" })?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or(CudaHmaError::ArithmeticOverflow {
                what: "out_elems * sizeof(f32)",
            })?;
        let required = prices_bytes
            .checked_add(periods_bytes)
            .ok_or(CudaHmaError::ArithmeticOverflow {
                what: "prices+periods",
            })?
            .checked_add(ring_bytes)
            .ok_or(CudaHmaError::ArithmeticOverflow { what: "prev+ring" })?
            .checked_add(out_bytes)
            .ok_or(CudaHmaError::ArithmeticOverflow { what: "prev+out" })?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit_checked(required, headroom)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods, &self.stream) }?;

        let elems = n * len;
        let mut d_ring: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(ring_elems, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => match std::env::var("HMA_BLOCK_X")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
            {
                Some(v) if v > 0 => v,
                _ => 1,
            },
        };
        unsafe {
            let this = self as *const _ as *mut CudaHma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        let periods_ptr = unsafe { d_periods.as_device_ptr().as_raw() };
        let ring_ptr = unsafe { d_ring.as_device_ptr().as_raw() };
        let out_ptr = unsafe { d_out.as_device_ptr().as_raw() };
        let shared_bytes: usize = if Self::ring_in_shared() {
            max_sqrt_len * (block_x as usize) * std::mem::size_of::<f32>()
        } else {
            0
        };

        self.launch_batch_kernel(
            &d_prices,
            periods_ptr,
            len,
            n,
            first_valid,
            max_sqrt_len,
            ring_ptr,
            out_ptr,
            block_x,
            shared_bytes,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n,
            cols: len,
        })
    }

    pub fn hma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &HmaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<HmaParams>), CudaHmaError> {
        let (combos, first_valid, len, max_sqrt_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len, max_sqrt_len)?;
        Ok((dev, combos))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_sqrt_len: usize,
        d_ring: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaHmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaHmaError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if d_prices.len() != series_len {
            return Err(CudaHmaError::InvalidInput(
                "prices buffer length mismatch".into(),
            ));
        }
        if d_periods.len() < n_combos {
            return Err(CudaHmaError::InvalidInput(
                "periods buffer length mismatch".into(),
            ));
        }

        let ring_elems = n_combos
            .checked_mul(max_sqrt_len)
            .ok_or(CudaHmaError::ArithmeticOverflow {
                what: "n_combos * max_sqrt_len",
            })?;
        if d_ring.len() < ring_elems {
            return Err(CudaHmaError::InvalidInput(format!(
                "ring buffer too small: got {}, need {}",
                d_ring.len(),
                ring_elems
            )));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaHmaError::InvalidInput(format!(
                "output buffer wrong length: got {}, expected {}",
                d_out.len(),
                n_combos * series_len
            )));
        }

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => match std::env::var("HMA_BLOCK_X")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
            {
                Some(v) if v > 0 => v,
                _ => 1,
            },
        };
        unsafe {
            let this = self as *const _ as *mut CudaHma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        let periods_ptr = unsafe { d_periods.as_device_ptr().as_raw() };
        let ring_ptr = unsafe { d_ring.as_device_ptr().as_raw() };
        let out_ptr = unsafe { d_out.as_device_ptr().as_raw() };
        let shared_bytes: usize = if Self::ring_in_shared() {
            max_sqrt_len
                .checked_mul(block_x as usize)
                .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
                .ok_or(CudaHmaError::ArithmeticOverflow {
                    what: "shared_bytes",
                })?
        } else {
            0
        };

        self.launch_batch_kernel(
            d_prices,
            periods_ptr,
            series_len,
            n_combos,
            first_valid,
            max_sqrt_len,
            ring_ptr,
            out_ptr,
            block_x,
            shared_bytes,
        )
    }

    pub fn hma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &HmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<HmaParams>), CudaHmaError> {
        let (combos, first_valid, len, max_sqrt_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaHmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len, max_sqrt_len)?;

        let n_elems = out.len();
        if n_elems >= (1 << 20) {
            let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(n_elems)? };
            unsafe {
                dev.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
            }
            self.stream.synchronize()?;
            out.copy_from_slice(pinned.as_slice());
        } else {
            unsafe {
                dev.buf.async_copy_to(out, &self.stream)?;
            }
            self.stream.synchronize()?;
        }
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HmaParams,
    ) -> Result<(Vec<i32>, usize, usize), CudaHmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaHmaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or(CudaHmaError::ArithmeticOverflow {
                what: "cols * rows",
            })?;
        if data_tm_f32.len() != expected {
            return Err(CudaHmaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                expected,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        if period < 2 {
            return Err(CudaHmaError::InvalidInput(
                "period must be at least 2".into(),
            ));
        }
        if period > rows {
            return Err(CudaHmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        let half = period / 2;
        if half == 0 {
            return Err(CudaHmaError::InvalidInput(format!(
                "period {} results in zero half-window",
                period
            )));
        }
        let sqrt_len = ((period as f64).sqrt().floor() as usize).max(1);

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaHmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period + sqrt_len - 1 {
                return Err(CudaHmaError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {}, need >= {})",
                    series,
                    period,
                    rows - fv,
                    period + sqrt_len - 1
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period, sqrt_len))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
        period: usize,
        max_sqrt_len: usize,
        d_ring_ptr: u64,
        d_out_ptr: u64,
        block_x: u32,
        shared_bytes: usize,
    ) -> Result<(), CudaHmaError> {
        let func = self
            .module
            .get_function("hma_many_series_one_param_f32")
            .map_err(|_| CudaHmaError::MissingKernelSymbol {
                name: "hma_many_series_one_param_f32",
            })?;

        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut period_i = period as i32;
            let mut max_sqrt_i = max_sqrt_len as i32;
            let mut ring_ptr = d_ring_ptr;
            let mut out_ptr = d_out_ptr;

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut max_sqrt_i as *mut _ as *mut c_void,
                &mut ring_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)?;
        }

        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
        sqrt_len: usize,
    ) -> Result<DeviceArrayF32, CudaHmaError> {
        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(first_valids, &self.stream) }?;

        let elems = cols * rows;
        let ring_elems = cols
            .checked_mul(sqrt_len)
            .ok_or(CudaHmaError::ArithmeticOverflow {
                what: "cols * sqrt_len",
            })?;
        let mut d_ring: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(ring_elems, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => match std::env::var("HMA_MS_BLOCK_X")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
            {
                Some(v) if v == 128 || v == 256 || v == 512 => v,
                _ => 256,
            },
        };
        unsafe {
            let this = self as *const _ as *mut CudaHma;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let ring_ptr = unsafe { d_ring.as_device_ptr().as_raw() };
        let out_ptr = unsafe { d_out.as_device_ptr().as_raw() };
        let shared_bytes: usize = if Self::ring_in_shared() {
            sqrt_len * (block_x as usize) * std::mem::size_of::<f32>()
        } else {
            0
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            cols,
            rows,
            period,
            sqrt_len,
            ring_ptr,
            out_ptr,
            block_x,
            shared_bytes,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn hma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HmaParams,
    ) -> Result<DeviceArrayF32, CudaHmaError> {
        let (first_valids, period, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period, sqrt_len)
    }

    pub fn hma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaHmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaHmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev =
            self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period, sqrt_len)?;

        let n_elems = out_tm.len();
        if n_elems >= (1 << 20) {
            let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(n_elems)? };
            unsafe {
                dev.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
            }
            self.stream.synchronize()?;
            out_tm.copy_from_slice(pinned.as_slice());
        } else {
            unsafe {
                dev.buf.async_copy_to(out_tm, &self.stream)?;
            }
            self.stream.synchronize()?;
        }
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::hma::{HmaBatchRange, HmaParams};

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

    struct HmaBatchDevState {
        cuda: CudaHma,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_sqrt_len: usize,
        d_ring: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        block_x: u32,
        shared_bytes: usize,
    }
    impl CudaBenchState for HmaBatchDevState {
        fn launch(&mut self) {
            let periods_ptr = unsafe { self.d_periods.as_device_ptr().as_raw() };
            let ring_ptr = unsafe { self.d_ring.as_device_ptr().as_raw() };
            let out_ptr = unsafe { self.d_out.as_device_ptr().as_raw() };
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    periods_ptr,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    self.max_sqrt_len,
                    ring_ptr,
                    out_ptr,
                    self.block_x,
                    self.shared_bytes,
                )
                .expect("hma batch kernel");
            self.cuda.stream.synchronize().expect("hma sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaHma::new(0).expect("cuda hma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = HmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };

        let (combos, first_valid, series_len, max_sqrt_len) =
            CudaHma::prepare_batch_inputs(&price, &sweep).expect("hma prepare batch inputs");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(0) as i32)
            .collect();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let ring_elems = n_combos
            .checked_mul(max_sqrt_len)
            .expect("ring elems overflow");
        let d_ring: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(ring_elems) }.expect("d_ring");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.expect("d_out");

        let block_x = match cuda.policy().batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => match std::env::var("HMA_BLOCK_X")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
            {
                Some(v) if v > 0 => v,
                _ => 1,
            },
        };
        let shared_bytes: usize = if CudaHma::ring_in_shared() {
            max_sqrt_len * (block_x as usize) * std::mem::size_of::<f32>()
        } else {
            0
        };

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(HmaBatchDevState {
            cuda,
            d_prices,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            max_sqrt_len,
            d_ring,
            d_out,
            block_x,
            shared_bytes,
        })
    }

    struct HmaManyDevState {
        cuda: CudaHma,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        sqrt_len: usize,
        d_ring: DeviceBuffer<f32>,
        d_out_tm: DeviceBuffer<f32>,
        block_x: u32,
        shared_bytes: usize,
    }
    impl CudaBenchState for HmaManyDevState {
        fn launch(&mut self) {
            let ring_ptr = unsafe { self.d_ring.as_device_ptr().as_raw() };
            let out_ptr = unsafe { self.d_out_tm.as_device_ptr().as_raw() };
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.period,
                    self.sqrt_len,
                    ring_ptr,
                    out_ptr,
                    self.block_x,
                    self.shared_bytes,
                )
                .expect("hma many-series kernel");
            self.cuda.stream.synchronize().expect("hma sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaHma::new(0).expect("cuda hma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = HmaParams { period: Some(64) };

        let (first_valids, period, sqrt_len) =
            CudaHma::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("hma prepare many-series inputs");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let ring_elems = cols.checked_mul(sqrt_len).expect("ring elems overflow");
        let d_ring: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(ring_elems) }.expect("d_ring");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");

        let block_x = match cuda.policy().many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => match std::env::var("HMA_MS_BLOCK_X")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
            {
                Some(v) if v == 128 || v == 256 || v == 512 => v,
                _ => 256,
            },
        };
        let shared_bytes: usize = if CudaHma::ring_in_shared() {
            sqrt_len * (block_x as usize) * std::mem::size_of::<f32>()
        } else {
            0
        };

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(HmaManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            sqrt_len,
            d_ring,
            d_out_tm,
            block_x,
            shared_bytes,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "hma",
                "one_series_many_params",
                "hma_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "hma",
                "many_series_one_param",
                "hma_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
