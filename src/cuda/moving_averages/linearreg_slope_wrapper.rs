#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::linearreg_slope::{LinearRegSlopeBatchRange, LinearRegSlopeParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Prefix { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaLinearregSlopePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaLinearregSlopePolicy {
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
    Prefix { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Error, Debug)]
pub enum CudaLinearregSlopeError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaLinearregSlope {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaLinearregSlopePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: i32,
}

impl CudaLinearregSlope {
    pub fn new(device_id: usize) -> Result<Self, CudaLinearregSlopeError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/linearreg_slope_kernel.ptx"));
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        ) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?,
            },
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaLinearregSlopePolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaLinearregSlopePolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaLinearregSlopePolicy {
        &self.policy
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaLinearregSlopeError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
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
                    eprintln!("[DEBUG] LRS batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut Self)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] LRS many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut Self)).debug_many_logged = true;
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
    fn will_fit(
        required_bytes: usize,
        headroom_bytes: usize,
    ) -> Result<(), CudaLinearregSlopeError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaLinearregSlopeError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    #[inline]
    fn validate_launch(
        &self,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    ) -> Result<(), CudaLinearregSlopeError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads {
            return Err(CudaLinearregSlopeError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            });
        }
        Ok(())
    }

    #[inline]
    fn expand_periods(
        sweep: &LinearRegSlopeBatchRange,
    ) -> Result<Vec<usize>, CudaLinearregSlopeError> {
        let (start, end, step) = sweep.period;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let st = step.max(1);
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(st) {
                    Some(next) => x = next,
                    None => break,
                }
            }
            if v.is_empty() {
                return Err(CudaLinearregSlopeError::InvalidInput(
                    "empty period sweep".into(),
                ));
            }
            return Ok(v);
        }

        let mut v = Vec::new();
        let st = step.max(1) as isize;
        let mut x = start as isize;
        let end_i = end as isize;
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(CudaLinearregSlopeError::InvalidInput(
                "empty period sweep".into(),
            ));
        }
        Ok(v)
    }

    #[allow(clippy::type_complexity)]
    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &LinearRegSlopeBatchRange,
    ) -> Result<
        (
            Vec<LinearRegSlopeParams>,
            usize,
            usize,
            Vec<i32>,
            Vec<f32>,
            Vec<f32>,
        ),
        CudaLinearregSlopeError,
    > {
        if data_f32.is_empty() {
            return Err(CudaLinearregSlopeError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaLinearregSlopeError::InvalidInput("all values are NaN".into()))?;

        let periods = Self::expand_periods(sweep)?;
        if periods.is_empty() {
            return Err(CudaLinearregSlopeError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let combos: Vec<LinearRegSlopeParams> = periods
            .iter()
            .map(|&p| LinearRegSlopeParams { period: Some(p) })
            .collect();

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut x_sums = Vec::with_capacity(combos.len());
        let mut denom_invs = Vec::with_capacity(combos.len());

        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaLinearregSlopeError::InvalidInput(
                    "period must be >= 2".into(),
                ));
            }
            if period > len {
                return Err(CudaLinearregSlopeError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaLinearregSlopeError::InvalidInput(format!(
                    "not enough valid data for period {} (tail={})",
                    period,
                    len - first_valid
                )));
            }

            let pf = period as f64;
            let x_sum = pf * (pf + 1.0) * 0.5;
            let x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
            let denom = pf * x2_sum - x_sum * x_sum;
            let denom_inv = 1.0 / denom;

            periods_i32.push(period as i32);
            x_sums.push(x_sum as f32);
            denom_invs.push(denom_inv as f32);
        }

        Ok((combos, first_valid, len, periods_i32, x_sums, denom_invs))
    }

    fn grid_1d_for(&self, work_items: usize, block_x: u32) -> (GridSize, u32) {
        let gx = ((work_items as u32) + block_x - 1) / block_x;
        let gx_clamped = gx.max(self.sm_count as u32);
        ((gx_clamped, 1, 1).into(), gx_clamped)
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_x_sums: &DeviceBuffer<f32>,
        d_denom_invs: &DeviceBuffer<f32>,
        series_len: usize,
        combos_len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinearregSlopeError> {
        let func = self
            .module
            .get_function("linearreg_slope_batch_f32")
            .map_err(|_| CudaLinearregSlopeError::MissingKernelSymbol {
                name: "linearreg_slope_batch_f32",
            })?;
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(256),
            BatchKernelPolicy::Prefix { .. } => {
                return Err(CudaLinearregSlopeError::InvalidPolicy(
                    "Prefix policy requires launch_batch_from_prefix_kernel",
                ));
            }
        };
        let (grid, gx) = self.grid_1d_for(combos_len, block_x);
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(gx, 1, 1, block_x, 1, 1)?;
        unsafe {
            (*(self as *const _ as *mut Self)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut xs_ptr = d_x_sums.as_device_ptr().as_raw();
            let mut dinv_ptr = d_denom_invs.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_len_i = combos_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut xs_ptr as *mut _ as *mut c_void,
                &mut dinv_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_prefix_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        d_prefix_y: &mut DeviceBuffer<f64>,
        d_prefix_yi: &mut DeviceBuffer<f64>,
    ) -> Result<(), CudaLinearregSlopeError> {
        let func = self
            .module
            .get_function("linearreg_slope_exclusive_prefix_y_yi_f64")
            .map_err(|_| CudaLinearregSlopeError::MissingKernelSymbol {
                name: "linearreg_slope_exclusive_prefix_y_yi_f64",
            })?;

        let grid: GridSize = (1u32, 1u32, 1u32).into();
        let block: BlockSize = (1u32, 1u32, 1u32).into();
        self.validate_launch(1, 1, 1, 1, 1, 1)?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut prefix_y_ptr = d_prefix_y.as_device_ptr().as_raw();
            let mut prefix_yi_ptr = d_prefix_yi.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut prefix_y_ptr as *mut _ as *mut c_void,
                &mut prefix_yi_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn launch_batch_from_prefix_kernel(
        &self,
        d_prefix_y: &DeviceBuffer<f64>,
        d_prefix_yi: &DeviceBuffer<f64>,
        d_periods: &DeviceBuffer<i32>,
        d_x_sums: &DeviceBuffer<f32>,
        d_denom_invs: &DeviceBuffer<f32>,
        series_len: usize,
        combos_len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinearregSlopeError> {
        let func = self
            .module
            .get_function("linearreg_slope_batch_from_prefix_f64")
            .map_err(|_| CudaLinearregSlopeError::MissingKernelSymbol {
                name: "linearreg_slope_batch_from_prefix_f64",
            })?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => match env::var("LINREG_PREFIX_BLOCK_X").ok().as_deref() {
                Some(s) => s
                    .parse::<u32>()
                    .ok()
                    .filter(|&v| v > 0)
                    .unwrap_or(256)
                    .max(32)
                    .min(256),
                None => 256,
            },
            BatchKernelPolicy::Prefix { block_x } => block_x.max(32).min(256),
            BatchKernelPolicy::Plain { .. } => {
                return Err(CudaLinearregSlopeError::InvalidPolicy(
                    "Plain policy requires launch_batch_kernel",
                ));
            }
        };

        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid_y = combos_len as u32;
        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(grid_x.max(1), grid_y.max(1), 1, block_x, 1, 1)?;

        unsafe {
            (*(self as *const _ as *mut Self)).last_batch =
                Some(BatchKernelSelected::Prefix { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prefix_y_ptr = d_prefix_y.as_device_ptr().as_raw();
            let mut prefix_yi_ptr = d_prefix_yi.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut xs_ptr = d_x_sums.as_device_ptr().as_raw();
            let mut dinv_ptr = d_denom_invs.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_len_i = combos_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prefix_y_ptr as *mut _ as *mut c_void,
                &mut prefix_yi_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut xs_ptr as *mut _ as *mut c_void,
                &mut dinv_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        periods_i32: &[i32],
        x_sums: &[f32],
        denom_invs: &[f32],
        len: usize,
        first_valid: usize,
    ) -> Result<DeviceArrayF32, CudaLinearregSlopeError> {
        let nrows = periods_i32.len();
        let elems = nrows.checked_mul(len).ok_or_else(|| {
            CudaLinearregSlopeError::InvalidInput("size overflow (rows*len)".into())
        })?;
        let prices_bytes = len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            CudaLinearregSlopeError::InvalidInput("size overflow (prices_bytes)".into())
        })?;
        let periods_bytes = nrows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaLinearregSlopeError::InvalidInput("size overflow (periods_bytes)".into())
            })?;
        let consts_bytes = nrows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaLinearregSlopeError::InvalidInput("size overflow (consts_bytes)".into())
            })?;
        let consts2_bytes = consts_bytes.checked_mul(2).ok_or_else(|| {
            CudaLinearregSlopeError::InvalidInput("size overflow (consts_bytes*2)".into())
        })?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaLinearregSlopeError::InvalidInput("size overflow (out_bytes)".into())
            })?;

        let required = if matches!(self.policy.batch, BatchKernelPolicy::Plain { .. }) {
            prices_bytes
                .checked_add(periods_bytes)
                .and_then(|v| v.checked_add(consts2_bytes))
                .and_then(|v| v.checked_add(out_bytes))
                .ok_or_else(|| {
                    CudaLinearregSlopeError::InvalidInput("size overflow (bytes)".into())
                })?
        } else {
            let prefix_elems = len.checked_add(1).ok_or_else(|| {
                CudaLinearregSlopeError::InvalidInput("size overflow (len+1)".into())
            })?;
            let prefix_bytes = prefix_elems
                .checked_mul(std::mem::size_of::<f64>())
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| {
                    CudaLinearregSlopeError::InvalidInput("size overflow (prefix_bytes)".into())
                })?;
            prices_bytes
                .checked_add(periods_bytes)
                .and_then(|v| v.checked_add(consts2_bytes))
                .and_then(|v| v.checked_add(prefix_bytes))
                .and_then(|v| v.checked_add(out_bytes))
                .ok_or_else(|| {
                    CudaLinearregSlopeError::InvalidInput("size overflow (bytes)".into())
                })?
        };
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let d_periods = DeviceBuffer::from_slice(periods_i32)?;
        let d_xs = DeviceBuffer::from_slice(x_sums)?;
        let d_dinv = DeviceBuffer::from_slice(denom_invs)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        match self.policy.batch {
            BatchKernelPolicy::Plain { .. } => {
                self.launch_batch_kernel(
                    &d_prices,
                    &d_periods,
                    &d_xs,
                    &d_dinv,
                    len,
                    nrows,
                    first_valid,
                    &mut d_out,
                )?;
            }
            BatchKernelPolicy::Auto | BatchKernelPolicy::Prefix { .. } => {
                let mut d_prefix_y = unsafe { DeviceBuffer::<f64>::uninitialized(len + 1) }?;
                let mut d_prefix_yi = unsafe { DeviceBuffer::<f64>::uninitialized(len + 1) }?;
                self.launch_prefix_kernel(
                    &d_prices,
                    len,
                    first_valid,
                    &mut d_prefix_y,
                    &mut d_prefix_yi,
                )?;
                self.launch_batch_from_prefix_kernel(
                    &d_prefix_y,
                    &d_prefix_yi,
                    &d_periods,
                    &d_xs,
                    &d_dinv,
                    len,
                    nrows,
                    first_valid,
                    &mut d_out,
                )?;
            }
        }
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: nrows,
            cols: len,
        })
    }

    pub fn linearreg_slope_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &LinearRegSlopeBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<LinearRegSlopeParams>), CudaLinearregSlopeError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(
            data_f32,
            &periods_i32,
            &x_sums,
            &denom_invs,
            len,
            first_valid,
        )?;
        self.stream.synchronize()?;
        Ok((dev, combos))
    }

    pub fn linearreg_slope_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &LinearRegSlopeBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<LinearRegSlopeParams>), CudaLinearregSlopeError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let nrows = combos.len();
        let expected = nrows.checked_mul(len).ok_or_else(|| {
            CudaLinearregSlopeError::InvalidInput("size overflow (rows*len)".into())
        })?;
        if out.len() != expected {
            return Err(CudaLinearregSlopeError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(
            data_f32,
            &periods_i32,
            &x_sums,
            &denom_invs,
            len,
            first_valid,
        )?;
        dev.buf.copy_to(out)?;
        self.stream.synchronize()?;
        Ok((nrows, len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinearRegSlopeParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaLinearregSlopeError> {
        if cols == 0 || rows == 0 {
            return Err(CudaLinearregSlopeError::InvalidInput("empty matrix".into()));
        }
        let elems = cols.checked_mul(rows).ok_or_else(|| {
            CudaLinearregSlopeError::InvalidInput("size overflow (cols*rows)".into())
        })?;
        if data_tm_f32.len() != elems {
            return Err(CudaLinearregSlopeError::InvalidInput(
                "invalid time-major shape".into(),
            ));
        }
        let period = params.period.unwrap_or(0);
        if period < 2 || period > rows {
            return Err(CudaLinearregSlopeError::InvalidInput(format!(
                "invalid period {} for rows {}",
                period, rows
            )));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for r in 0..rows {
                let v = data_tm_f32[r * cols + s];
                if !v.is_nan() {
                    fv = r as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaLinearregSlopeError::InvalidInput(format!(
                    "series {} contains only NaN",
                    s
                )));
            }
            first_valids[s] = fv;
            if (rows as i32 - fv) < period as i32 {
                return Err(CudaLinearregSlopeError::InvalidInput(format!(
                    "series {} insufficient data for period {} (first_valid={}, rows={})",
                    s, period, fv, rows
                )));
            }
        }
        let pf = period as f64;
        let x_sum = (pf * (pf + 1.0) * 0.5) as f32;
        let x2_sum = (pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0) as f32;
        let denom_inv = (1.0f64 / (pf * x2_sum as f64 - (x_sum as f64) * (x_sum as f64))) as f32;
        Ok((first_valids, period, x_sum, denom_inv))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        x_sum: f32,
        denom_inv: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinearregSlopeError> {
        let func = self
            .module
            .get_function("linearreg_slope_many_series_one_param_f32")
            .map_err(|_| CudaLinearregSlopeError::MissingKernelSymbol {
                name: "linearreg_slope_many_series_one_param_f32",
            })?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(256),
        };
        let (grid, gx) = self.grid_1d_for(cols, block_x);
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(gx, 1, 1, block_x, 1, 1)?;
        unsafe {
            (*(self as *const _ as *mut Self)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut x_sum_f = x_sum;
            let mut denom_inv_f = denom_inv;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut x_sum_f as *mut _ as *mut c_void,
                &mut denom_inv_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
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
        x_sum: f32,
        denom_inv: f32,
    ) -> Result<DeviceArrayF32, CudaLinearregSlopeError> {
        let elems = cols.checked_mul(rows).ok_or_else(|| {
            CudaLinearregSlopeError::InvalidInput("size overflow (cols*rows)".into())
        })?;
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaLinearregSlopeError::InvalidInput("size overflow (prices_bytes)".into())
            })?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaLinearregSlopeError::InvalidInput("size overflow (first_bytes)".into())
            })?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaLinearregSlopeError::InvalidInput("size overflow (out_bytes)".into())
            })?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaLinearregSlopeError::InvalidInput("size overflow (bytes)".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(first_valids)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        self.launch_many_series_kernel(
            &d_prices, &d_first, cols, rows, period, x_sum, denom_inv, &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn linearreg_slope_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinearRegSlopeParams,
    ) -> Result<DeviceArrayF32, CudaLinearregSlopeError> {
        let (first_valids, period, x_sum, denom_inv) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            x_sum,
            denom_inv,
        )?;
        self.stream.synchronize()?;
        Ok(dev)
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::linearreg_slope::LinearRegSlopeParams;

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

    struct BatchDevState {
        cuda: CudaLinearregSlope,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_x_sums: DeviceBuffer<f32>,
        d_denom_invs: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_x_sums,
                    &self.d_denom_invs,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("linearreg_slope batch kernel");
            self.cuda
                .stream
                .synchronize()
                .expect("linearreg_slope sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaLinearregSlope::new(0).expect("cuda linearreg_slope");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = crate::indicators::linearreg_slope::LinearRegSlopeBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (_combos, first_valid, series_len, periods_i32, x_sums, denom_invs) =
            CudaLinearregSlope::prepare_batch_inputs(&price, &sweep)
                .expect("linearreg_slope prepare batch");
        let n_combos = periods_i32.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_x_sums = DeviceBuffer::from_slice(&x_sums).expect("d_x_sums");
        let d_denom_invs = DeviceBuffer::from_slice(&denom_invs).expect("d_denom_invs");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(series_len.checked_mul(n_combos).expect("out size"))
        }
        .expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(BatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_x_sums,
            d_denom_invs,
            series_len,
            n_combos,
            first_valid,
            d_out,
        })
    }

    struct ManyDevState {
        cuda: CudaLinearregSlope,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        x_sum: f32,
        denom_inv: f32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.period,
                    self.x_sum,
                    self.denom_inv,
                    &mut self.d_out_tm,
                )
                .expect("linearreg_slope many-series kernel");
            self.cuda
                .stream
                .synchronize()
                .expect("linearreg_slope sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaLinearregSlope::new(0).expect("cuda linearreg_slope");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = LinearRegSlopeParams { period: Some(64) };
        let (first_valids, period, x_sum, denom_inv) =
            CudaLinearregSlope::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("linearreg_slope prepare many");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols.checked_mul(rows).expect("out size")) }
                .expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(ManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            x_sum,
            denom_inv,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "linearreg_slope",
                "one_series_many_params",
                "linearreg_slope_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "linearreg_slope",
                "many_series_one_param",
                "linearreg_slope_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
