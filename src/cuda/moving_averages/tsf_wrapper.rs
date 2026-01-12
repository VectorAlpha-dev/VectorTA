#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::tsf::{TsfBatchRange, TsfParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, CopyDestination, DeviceBuffer};
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
pub struct CudaTsfPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaTsfPolicy {
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
pub enum CudaTsfError {
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

pub struct CudaTsf {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaTsfPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: i32,
}

impl CudaTsf {
    pub fn new(device_id: usize) -> Result<Self, CudaTsfError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/tsf_kernel.ptx"));
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
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
            policy: CudaTsfPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaTsfPolicy) -> Result<Self, CudaTsfError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    pub fn synchronize(&self) -> Result<(), CudaTsfError> {
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
                    eprintln!("[DEBUG] TSF batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTsf)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] TSF many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTsf)).debug_many_logged = true;
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
        cust::memory::mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
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
    ) -> Result<(), CudaTsfError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads
            || bx > max_bx
            || by > max_by
            || bz > max_bz
            || gx > max_gx
            || gy > max_gy
            || gz > max_gz
        {
            return Err(CudaTsfError::LaunchConfigTooLarge {
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

    #[allow(clippy::type_complexity)]
    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &TsfBatchRange,
    ) -> Result<
        (
            Vec<TsfParams>,
            usize,
            usize,
            Vec<i32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
        ),
        CudaTsfError,
    > {
        if data_f32.is_empty() {
            return Err(CudaTsfError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTsfError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_local(sweep);
        if combos.is_empty() {
            let (start, end, step) = sweep.period;
            return Err(CudaTsfError::InvalidInput(format!(
                "invalid TSF batch range: start={}, end={}, step={}",
                start, end, step
            )));
        }

        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if combos.len().checked_mul(max_period).is_none() {
            return Err(CudaTsfError::InvalidInput(
                "combos * max_period overflow".into(),
            ));
        }

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut x_sums = Vec::with_capacity(combos.len());
        let mut denom_invs = Vec::with_capacity(combos.len());
        let mut inv_periods = Vec::with_capacity(combos.len());

        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaTsfError::InvalidInput("period must be >= 2".into()));
            }
            if period > len {
                return Err(CudaTsfError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaTsfError::InvalidInput(format!(
                    "not enough valid data for period {} (tail = {})",
                    period,
                    len - first_valid
                )));
            }

            let pf = period as f64;

            let x_sum = pf * (pf + 1.0) * 0.5;
            let x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
            let denom = pf * x2_sum - x_sum * x_sum;
            periods_i32.push(period as i32);
            x_sums.push(x_sum as f32);
            denom_invs.push((1.0 / denom) as f32);
            inv_periods.push((1.0 / pf) as f32);
        }

        Ok((
            combos,
            first_valid,
            len,
            periods_i32,
            x_sums,
            denom_invs,
            inv_periods,
        ))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_x_sums: &DeviceBuffer<f32>,
        d_denom_invs: &DeviceBuffer<f32>,
        d_inv_periods: &DeviceBuffer<f32>,
        series_len: usize,
        combos_len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsfError> {
        let func = self.module.get_function("tsf_batch_f32").map_err(|_| {
            CudaTsfError::MissingKernelSymbol {
                name: "tsf_batch_f32",
            }
        })?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(256),
            BatchKernelPolicy::Prefix { .. } => {
                return Err(CudaTsfError::InvalidPolicy(
                    "Prefix policy requires launch_batch_from_prefix_kernel",
                ));
            }
        };
        let bx = block_x.max(64).min(256);
        let grid_x = ((combos_len as u32) + bx - 1) / bx;
        let gx = grid_x.max(self.sm_count as u32);
        self.validate_launch(gx, 1, 1, bx, 1, 1)?;
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (bx, 1, 1).into();

        unsafe {
            (*(self as *const _ as *mut CudaTsf)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut x_sums_ptr = d_x_sums.as_device_ptr().as_raw();
            let mut denom_ptr = d_denom_invs.as_device_ptr().as_raw();
            let mut inv_p_ptr = d_inv_periods.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut combos_i = combos_len as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut x_sums_ptr as *mut _ as *mut c_void,
                &mut denom_ptr as *mut _ as *mut c_void,
                &mut inv_p_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
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
    ) -> Result<(), CudaTsfError> {
        let func = self
            .module
            .get_function("tsf_exclusive_prefix_y_yi_f64")
            .map_err(|_| CudaTsfError::MissingKernelSymbol {
                name: "tsf_exclusive_prefix_y_yi_f64",
            })?;

        let grid: GridSize = (1u32, 1u32, 1u32).into();
        let block: BlockSize = (1u32, 1u32, 1u32).into();
        self.validate_launch(1, 1, 1, 1, 1, 1)?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut first_i = first_valid as i32;
            let mut prefix_y_ptr = d_prefix_y.as_device_ptr().as_raw();
            let mut prefix_yi_ptr = d_prefix_yi.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
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
        d_inv_periods: &DeviceBuffer<f32>,
        series_len: usize,
        combos_len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsfError> {
        let func = self
            .module
            .get_function("tsf_batch_from_prefix_f64")
            .map_err(|_| CudaTsfError::MissingKernelSymbol {
                name: "tsf_batch_from_prefix_f64",
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
                return Err(CudaTsfError::InvalidPolicy(
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
            (*(self as *const _ as *mut CudaTsf)).last_batch =
                Some(BatchKernelSelected::Prefix { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prefix_y_ptr = d_prefix_y.as_device_ptr().as_raw();
            let mut prefix_yi_ptr = d_prefix_yi.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut x_sums_ptr = d_x_sums.as_device_ptr().as_raw();
            let mut denom_ptr = d_denom_invs.as_device_ptr().as_raw();
            let mut inv_p_ptr = d_inv_periods.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut combos_i = combos_len as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prefix_y_ptr as *mut _ as *mut c_void,
                &mut prefix_yi_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut x_sums_ptr as *mut _ as *mut c_void,
                &mut denom_ptr as *mut _ as *mut c_void,
                &mut inv_p_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[TsfParams],
        first_valid: usize,
        series_len: usize,
        periods_i32: &[i32],
        x_sums: &[f32],
        denom_invs: &[f32],
        inv_periods: &[f32],
    ) -> Result<DeviceArrayF32, CudaTsfError> {
        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaTsfError::InvalidInput("series_len * sizeof(f32) overflow".into())
            })?;
        let per_combo_bytes = std::mem::size_of::<i32>() + 3 * std::mem::size_of::<f32>();
        let params_bytes = combos
            .len()
            .checked_mul(per_combo_bytes)
            .ok_or_else(|| CudaTsfError::InvalidInput("combos * param bytes overflow".into()))?;
        let out_elems = combos
            .len()
            .checked_mul(series_len)
            .ok_or_else(|| CudaTsfError::InvalidInput("combos * series_len overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaTsfError::InvalidInput("output bytes overflow".into()))?;
        let required = if matches!(self.policy.batch, BatchKernelPolicy::Plain { .. }) {
            prices_bytes
                .checked_add(params_bytes)
                .and_then(|v| v.checked_add(out_bytes))
                .ok_or_else(|| CudaTsfError::InvalidInput("required bytes overflow".into()))?
        } else {
            let prefix_elems = series_len
                .checked_add(1)
                .ok_or_else(|| CudaTsfError::InvalidInput("prefix elems overflow".into()))?;
            let prefix_bytes = prefix_elems
                .checked_mul(std::mem::size_of::<f64>())
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| CudaTsfError::InvalidInput("prefix bytes overflow".into()))?;
            prices_bytes
                .checked_add(params_bytes)
                .and_then(|v| v.checked_add(prefix_bytes))
                .and_then(|v| v.checked_add(out_bytes))
                .ok_or_else(|| CudaTsfError::InvalidInput("required bytes overflow".into()))?
        };
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaTsfError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaTsfError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream)? };
        let d_periods = DeviceBuffer::from_slice(periods_i32)?;
        let d_x_sums = DeviceBuffer::from_slice(x_sums)?;
        let d_denoms = DeviceBuffer::from_slice(denom_invs)?;
        let d_inv_p = DeviceBuffer::from_slice(inv_periods)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(out_elems)? };

        match self.policy.batch {
            BatchKernelPolicy::Plain { .. } => {
                self.launch_batch_kernel(
                    &d_prices,
                    &d_periods,
                    &d_x_sums,
                    &d_denoms,
                    &d_inv_p,
                    series_len,
                    combos.len(),
                    first_valid,
                    &mut d_out,
                )?;
            }
            BatchKernelPolicy::Auto | BatchKernelPolicy::Prefix { .. } => {
                let mut d_prefix_y = unsafe { DeviceBuffer::<f64>::uninitialized(series_len + 1)? };
                let mut d_prefix_yi =
                    unsafe { DeviceBuffer::<f64>::uninitialized(series_len + 1)? };
                self.launch_prefix_kernel(
                    &d_prices,
                    series_len,
                    first_valid,
                    &mut d_prefix_y,
                    &mut d_prefix_yi,
                )?;
                self.launch_batch_from_prefix_kernel(
                    &d_prefix_y,
                    &d_prefix_yi,
                    &d_periods,
                    &d_x_sums,
                    &d_denoms,
                    &d_inv_p,
                    series_len,
                    combos.len(),
                    first_valid,
                    &mut d_out,
                )?;
            }
        }
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: series_len,
        })
    }

    pub fn tsf_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &TsfBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<TsfParams>), CudaTsfError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs, inv_periods) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let arr = self.run_batch_kernel(
            data_f32,
            &combos,
            first_valid,
            len,
            &periods_i32,
            &x_sums,
            &denom_invs,
            &inv_periods,
        )?;
        self.stream.synchronize()?;
        Ok((arr, combos))
    }

    pub fn tsf_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &TsfBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TsfParams>), CudaTsfError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs, inv_periods) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaTsfError::InvalidInput("combos * len overflow".into()))?;
        if out.len() != expected {
            return Err(CudaTsfError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let dev = self.run_batch_kernel(
            data_f32,
            &combos,
            first_valid,
            len,
            &periods_i32,
            &x_sums,
            &denom_invs,
            &inv_periods,
        )?;
        self.stream.synchronize()?;
        dev.buf.copy_to(out)?;
        Ok((dev.rows, dev.cols, combos))
    }

    #[allow(clippy::type_complexity)]
    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TsfParams,
    ) -> Result<(Vec<i32>, usize, f32, f32, f32), CudaTsfError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTsfError::InvalidInput("cols * rows overflow".into()))?;
        if data_tm_f32.len() != expected {
            return Err(CudaTsfError::InvalidInput("data_tm size mismatch".into()));
        }
        let period = params.period.unwrap_or(0);
        if period < 2 {
            return Err(CudaTsfError::InvalidInput("period must be >= 2".into()));
        }
        if period > rows {
            return Err(CudaTsfError::InvalidInput(
                "period exceeds series length".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaTsfError::InvalidInput(format!("series {} all NaN", s)))?;
            if (rows as i32 - fv) < period as i32 {
                return Err(CudaTsfError::InvalidInput(format!(
                    "series {} not enough valid data (needed {}, have {})",
                    s,
                    period,
                    rows as i32 - fv
                )));
            }
            first_valids[s] = fv;
        }

        let pf = period as f64;
        let x_sum = (pf * (pf + 1.0) * 0.5) as f32;
        let x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
        let denom_inv = (1.0 / (pf * x2_sum - (pf * (pf + 1.0) * 0.5).powi(2))) as f32;
        let inv_period = (1.0 / pf) as f32;
        Ok((first_valids, period, x_sum, denom_inv, inv_period))
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
        inv_period: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTsfError> {
        let func = self
            .module
            .get_function("tsf_many_series_one_param_f32")
            .map_err(|_| CudaTsfError::MissingKernelSymbol {
                name: "tsf_many_series_one_param_f32",
            })?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(64).min(256),
        };
        let bx = block_x.max(64).min(256);
        let grid_x = ((cols as u32) + bx - 1) / bx;
        let gx = grid_x.max(self.sm_count as u32);
        self.validate_launch(gx, 1, 1, bx, 1, 1)?;
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (bx, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaTsf)).last_many =
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
            let mut inv_period_f = inv_period;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut x_sum_f as *mut _ as *mut c_void,
                &mut denom_inv_f as *mut _ as *mut c_void,
                &mut inv_period_f as *mut _ as *mut c_void,
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
        inv_period: f32,
    ) -> Result<DeviceArrayF32, CudaTsfError> {
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTsfError::InvalidInput("cols * rows overflow".into()))?;
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaTsfError::InvalidInput("prices bytes overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaTsfError::InvalidInput("first_valid bytes overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaTsfError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaTsfError::InvalidInput("required bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _)) = Self::device_mem_info() {
                return Err(CudaTsfError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            } else {
                return Err(CudaTsfError::InvalidInput(
                    "insufficient device memory".into(),
                ));
            }
        }
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(first_valids)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems)? };

        self.launch_many_series_kernel(
            &d_prices, &d_first, cols, rows, period, x_sum, denom_inv, inv_period, &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn tsf_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TsfParams,
    ) -> Result<DeviceArrayF32, CudaTsfError> {
        let (first_valids, period, x_sum, denom_inv, inv_period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            x_sum,
            denom_inv,
            inv_period,
        )?;
        self.stream.synchronize()?;
        Ok(dev)
    }

    pub fn tsf_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TsfParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTsfError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTsfError::InvalidInput("cols * rows overflow".into()))?;
        if out_tm.len() != expected {
            return Err(CudaTsfError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out_tm.len()
            )));
        }
        let (first_valids, period, x_sum, denom_inv, inv_period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            x_sum,
            denom_inv,
            inv_period,
        )?;
        self.stream.synchronize()?;
        dev.buf.copy_to(out_tm)?;
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::tsf::TsfParams;

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

    struct TsfBatchDevState {
        cuda: CudaTsf,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_x_sums: DeviceBuffer<f32>,
        d_denom_invs: DeviceBuffer<f32>,
        d_inv_periods: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TsfBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_x_sums,
                    &self.d_denom_invs,
                    &self.d_inv_periods,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("tsf batch kernel");
            self.cuda.stream.synchronize().expect("tsf sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaTsf::new(0).expect("cuda tsf");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = TsfBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (_combos, first_valid, len, periods_i32, x_sums, denom_invs, inv_periods) =
            CudaTsf::prepare_batch_inputs(&price, &sweep).expect("tsf prepare batch inputs");
        let n_combos = periods_i32.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_x_sums = DeviceBuffer::from_slice(&x_sums).expect("d_x_sums");
        let d_denom_invs = DeviceBuffer::from_slice(&denom_invs).expect("d_denom_invs");
        let d_inv_periods = DeviceBuffer::from_slice(&inv_periods).expect("d_inv_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * len) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(TsfBatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_x_sums,
            d_denom_invs,
            d_inv_periods,
            series_len: len,
            n_combos,
            first_valid,
            d_out,
        })
    }

    struct TsfManyDevState {
        cuda: CudaTsf,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        x_sum: f32,
        denom_inv: f32,
        inv_period: f32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TsfManyDevState {
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
                    self.inv_period,
                    &mut self.d_out_tm,
                )
                .expect("tsf many-series kernel");
            self.cuda.stream.synchronize().expect("tsf sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaTsf::new(0).expect("cuda tsf");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = TsfParams { period: Some(64) };
        let (first_valids, period, x_sum, denom_inv, inv_period) =
            CudaTsf::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("tsf prepare many-series inputs");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(TsfManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            x_sum,
            denom_inv,
            inv_period,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "tsf",
                "one_series_many_params",
                "tsf_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "tsf",
                "many_series_one_param",
                "tsf_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

#[inline]
fn expand_grid_local(r: &TsfBatchRange) -> Vec<TsfParams> {
    let (start, end, step) = r.period;
    if step == 0 || start == end {
        return vec![TsfParams {
            period: Some(start),
        }];
    }
    let mut v = Vec::new();
    if start <= end {
        let mut p = start;
        loop {
            if p > end {
                break;
            }
            v.push(TsfParams { period: Some(p) });
            match p.checked_add(step) {
                Some(nxt) => p = nxt,
                None => break,
            }
        }
    } else {
        let mut p = start;
        loop {
            if p < end {
                break;
            }
            v.push(TsfParams { period: Some(p) });
            if p == end {
                break;
            }
            match p.checked_sub(step) {
                Some(nxt) => p = nxt,
                None => break,
            }
        }
    }
    v
}
