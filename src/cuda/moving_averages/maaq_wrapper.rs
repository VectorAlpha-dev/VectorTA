#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::maaq::{expand_grid, MaaqBatchRange, MaaqParams};
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

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

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaMaaqPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Debug, Error)]
pub enum CudaMaaqError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error(
        "Out of memory on device: required={required} bytes, free={free} bytes, headroom={headroom} bytes"
    )]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Launch configuration too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buffer on device {buf}, current device {current}")]
    DeviceMismatch { buf: i32, current: i32 },
    #[error("arithmetic overflow computing {0}")]
    ArithmeticOverflow(&'static str),
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

pub struct DeviceArrayF32Maaq {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub device_id: u32,
    pub(crate) _ctx: Arc<Context>,
}

impl DeviceArrayF32Maaq {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaMaaq {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaMaaqPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,

    max_grid_x: usize,
    max_threads_per_block: u32,
}

impl CudaMaaq {
    pub fn maaq_batch_dev_ex(
        &self,
        data_f32: &[f32],
        sweep: &MaaqBatchRange,
    ) -> Result<DeviceArrayF32Maaq, CudaMaaqError> {
        let (combos, first_valid, len, max_period) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        let prices_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaMaaqError::ArithmeticOverflow("prices_bytes"))?;
        let params_each = std::mem::size_of::<i32>()
            .checked_add(2 * std::mem::size_of::<f32>())
            .ok_or(CudaMaaqError::ArithmeticOverflow("params_each"))?;
        let params_bytes = n_combos
            .checked_mul(params_each)
            .ok_or(CudaMaaqError::ArithmeticOverflow("params_bytes"))?;
        let out_elems = n_combos
            .checked_mul(len)
            .ok_or(CudaMaaqError::ArithmeticOverflow("out_elems"))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaMaaqError::ArithmeticOverflow("out_bytes"))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or(CudaMaaqError::ArithmeticOverflow("required_bytes"))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit_checked(required, headroom)?;

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut fast_scs = Vec::with_capacity(n_combos);
        let mut slow_scs = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            let fast = prm.fast_period.unwrap();
            let slow = prm.slow_period.unwrap();
            periods_i32.push(period as i32);
            fast_scs.push(2.0f32 / (fast as f32 + 1.0f32));
            slow_scs.push(2.0f32 / (slow as f32 + 1.0f32));
        }

        let d_prices = self.upload_f32_large(data_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_fast = DeviceBuffer::from_slice(&fast_scs)?;
        let d_slow = DeviceBuffer::from_slice(&slow_scs)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * len, &self.stream) }?;

        self.launch_batch_kernel_plain(
            &d_prices,
            &d_periods,
            &d_fast,
            &d_slow,
            first_valid,
            len,
            n_combos,
            max_period,
            &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32Maaq {
            buf: d_out,
            rows: n_combos,
            cols: len,
            device_id: self.device_id,
            _ctx: Arc::clone(&self._context),
        })
    }

    pub fn new(device_id: usize) -> Result<Self, CudaMaaqError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;
        let context = Arc::new(context);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/maaq_kernel.ptx"));

        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let max_grid_x = device.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)? as usize;
        let max_threads_per_block =
            device.get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)? as u32;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaMaaqPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            max_grid_x,
            max_threads_per_block,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaMaaqPolicy,
    ) -> Result<Self, CudaMaaqError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaMaaqPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaMaaqPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
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
                    eprintln!("[DEBUG] MAAQ batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMaaq)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] MAAQ many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaMaaq)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMaaqError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        match mem_get_info() {
            Ok((free, _total)) => {
                let need = required_bytes
                    .checked_add(headroom_bytes)
                    .ok_or(CudaMaaqError::ArithmeticOverflow("required+headroom"))?;
                if need <= free {
                    Ok(())
                } else {
                    Err(CudaMaaqError::OutOfMemory {
                        required: required_bytes,
                        free,
                        headroom: headroom_bytes,
                    })
                }
            }
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn chunk_pairs(total: usize, chunk_max: usize) -> impl Iterator<Item = (usize, usize)> {
        (0..total).step_by(chunk_max).map(move |start| {
            let len = (total - start).min(chunk_max);
            (start, len)
        })
    }

    #[inline]
    fn grid_x_limit(&self) -> usize {
        self.max_grid_x
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &MaaqBatchRange,
    ) -> Result<(Vec<MaaqParams>, usize, usize, usize), CudaMaaqError> {
        if data_f32.is_empty() {
            return Err(CudaMaaqError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaMaaqError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaMaaqError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            let fast = prm.fast_period.unwrap_or(0);
            let slow = prm.slow_period.unwrap_or(0);
            if period == 0 || fast == 0 || slow == 0 {
                return Err(CudaMaaqError::InvalidInput(
                    "period, fast_period, and slow_period must be > 0".into(),
                ));
            }
            if period > len {
                return Err(CudaMaaqError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaMaaqError::InvalidInput(format!(
                    "not enough valid data: need {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            if period > i32::MAX as usize {
                return Err(CudaMaaqError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            if max_period < period {
                max_period = period;
            }
        }

        Ok((combos, first_valid, len, max_period))
    }

    fn launch_batch_kernel_plain(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_fast_scs: &DeviceBuffer<f32>,
        d_slow_scs: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if series_len == 0 || n_combos == 0 || max_period == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaMaaqError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }
        if first_valid > i32::MAX as usize {
            return Err(CudaMaaqError::InvalidInput(
                "first_valid exceeds i32::MAX".into(),
            ));
        }

        let func = self.module.get_function("maaq_batch_f32").map_err(|_| {
            CudaMaaqError::MissingKernelSymbol {
                name: "maaq_batch_f32",
            }
        })?;

        let mut block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 32u32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
        };
        if block_x > self.max_threads_per_block {
            return Err(CudaMaaqError::LaunchConfigTooLarge {
                gx: n_combos as u32,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        unsafe {
            (*(self as *const _ as *mut CudaMaaq)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        let max_chunk = self.grid_x_limit();
        for (start, len) in Self::chunk_pairs(n_combos, max_chunk) {
            let grid: GridSize = (len as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut fast_ptr = d_fast_scs.as_device_ptr().add(start).as_raw();
                let mut slow_ptr = d_slow_scs.as_device_ptr().add(start).as_raw();
                let mut first_valid_i = first_valid as i32;
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = len as i32;
                let mut max_period_i = max_period as i32;

                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut fast_ptr as *mut _ as *mut c_void,
                    &mut slow_ptr as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaMaaqError::Cuda(e))?;
            }
        }

        Ok(())
    }

    pub fn maaq_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_fast_scs: &DeviceBuffer<f32>,
        d_slow_scs: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if series_len == 0 || n_combos == 0 || max_period == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        if first_valid > series_len {
            return Err(CudaMaaqError::InvalidInput(
                "first_valid out of range".into(),
            ));
        }
        self.launch_batch_kernel_plain(
            d_prices,
            d_periods,
            d_fast_scs,
            d_slow_scs,
            first_valid,
            series_len,
            n_combos,
            max_period,
            d_out,
        )
    }

    pub fn maaq_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MaaqBatchRange,
    ) -> Result<DeviceArrayF32, CudaMaaqError> {
        let (combos, first_valid, len, max_period) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        let prices_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaMaaqError::ArithmeticOverflow("prices_bytes"))?;
        let params_each = std::mem::size_of::<i32>()
            .checked_add(2 * std::mem::size_of::<f32>())
            .ok_or(CudaMaaqError::ArithmeticOverflow("params_each"))?;
        let params_bytes = n_combos
            .checked_mul(params_each)
            .ok_or(CudaMaaqError::ArithmeticOverflow("params_bytes"))?;
        let out_elems = n_combos
            .checked_mul(len)
            .ok_or(CudaMaaqError::ArithmeticOverflow("out_elems"))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaMaaqError::ArithmeticOverflow("out_bytes"))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or(CudaMaaqError::ArithmeticOverflow("required_bytes"))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit_checked(required, headroom)?;

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut fast_scs = Vec::with_capacity(n_combos);
        let mut slow_scs = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            let fast = prm.fast_period.unwrap();
            let slow = prm.slow_period.unwrap();
            periods_i32.push(period as i32);
            fast_scs.push(2.0f32 / (fast as f32 + 1.0f32));
            slow_scs.push(2.0f32 / (slow as f32 + 1.0f32));
        }

        let d_prices = self.upload_f32_large(data_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_fast = DeviceBuffer::from_slice(&fast_scs)?;
        let d_slow = DeviceBuffer::from_slice(&slow_scs)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * len, &self.stream) }?;

        self.launch_batch_kernel_plain(
            &d_prices,
            &d_periods,
            &d_fast,
            &d_slow,
            first_valid,
            len,
            n_combos,
            max_period,
            &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaMaaqError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaMaaqError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        let fast = params.fast_period.unwrap_or(0);
        let slow = params.slow_period.unwrap_or(0);
        if period == 0 || fast == 0 || slow == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "period, fast_period, and slow_period must be > 0".into(),
            ));
        }
        if period > rows {
            return Err(CudaMaaqError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        if period > i32::MAX as usize {
            return Err(CudaMaaqError::InvalidInput(
                "period exceeds i32::MAX".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaMaaqError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period {
                return Err(CudaMaaqError::InvalidInput(format!(
                    "series {} lacks enough valid data: need {} have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaMaaqError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        let fast_sc = 2.0f32 / (fast as f32 + 1.0f32);
        let slow_sc = 2.0f32 / (slow as f32 + 1.0f32);

        Ok((first_valids, period, fast_sc, slow_sc))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        fast_sc: f32,
        slow_sc: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaMaaqError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaMaaqError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("maaq_multi_series_one_param_f32")
            .map_err(|_| CudaMaaqError::MissingKernelSymbol {
                name: "maaq_multi_series_one_param_f32",
            })?;

        let mut block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 32u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        };
        if block_x > self.max_threads_per_block {
            return Err(CudaMaaqError::LaunchConfigTooLarge {
                gx: num_series as u32,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        unsafe {
            (*(self as *const _ as *mut CudaMaaq)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let max_chunk = self.grid_x_limit();
        for (start, len) in Self::chunk_pairs(num_series, max_chunk) {
            let grid: GridSize = (len as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

            unsafe {
                let mut prices_ptr = d_prices_tm.as_device_ptr().add(start).as_raw();
                let mut period_i = period as i32;
                let mut fast = fast_sc;
                let mut slow = slow_sc;
                let mut num_series_i = num_series as i32;
                let mut series_len_i = series_len as i32;
                let mut first_ptr = d_first_valids.as_device_ptr().add(start).as_raw();

                let mut out_ptr = d_out_tm.as_device_ptr().add(start).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut fast as *mut _ as *mut c_void,
                    &mut slow as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaMaaqError::Cuda(e))?;
            }
        }

        Ok(())
    }

    pub fn maaq_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: i32,
        fast_sc: f32,
        slow_sc: f32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMaaqError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaMaaqError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            period as usize,
            fast_sc,
            slow_sc,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn maaq_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
    ) -> Result<DeviceArrayF32, CudaMaaqError> {
        let (first_valids, period, fast_sc, slow_sc) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let prices_bytes = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaMaaqError::ArithmeticOverflow("prices_bytes"))?;
        let first_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(CudaMaaqError::ArithmeticOverflow("first_bytes"))?;
        let out_bytes = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaMaaqError::ArithmeticOverflow("out_bytes"))?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or(CudaMaaqError::ArithmeticOverflow("required_bytes"))?;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        let d_prices_tm = self.upload_f32_large(data_tm_f32)?;
        let d_first_valids =
            DeviceBuffer::from_slice(&first_valids).map_err(CudaMaaqError::Cuda)?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            fast_sc,
            slow_sc,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn maaq_multi_series_one_param_time_major_dev_ex(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
    ) -> Result<DeviceArrayF32Maaq, CudaMaaqError> {
        let (first_valids, period, fast_sc, slow_sc) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let prices_bytes = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaMaaqError::ArithmeticOverflow("prices_bytes"))?;
        let first_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(CudaMaaqError::ArithmeticOverflow("first_bytes"))?;
        let out_bytes = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaMaaqError::ArithmeticOverflow("out_bytes"))?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or(CudaMaaqError::ArithmeticOverflow("required_bytes"))?;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        let d_prices_tm = self.upload_f32_large(data_tm_f32)?;
        let d_first_valids =
            DeviceBuffer::from_slice(&first_valids).map_err(CudaMaaqError::Cuda)?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            fast_sc,
            slow_sc,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32Maaq {
            buf: d_out_tm,
            rows,
            cols,
            device_id: self.device_id,
            _ctx: Arc::clone(&self._context),
        })
    }

    pub fn maaq_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MaaqParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaMaaqError> {
        if out_tm.len() != cols * rows {
            return Err(CudaMaaqError::InvalidInput(format!(
                "output slice wrong length: got {} expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period, fast_sc, slow_sc) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = self.upload_f32_large(data_tm_f32)?;
        let d_first_valids =
            DeviceBuffer::from_slice(&first_valids).map_err(CudaMaaqError::Cuda)?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            fast_sc,
            slow_sc,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        self.stream.synchronize()?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(cols * rows)? };
        unsafe {
            d_out_tm.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

impl CudaMaaq {
    #[inline]
    fn upload_f32_large(&self, src: &[f32]) -> Result<DeviceBuffer<f32>, CudaMaaqError> {
        const PINNED_THRESH_BYTES: usize = 1 << 20;
        let n = src.len();
        if n * std::mem::size_of::<f32>() >= PINNED_THRESH_BYTES {
            let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(n) }?;
            pinned.as_mut_slice().copy_from_slice(src);

            let mut d = unsafe { DeviceBuffer::uninitialized_async(n, &self.stream) }?;
            unsafe {
                d.async_copy_from(pinned.as_slice(), &self.stream)?;
            }
            Ok(d)
        } else {
            unsafe { DeviceBuffer::from_slice_async(src, &self.stream) }.map_err(Into::into)
        }
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::maaq::{MaaqBatchRange, MaaqParams};

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

    struct MaaqBatchDevState {
        cuda: CudaMaaq,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_fast_scs: DeviceBuffer<f32>,
        d_slow_scs: DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MaaqBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .maaq_batch_device(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_fast_scs,
                    &self.d_slow_scs,
                    self.first_valid,
                    self.series_len,
                    self.n_combos,
                    self.max_period,
                    &mut self.d_out,
                )
                .expect("maaq batch kernel");
            self.cuda.stream.synchronize().expect("maaq sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMaaq::new(0).expect("cuda maaq");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = MaaqBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            fast_period: (2, 2, 0),
            slow_period: (30, 30, 0),
        };

        let (combos, first_valid, series_len, max_period) =
            CudaMaaq::prepare_batch_inputs(&price, &sweep).expect("maaq prepare batch inputs");
        let n_combos = combos.len();

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut fast_scs = Vec::with_capacity(n_combos);
        let mut slow_scs = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            let fast = prm.fast_period.unwrap();
            let slow = prm.slow_period.unwrap();
            periods_i32.push(period as i32);
            fast_scs.push(2.0f32 / (fast as f32 + 1.0f32));
            slow_scs.push(2.0f32 / (slow as f32 + 1.0f32));
        }

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_fast_scs = DeviceBuffer::from_slice(&fast_scs).expect("d_fast_scs");
        let d_slow_scs = DeviceBuffer::from_slice(&slow_scs).expect("d_slow_scs");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(MaaqBatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_fast_scs,
            d_slow_scs,
            first_valid,
            series_len,
            n_combos,
            max_period,
            d_out,
        })
    }

    struct MaaqManyDevState {
        cuda: CudaMaaq,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        period: usize,
        fast_sc: f32,
        slow_sc: f32,
        cols: usize,
        rows: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for MaaqManyDevState {
        fn launch(&mut self) {
            self.cuda
                .maaq_multi_series_one_param_device(
                    &self.d_prices_tm,
                    self.period as i32,
                    self.fast_sc,
                    self.slow_sc,
                    self.cols as i32,
                    self.rows as i32,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("maaq many-series kernel");
            self.cuda.stream.synchronize().expect("maaq sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaMaaq::new(0).expect("cuda maaq");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = MaaqParams {
            period: Some(64),
            fast_period: Some(2),
            slow_period: Some(30),
        };

        let (first_valids, period, fast_sc, slow_sc) =
            CudaMaaq::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("maaq prepare many-series inputs");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(MaaqManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            period,
            fast_sc,
            slow_sc,
            cols,
            rows,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "maaq",
                "one_series_many_params",
                "maaq_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "maaq",
                "many_series_one_param",
                "maaq_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
