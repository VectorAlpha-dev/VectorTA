#![cfg(feature = "cuda")]

use super::DeviceArrayF32;

use super::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::indicators::moving_averages::reflex::{ReflexBatchRange, ReflexParams};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub enum CudaReflexError {
    #[allow(dead_code)]
    Cuda(cust::error::CudaError),
    InvalidInput(String),
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    MissingKernelSymbol {
        name: &'static str,
    },
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[allow(dead_code)]
    InvalidPolicy(&'static str),
    #[allow(dead_code)]
    DeviceMismatch {
        buf: u32,
        current: u32,
    },
    NotImplemented,
}

impl fmt::Display for CudaReflexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaReflexError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaReflexError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
            CudaReflexError::OutOfMemory {
                required,
                free,
                headroom,
            } => write!(
                f,
                "insufficient device memory: required={}B (including headroom={}B), free={}B",
                required, headroom, free
            ),
            CudaReflexError::MissingKernelSymbol { name } => {
                write!(f, "missing kernel symbol: {}", name)
            }
            CudaReflexError::LaunchConfigTooLarge {
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
            } => write!(
                f,
                "launch config too large (grid=({}, {}, {}), block=({}, {}, {}))",
                gx, gy, gz, bx, by, bz
            ),
            CudaReflexError::InvalidPolicy(p) => write!(f, "invalid policy: {}", p),
            CudaReflexError::DeviceMismatch { buf, current } => write!(
                f,
                "device mismatch: buffer on device {} but current device {}",
                buf, current
            ),
            CudaReflexError::NotImplemented => write!(f, "not implemented"),
        }
    }
}

impl std::error::Error for CudaReflexError {}

impl From<cust::error::CudaError> for CudaReflexError {
    fn from(e: cust::error::CudaError) -> Self {
        CudaReflexError::Cuda(e)
    }
}

pub struct CudaReflex {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,

    policy: CudaReflexPolicy,
    last_batch: Option<ReflexBatchKernelSelected>,
    last_many: Option<ReflexManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    max_grid_x: u32,
}

impl CudaReflex {
    pub fn new(device_id: usize) -> Result<Self, CudaReflexError> {
        cust::init(CudaFlags::empty())?;

        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/reflex_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        let prio = std::env::var("CUDA_STREAM_PRIORITY")
            .ok()
            .and_then(|v| v.parse::<i32>().ok());
        let stream = Stream::new(StreamFlags::NON_BLOCKING, prio)?;

        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(65_535) as u32;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaReflexPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            max_grid_x,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaReflexPolicy,
    ) -> Result<Self, CudaReflexError> {
        let mut me = Self::new(device_id)?;
        me.policy = policy;
        Ok(me)
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaReflexPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaReflexPolicy {
        &self.policy
    }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<ReflexBatchKernelSelected> {
        self.last_batch
    }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ReflexManySeriesKernelSelected> {
        self.last_many
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaReflexError> {
        self.stream.synchronize().map_err(CudaReflexError::from)
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self._context.clone()
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
    fn will_fit_or_err(
        required_bytes: usize,
        headroom_bytes: usize,
    ) -> Result<(), CudaReflexError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            let need = required_bytes.saturating_add(headroom_bytes);
            if need <= free {
                Ok(())
            } else {
                Err(CudaReflexError::OutOfMemory {
                    required: need,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    pub fn reflex_batch_dev(
        &self,
        prices: &[f32],
        sweep: &ReflexBatchRange,
    ) -> Result<DeviceArrayF32, CudaReflexError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn reflex_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &ReflexBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<ReflexParams>), CudaReflexError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaReflexError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;
        arr.buf.copy_to(out).map_err(CudaReflexError::Cuda)?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    pub fn reflex_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaReflexError> {
        let cur_dev = unsafe {
            let mut dev: i32 = 0;
            let _ = cu::cuCtxGetDevice(&mut dev as *mut _);
            dev as u32
        };
        if cur_dev != self.device_id {
            return Err(CudaReflexError::DeviceMismatch {
                buf: self.device_id,
                current: cur_dev,
            });
        }

        if series_len == 0 || n_combos == 0 {
            return Err(CudaReflexError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaReflexError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        if max_period < 2 {
            return Err(CudaReflexError::InvalidInput(
                "max_period must be >= 2".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            max_period,
            d_out,
        )
    }

    pub fn reflex_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: Option<&DeviceBuffer<i32>>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaReflexError> {
        let cur_dev = unsafe {
            let mut dev: i32 = 0;
            let _ = cu::cuCtxGetDevice(&mut dev as *mut _);
            dev as u32
        };
        if cur_dev != self.device_id {
            return Err(CudaReflexError::DeviceMismatch {
                buf: self.device_id,
                current: cur_dev,
            });
        }

        if period < 2 || num_series == 0 || series_len == 0 {
            return Err(CudaReflexError::InvalidInput(
                "period >= 2 and positive dimensions required".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaReflexError::InvalidInput(
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

    pub fn reflex_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaReflexError> {
        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)
    }

    pub fn reflex_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        out_tm: &mut [f32],
    ) -> Result<(), CudaReflexError> {
        if out_tm.len()
            != cols
                .checked_mul(rows)
                .ok_or_else(|| CudaReflexError::InvalidInput("rows*cols overflow".into()))?
        {
            return Err(CudaReflexError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }

        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, period)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, cols, rows, period, &prepared)?;
        arr.buf.copy_to(out_tm).map_err(CudaReflexError::Cuda)?;
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaReflexError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        let prices_bytes = series_len.saturating_mul(std::mem::size_of::<f32>());
        let periods_bytes = n_combos.saturating_mul(std::mem::size_of::<i32>());
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaReflexError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems.saturating_mul(std::mem::size_of::<f32>());
        let required = prices_bytes + periods_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;

        Self::will_fit_or_err(required, headroom)?;

        let d_prices = self.htod_copy_f32(prices)?;
        let d_periods = DeviceBuffer::from_slice(&inputs.periods).map_err(CudaReflexError::from)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(series_len * n_combos, &self.stream) }
                .map_err(CudaReflexError::from)?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            series_len,
            n_combos,
            inputs.first_valid,
            inputs.max_period,
            &mut d_out,
        )?;

        self.stream.synchronize().map_err(CudaReflexError::from)?;

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
    ) -> Result<DeviceArrayF32, CudaReflexError> {
        let prices_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let out_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024;

        if !Self::will_fit(required, headroom) {
            return Err(CudaReflexError::InvalidInput(
                "insufficient device memory for Reflex many-series launch".into(),
            ));
        }

        let d_prices_tm = self.htod_copy_f32(prices_tm_f32)?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prices_tm_f32.len(), &self.stream) }
                .map_err(CudaReflexError::from)?;

        self.launch_many_series_kernel(&d_prices_tm, period, cols, rows, None, &mut d_out_tm)?;

        self.stream.synchronize().map_err(CudaReflexError::from)?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaReflexError> {
        let func = match self.module.get_function("reflex_batch_f32") {
            Ok(f) => f,
            Err(_) => self
                .module
                .get_function("reflex_batch_f32_precomp")
                .map_err(|_| CudaReflexError::MissingKernelSymbol {
                    name: "reflex_batch_f32_precomp",
                })?,
        };

        unsafe {
            let this = self as *const _ as *mut CudaReflex;
            (*this).last_batch = Some(ReflexBatchKernelSelected::Plain1D { block_x: 1 });
        }
        self.maybe_log_batch_debug();

        const MAX_CHUNK: usize = 65_535;
        let block: BlockSize = (1, 1, 1).into();

        let shared_bytes_u64 =
            (max_period as u64 + 1).saturating_mul(std::mem::size_of::<f64>() as u64);
        if shared_bytes_u64 > (u32::MAX as u64) {
            return Err(CudaReflexError::InvalidInput(
                "dynamic shared memory size exceeds u32".into(),
            ));
        }
        let shared_bytes = shared_bytes_u64 as u32;
        let mut start = 0usize;
        while start < n_combos {
            let len = (n_combos - start).min(MAX_CHUNK);
            let grid_x = len as u32;
            if grid_x > self.max_grid_x {
                return Err(CudaReflexError::LaunchConfigTooLarge {
                    gx: grid_x,
                    gy: 1,
                    gz: 1,
                    bx: 1,
                    by: 1,
                    bz: 1,
                });
            }
            let grid: GridSize = (grid_x, 1, 1).into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();

                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(CudaReflexError::from)?;
            }
            start += len;
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: Option<&DeviceBuffer<i32>>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaReflexError> {
        let func = self
            .module
            .get_function("reflex_many_series_one_param_f32")
            .map_err(|_| CudaReflexError::MissingKernelSymbol {
                name: "reflex_many_series_one_param_f32",
            })?;

        unsafe {
            let this = self as *const _ as *mut CudaReflex;
            (*this).last_many = Some(ReflexManySeriesKernelSelected::OneD { block_x: 1 });
        }
        self.maybe_log_many_debug();

        let grid_x = num_series as u32;
        if grid_x > self.max_grid_x {
            return Err(CudaReflexError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: 1,
                by: 1,
                bz: 1,
            });
        }
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        let shared_bytes_u64 =
            (period as u64 + 1).saturating_mul(std::mem::size_of::<f64>() as u64);
        if shared_bytes_u64 > (u32::MAX as u64) {
            return Err(CudaReflexError::InvalidInput(
                "dynamic shared memory size exceeds u32".into(),
            ));
        }
        let shared_bytes = shared_bytes_u64 as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids
                .map(|b| b.as_device_ptr().as_raw())
                .unwrap_or(0u64);
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
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(CudaReflexError::from)?
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &ReflexBatchRange,
    ) -> Result<BatchInputs, CudaReflexError> {
        if prices.is_empty() {
            return Err(CudaReflexError::InvalidInput("empty prices".into()));
        }
        let combos = expand_grid_reflex_checked(sweep)?;

        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaReflexError::InvalidInput("all values are NaN".into()))?;

        let series_len = prices.len();
        let mut periods = Vec::with_capacity(combos.len());
        let mut max_period = 0usize;
        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaReflexError::InvalidInput("period must be >= 2".into()));
            }
            if period > i32::MAX as usize {
                return Err(CudaReflexError::InvalidInput(
                    "period exceeds i32 kernel limit".into(),
                ));
            }
            periods.push(period as i32);
            max_period = max_period.max(period);
        }

        if series_len - first_valid < max_period {
            return Err(CudaReflexError::InvalidInput(format!(
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
            max_period,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<ManySeriesInputs, CudaReflexError> {
        if cols == 0 || rows == 0 {
            return Err(CudaReflexError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if prices_tm_f32.len()
            != cols
                .checked_mul(rows)
                .ok_or_else(|| CudaReflexError::InvalidInput("rows*cols overflow".into()))?
        {
            return Err(CudaReflexError::InvalidInput(
                "matrix shape mismatch".into(),
            ));
        }
        if period < 2 {
            return Err(CudaReflexError::InvalidInput("period must be >= 2".into()));
        }
        if period > i32::MAX as usize {
            return Err(CudaReflexError::InvalidInput(
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
                CudaReflexError::InvalidInput(format!("series {} has all NaN values", series_idx))
            })?;
            if rows - first < period {
                return Err(CudaReflexError::InvalidInput(format!(
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

impl CudaReflex {
    #[inline]
    fn htod_copy_f32(&self, src: &[f32]) -> Result<DeviceBuffer<f32>, CudaReflexError> {
        match LockedBuffer::from_slice(src) {
            Ok(h_pinned) => unsafe {
                let mut dst = DeviceBuffer::uninitialized_async(src.len(), &self.stream)
                    .map_err(CudaReflexError::from)?;
                dst.async_copy_from(&h_pinned, &self.stream)
                    .map_err(CudaReflexError::from)?;
                Ok(dst)
            },
            Err(_) => DeviceBuffer::from_slice(src).map_err(CudaReflexError::from),
        }
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let periods_bytes = PARAM_SWEEP * std::mem::size_of::<i32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + periods_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct ReflexBatchDeviceState {
        cuda: CudaReflex,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
    }
    impl CudaBenchState for ReflexBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .reflex_batch_device(
                    &self.d_prices,
                    &self.d_periods,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    self.max_period,
                    &mut self.d_out,
                )
                .expect("reflex_batch_device");
            self.cuda.synchronize().expect("sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaReflex::new(0).expect("cuda reflex");
        let price = gen_series(ONE_SERIES_LEN);
        let first_valid = price.iter().position(|x| !x.is_nan()).unwrap_or(0);

        let start = 10i32;
        let mut periods = Vec::with_capacity(PARAM_SWEEP);
        for i in 0..PARAM_SWEEP {
            periods.push(start + i as i32);
        }
        let max_period = (start as usize) + PARAM_SWEEP - 1;

        let d_prices = DeviceBuffer::from_slice(&price).expect("upload prices");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("upload periods");
        let out_elems = ONE_SERIES_LEN * PARAM_SWEEP;
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(out_elems) }.expect("alloc out");

        Box::new(ReflexBatchDeviceState {
            cuda,
            d_prices,
            d_periods,
            d_out,
            series_len: ONE_SERIES_LEN,
            n_combos: PARAM_SWEEP,
            first_valid,
            max_period,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "reflex",
            "one_series_many_params",
            "reflex_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())]
    }
}

fn expand_grid_reflex_checked(
    range: &ReflexBatchRange,
) -> Result<Vec<ReflexParams>, CudaReflexError> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return Ok(vec![ReflexParams {
            period: Some(start),
        }]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut cur = start;
        while cur <= end {
            out.push(ReflexParams { period: Some(cur) });
            cur = match cur.checked_add(step) {
                Some(v) => v,
                None => break,
            };
        }
    } else {
        let mut cur = start;
        while cur >= end {
            out.push(ReflexParams { period: Some(cur) });
            cur = match cur.checked_sub(step) {
                Some(v) => v,
                None => break,
            };
            if cur == 0 {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(CudaReflexError::InvalidInput("invalid period range".into()));
    }
    Ok(out)
}

struct BatchInputs {
    combos: Vec<ReflexParams>,
    periods: Vec<i32>,
    first_valid: usize,
    series_len: usize,
    max_period: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaReflexPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaReflexPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ReflexBatchKernelSelected {
    Plain1D { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ReflexManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[inline]
fn maybe_log_once(flag: &mut bool, msg: String) {
    static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
    if *flag {
        return;
    }
    if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
        let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
        if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
            eprintln!("{}", msg);
        }
        *flag = true;
    }
}

impl CudaReflex {
    #[inline]
    fn maybe_log_batch_debug(&self) {
        if let Some(sel) = self.last_batch {
            let msg = format!("[DEBUG] Reflex batch selected kernel: {:?}", sel);

            unsafe {
                maybe_log_once(
                    &mut (*(self as *const _ as *mut CudaReflex)).debug_batch_logged,
                    msg,
                );
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        if let Some(sel) = self.last_many {
            let msg = format!("[DEBUG] Reflex many-series selected kernel: {:?}", sel);
            unsafe {
                maybe_log_once(
                    &mut (*(self as *const _ as *mut CudaReflex)).debug_many_logged,
                    msg,
                );
            }
        }
    }
}
