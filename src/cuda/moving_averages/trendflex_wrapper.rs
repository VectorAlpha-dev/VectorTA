#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::trendflex::{
    expand_grid_trendflex_checked, TrendFlexBatchRange, TrendFlexParams,
};
use cust::context::CacheConfig;
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
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

#[derive(Debug, Error)]
pub enum CudaTrendflexError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("out of memory: required={required}B, free={free}B, headroom={headroom}B")]
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
    #[error("device mismatch: buffer on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaTrendflex {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaTrendflexPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
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
pub struct CudaTrendflexPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaTrendflexPolicy {
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

impl CudaTrendflex {
    pub fn new(device_id: usize) -> Result<Self, CudaTrendflexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/trendflex_kernel.ptx"));

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

        let stream_priority = std::env::var("CUDA_STREAM_PRIORITY")
            .ok()
            .and_then(|s| s.parse::<i32>().ok());
        let stream = Stream::new(StreamFlags::NON_BLOCKING, stream_priority)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaTrendflexPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaTrendflexPolicy,
    ) -> Result<Self, CudaTrendflexError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaTrendflexPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaTrendflexPolicy {
        &self.policy
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> {
        Arc::clone(&self._context)
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaTrendflexError> {
        self.stream.synchronize().map_err(Into::into)
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
                    eprintln!("[DEBUG] TrendFlex batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTrendflex)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] TrendFlex many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaTrendflex)).debug_many_logged = true;
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaTrendflexError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaTrendflexError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &TrendFlexBatchRange,
    ) -> Result<(Vec<TrendFlexParams>, usize, usize), CudaTrendflexError> {
        if data_f32.is_empty() {
            return Err(CudaTrendflexError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTrendflexError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_trendflex_checked(sweep)
            .map_err(|e| CudaTrendflexError::InvalidInput(e.to_string()))?;

        let len = data_f32.len();
        let tail_len = len - first_valid;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaTrendflexError::InvalidInput(
                    "period must be at least 1".into(),
                ));
            }
            if period >= len {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            let ss_period = ((period as f64) / 2.0).round() as usize;
            if tail_len < period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "not enough valid data for period {} (valid tail = {})",
                    period, tail_len
                )));
            }
            if tail_len < ss_period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "not enough valid data for smoother period {} (valid tail = {})",
                    ss_period, tail_len
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
        max_period: usize,
    ) -> Result<(), CudaTrendflexError> {
        let mut func = self
            .module
            .get_function("trendflex_batch_f32")
            .map_err(|_| CudaTrendflexError::MissingKernelSymbol {
                name: "trendflex_batch_f32",
            })?;

        func.set_cache_config(CacheConfig::PreferL1)?;

        if max_period == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "max_period must be positive".into(),
            ));
        }
        if max_period > i32::MAX as usize {
            return Err(CudaTrendflexError::InvalidInput(
                "max_period exceeds i32::MAX".into(),
            ));
        }
        let mut max_period_i = max_period as i32;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
            BatchKernelPolicy::Auto => 32,
        };
        let shared_bytes = (block_x as usize)
            .checked_mul(max_period)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaTrendflexError::InvalidInput("shared mem size overflow".into()))?
            as u32;

        let max_blocks: u32 = 65_535;
        let chunk_cap: usize = (max_blocks as usize) * (block_x as usize);
        let mut launched = 0usize;
        while launched < n_combos {
            let chunk = (n_combos - launched).min(chunk_cap);
            let grid_x = ((chunk as u32) + block_x - 1) / block_x;
            if block_x > 1024 || grid_x == 0 || grid_x > 65_535 {
                return Err(CudaTrendflexError::LaunchConfigTooLarge {
                    gx: grid_x,
                    gy: 1,
                    gz: 1,
                    bx: block_x,
                    by: 1,
                    bz: 1,
                });
            }
            let grid: GridSize = (grid_x.max(1), 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(launched).as_raw();
                let mut len_i = series_len as i32;
                let mut combos_i = chunk as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(launched * series_len).as_raw();

                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];

                self.stream.launch(&func, grid, block, shared_bytes, args)?;
            }
            launched += chunk;
        }

        unsafe {
            let this = self as *const _ as *mut CudaTrendflex;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[TrendFlexParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        let n_combos = combos.len();
        let prices_bytes = len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;

        let elems = combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| CudaTrendflexError::InvalidInput("size overflow".into()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            len,
            combos.len(),
            first_valid,
            &mut d_out,
            max_period,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn trendflex_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &TrendFlexBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<TrendFlexParams>), CudaTrendflexError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn trendflex_batch_dev_with_device_prices_into(
        &self,
        d_prices: &DeviceBuffer<f32>,
        len: usize,
        combos: &[TrendFlexParams],
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrendflexError> {
        if combos.is_empty() {
            return Err(CudaTrendflexError::InvalidInput("no combos".into()));
        }

        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        self.launch_batch_kernel(
            d_prices,
            &d_periods,
            len,
            combos.len(),
            first_valid,
            d_out,
            max_period,
        )
    }

    pub fn trendflex_batch_on_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        len: usize,
        combos: &[TrendFlexParams],
        first_valid: usize,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        if combos.is_empty() {
            return Err(CudaTrendflexError::InvalidInput("no combos".into()));
        }
        let elems = combos.len() * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(CudaTrendflexError::Cuda)?;

        self.trendflex_batch_dev_with_device_prices_into(
            d_prices,
            len,
            combos,
            first_valid,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn trendflex_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &TrendFlexBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TrendFlexParams>), CudaTrendflexError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "output slice length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;

        self.stream.synchronize()?;
        dev.buf.copy_to(out)?;
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrendFlexParams,
    ) -> Result<(Vec<i32>, usize), CudaTrendflexError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }
        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period >= rows {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        let ss_period = ((period as f64) / 2.0).round() as usize;
        if ss_period == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "smoother period must be positive".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let val = data_tm_f32[idx];
                if !val.is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaTrendflexError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            let tail = rows - fv;
            if tail < period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    series, period, tail
                )));
            }
            if tail < ss_period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "series {} insufficient data for smoother {} (tail = {})",
                    series, ss_period, tail
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_ssf: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrendflexError> {
        let mut func = self
            .module
            .get_function("trendflex_many_series_one_param_f32")
            .map_err(|_| CudaTrendflexError::MissingKernelSymbol {
                name: "trendflex_many_series_one_param_f32",
            })?;
        func.set_cache_config(CacheConfig::PreferL1)?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => {
                let (_min_grid, block) =
                    func.suggested_launch_configuration(0, (0, 0, 0).into())?;
                if block == 0 {
                    128
                } else {
                    block
                }
            }
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        if block_x > 1024 || grid_x == 0 || grid_x > 65_535 {
            return Err(CudaTrendflexError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut ssf_ptr = d_ssf.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut ssf_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        unsafe {
            let this = self as *const _ as *mut CudaTrendflex;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let firsts_bytes = cols * std::mem::size_of::<i32>();
        let scratch_bytes = cols * rows * std::mem::size_of::<f32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + firsts_bytes + scratch_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)?;
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTrendflexError::InvalidInput("size overflow".into()))?;
        let mut d_ssf = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            cols,
            rows,
            period,
            &mut d_ssf,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn trendflex_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrendFlexParams,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn trendflex_many_series_one_param_on_device_into(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        d_ssf_tm: &mut DeviceBuffer<f32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrendflexError> {
        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            d_ssf_tm,
            d_out_tm,
        )
    }

    pub fn trendflex_many_series_one_param_on_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        first_valids_host: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        let d_first_valids = DeviceBuffer::from_slice(first_valids_host)?;
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaTrendflexError::InvalidInput("size overflow".into()))?;
        let mut d_ssf = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;
        self.trendflex_many_series_one_param_on_device_into(
            d_prices_tm,
            cols,
            rows,
            &d_first_valids,
            period,
            &mut d_ssf,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn trendflex_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrendFlexParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTrendflexError> {
        if out_tm.len()
            != cols
                .checked_mul(rows)
                .ok_or_else(|| CudaTrendflexError::InvalidInput("size overflow".into()))?
        {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "output slice mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        self.stream.synchronize()?;
        dev.buf.copy_to(out_tm)?;
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::trendflex::{TrendFlexBatchRange, TrendFlexParams};

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

    struct TrendflexBatchDevState {
        cuda: CudaTrendflex,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TrendflexBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    &mut self.d_out,
                    self.max_period,
                )
                .expect("trendflex batch kernel");
            self.cuda.stream.synchronize().expect("trendflex sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaTrendflex::new(0).expect("cuda trendflex");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = TrendFlexBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (combos, first_valid, series_len) =
            CudaTrendflex::prepare_batch_inputs(&price, &sweep).expect("trendflex prepare batch");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(TrendflexBatchDevState {
            cuda,
            d_prices,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            max_period,
            d_out,
        })
    }

    struct TrendflexManyDevState {
        cuda: CudaTrendflex,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_ssf: DeviceBuffer<f32>,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for TrendflexManyDevState {
        fn launch(&mut self) {
            self.cuda
                .trendflex_many_series_one_param_on_device_into(
                    &self.d_prices_tm,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    self.period,
                    &mut self.d_ssf,
                    &mut self.d_out_tm,
                )
                .expect("trendflex many-series kernel");
            self.cuda.stream.synchronize().expect("trendflex sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaTrendflex::new(0).expect("cuda trendflex");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = TrendFlexParams { period: Some(64) };
        let (first_valids, period) =
            CudaTrendflex::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("trendflex prepare many-series");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let elems = cols * rows;
        let d_ssf: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_ssf");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(elems) }.expect("d_out_tm");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(TrendflexManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            d_ssf,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "trendflex",
                "one_series_many_params",
                "trendflex_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "trendflex",
                "many_series_one_param",
                "trendflex_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
