#![cfg(feature = "cuda")]

use crate::indicators::moving_averages::kama::{KamaBatchRange, KamaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaKamaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Out of memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Device mismatch: buffer device={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

pub struct DeviceArrayF32Kama {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Kama {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaKama {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaKamaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaKama {
    pub fn new(device_id: usize) -> Result<Self, CudaKamaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/kama_kernel.ptx"));

        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
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
            _context: context,
            device_id: device_id as u32,
            policy: CudaKamaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaKamaError> {
        self.stream.synchronize().map_err(Into::into)
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _total)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn has_function(&self, name: &str) -> bool {
        self.module.get_function(name).is_ok()
    }

    #[inline]
    fn grid_chunks(total: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535;
        (0..total).step_by(MAX).map(move |start| {
            let len = (total - start).min(MAX);
            (start, len)
        })
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
                    eprintln!("[DEBUG] KAMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaKama)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] KAMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaKama)).debug_many_logged = true;
                }
            }
        }
    }

    fn expand_periods(range: &KamaBatchRange) -> Vec<KamaParams> {
        let (start, end, step) = range.period;
        let periods: Vec<usize> = if step == 0 || start == end {
            vec![start]
        } else if start < end {
            (start..=end).step_by(step).collect::<Vec<_>>()
        } else {
            let mut v = Vec::new();
            let mut cur = start;
            loop {
                v.push(cur);
                match cur.checked_sub(step) {
                    Some(next) if next >= end => cur = next,
                    _ => break,
                }
            }
            v
        };
        periods
            .into_iter()
            .map(|p| KamaParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &KamaBatchRange,
    ) -> Result<(Vec<KamaParams>, usize, usize, usize), CudaKamaError> {
        if data_f32.is_empty() {
            return Err(CudaKamaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaKamaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaKamaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaKamaError::InvalidInput("period must be >= 1".into()));
            }
            if period > series_len {
                return Err(CudaKamaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid <= period {
                return Err(CudaKamaError::InvalidInput(format!(
                    "not enough valid data: need > {}, valid = {}",
                    period, valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, series_len, max_period))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKamaError> {
        let func = self.module.get_function("kama_batch_f32").map_err(|_| {
            CudaKamaError::MissingKernelSymbol {
                name: "kama_batch_f32",
            }
        })?;

        unsafe {
            let this = self as *const _ as *mut CudaKama;
            (*this).last_batch = Some(BatchKernelSelected::OneD { block_x: 32 });
        }
        self.maybe_log_batch_debug();

        const BLOCK_X: u32 = 32;
        for (start, len) in Self::grid_chunks(n_combos) {
            let grid: GridSize = (len as u32, 1, 1).into();
            let block: BlockSize = (BLOCK_X, 1, 1).into();

            let d_periods_off = unsafe { d_periods.as_device_ptr().add(start) };
            let d_out_off = unsafe { d_out.as_device_ptr().add(start * series_len) };

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods_off.as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out_off.as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
        }
        Ok(())
    }

    fn launch_batch_kernel_with_prefix(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_prefix_roc1: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKamaError> {
        let func = self
            .module
            .get_function("kama_batch_prefix_f32")
            .map_err(|_| CudaKamaError::MissingKernelSymbol {
                name: "kama_batch_prefix_f32",
            })?;

        unsafe {
            let this = self as *const _ as *mut CudaKama;
            (*this).last_batch = Some(BatchKernelSelected::OneD { block_x: 32 });
        }
        self.maybe_log_batch_debug();

        const BLOCK_X: u32 = 32;
        for (start, len) in Self::grid_chunks(n_combos) {
            let grid: GridSize = (len as u32, 1, 1).into();
            let block: BlockSize = (BLOCK_X, 1, 1).into();

            let d_periods_off = unsafe { d_periods.as_device_ptr().add(start) };
            let d_out_off = unsafe { d_out.as_device_ptr().add(start * series_len) };

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut prefix_ptr = d_prefix_roc1.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods_off.as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out_off.as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut prefix_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn build_roc1_prefix_bytes(series_len: usize) -> usize {
        (series_len + 1) * std::mem::size_of::<f32>()
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[KamaParams],
        first_valid: usize,
        series_len: usize,
        _max_period: usize,
    ) -> Result<DeviceArrayF32Kama, CudaKamaError> {
        let n_combos = combos.len();
        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaKamaError::InvalidInput("series_len byte size overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaKamaError::InvalidInput("periods byte size overflow".into()))?;
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaKamaError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaKamaError::InvalidInput("out byte size overflow".into()))?;

        let have_prefix_kernel = self.has_function("kama_batch_prefix_f32");
        let use_prefix = have_prefix_kernel;

        let prefix_bytes = if use_prefix {
            Self::build_roc1_prefix_bytes(series_len)
        } else {
            0
        };
        let required = prices_bytes
            .checked_add(periods_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .and_then(|x| x.checked_add(prefix_bytes))
            .ok_or_else(|| CudaKamaError::InvalidInput("aggregate byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let (free, _total) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaKamaError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(out_elems) }?;

        if use_prefix {
            let mut prefix: Vec<f32> = Vec::with_capacity(series_len + 1);
            prefix.push(0.0f32);
            let mut acc = 0.0f32;
            let mut prev = data_f32.get(0).copied().unwrap_or(f32::NAN);
            for i in 1..series_len {
                let cur = data_f32[i];
                let diff = if prev.is_nan() || cur.is_nan() {
                    0.0f32
                } else {
                    (cur - prev).abs()
                };
                acc += diff;
                prefix.push(acc);
                prev = cur;
            }

            prefix.push(acc);
            let d_prefix = DeviceBuffer::from_slice(&prefix)?;
            self.launch_batch_kernel_with_prefix(
                &d_prices,
                &d_prefix,
                &d_periods,
                series_len,
                n_combos,
                first_valid,
                &mut d_out,
            )?;
        } else {
            self.launch_batch_kernel(
                &d_prices,
                &d_periods,
                series_len,
                n_combos,
                first_valid,
                &mut d_out,
            )?;
        }

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Kama {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn kama_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &KamaBatchRange,
    ) -> Result<DeviceArrayF32Kama, CudaKamaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)
    }

    pub fn kama_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &KamaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<KamaParams>), CudaKamaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaKamaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)?;
        arr.buf.copy_to(out)?;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn kama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        _max_period: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKamaError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaKamaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &KamaParams,
    ) -> Result<(Vec<i32>, usize), CudaKamaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaKamaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaKamaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaKamaError::InvalidInput("period must be >= 1".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(t as i32);
                    break;
                }
            }
            let fv = found
                .ok_or_else(|| CudaKamaError::InvalidInput(format!("series {} all NaN", series)))?;
            let valid = rows as i32 - fv;
            if valid <= period as i32 {
                return Err(CudaKamaError::InvalidInput(format!(
                    "series {} lacks data: need > {}, valid = {}",
                    series, period, valid
                )));
            }
            first_valids[series] = fv;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKamaError> {
        let block_x = std::env::var("KAMA_BLOCK_X")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(32);
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let func = self
            .module
            .get_function("kama_many_series_one_param_time_major_f32")
            .map_err(|_| CudaKamaError::MissingKernelSymbol {
                name: "kama_many_series_one_param_time_major_f32",
            })?;

        unsafe {
            let this = self as *const _ as *mut CudaKama;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
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
    ) -> Result<DeviceArrayF32Kama, CudaKamaError> {
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaKamaError::InvalidInput("cols*rows overflow".into()))?;
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaKamaError::InvalidInput("prices byte size overflow".into()))?;
        let first_valid_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaKamaError::InvalidInput("first_valids byte size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaKamaError::InvalidInput("out byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(first_valid_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaKamaError::InvalidInput("aggregate byte size overflow".into()))?;
        let headroom = 16 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let (free, _total) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaKamaError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        self.launch_many_series_kernel(&d_prices, period, cols, rows, &d_first_valids, &mut d_out)?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Kama {
            buf: d_out,
            rows,
            cols,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn kama_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &KamaParams,
    ) -> Result<DeviceArrayF32Kama, CudaKamaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn kama_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &KamaParams,
        out: &mut [f32],
    ) -> Result<(), CudaKamaError> {
        if out.len() != cols * rows {
            return Err(CudaKamaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                cols * rows
            )));
        }
        let arr =
            self.kama_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        arr.buf.copy_to(out).map_err(Into::into)
    }

    pub fn kama_many_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaKamaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaKamaError::InvalidInput(
                "period and dimensions must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out,
        )
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::kama::KamaParams;

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

    struct KamaBatchDevState {
        cuda: CudaKama,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for KamaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    self.len,
                    self.rows,
                    self.first_valid,
                    &mut self.d_out,
                )
                .expect("kama batch kernel");
            self.cuda.stream.synchronize().expect("kama sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaKama::new(0).expect("cuda kama");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = KamaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (combos, first_valid, series_len, _max_period) =
            CudaKama::prepare_batch_inputs(&price, &sweep).expect("kama prepare batch inputs");
        let rows = combos.len();
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * rows) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(KamaBatchDevState {
            cuda,
            d_prices,
            d_periods,
            len: series_len,
            first_valid,
            rows,
            d_out,
        })
    }

    struct KamaManyDevState {
        cuda: CudaKama,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for KamaManyDevState {
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
                .expect("kama many-series kernel");
            self.cuda.stream.synchronize().expect("kama sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaKama::new(0).expect("cuda kama");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = KamaParams { period: Some(64) };
        let (first_valids, period) =
            CudaKama::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("kama prepare many-series inputs");
        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(KamaManyDevState {
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
                "kama",
                "one_series_many_params",
                "kama_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "kama",
                "many_series_one_param",
                "kama_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,

    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaKamaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
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

impl CudaKama {
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaKamaPolicy,
    ) -> Result<Self, CudaKamaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaKamaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaKamaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
}
