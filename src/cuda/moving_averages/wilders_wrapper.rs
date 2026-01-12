#![cfg(feature = "cuda")]

use crate::indicators::moving_averages::wilders::{WildersBatchRange, WildersParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::ffi::c_void;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaWildersError {
    #[error("CUDA: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error(
        "Out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes"
    )]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("Device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Not implemented")]
    NotImplemented,
}

pub struct CudaWilders {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaWildersPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,

    param_cache: RefCell<Option<ParamCache>>,
}

struct ParamCache {
    hash: u64,
    periods: DeviceBuffer<i32>,
    alphas: DeviceBuffer<f32>,
    warm: DeviceBuffer<i32>,
}

struct PreparedWildersBatch {
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    alphas_f32: Vec<f32>,
    warm_indices: Vec<i32>,
}

pub struct DeviceArrayF32Wilders {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Wilders {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

impl CudaWilders {
    pub fn new(device_id: usize) -> Result<Self, CudaWildersError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/wilders_kernel.ptx"));
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
            policy: CudaWildersPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            param_cache: RefCell::new(None),
        })
    }

    pub fn wilders_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &WildersBatchRange,
    ) -> Result<DeviceArrayF32Wilders, CudaWildersError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.periods_i32.len();

        let prices_bytes = prepared
            .series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let params_bytes_periods = prepared
            .periods_i32
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let params_bytes_alphas = prepared
            .alphas_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let params_bytes_warm = prepared
            .warm_indices
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let params_bytes = params_bytes_periods
            .checked_add(params_bytes_alphas)
            .and_then(|x| x.checked_add(params_bytes_warm))
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let out_elems = prepared
            .series_len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            let (free, _tot) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaWildersError::OutOfMemory {
                required,
                free,
                headroom,
            });
        }

        let d_prices = DeviceBuffer::from_slice(data_f32).map_err(CudaWildersError::from)?;

        let mut hasher = DefaultHasher::new();
        prepared.periods_i32.hash(&mut hasher);
        for &a in &prepared.alphas_f32 {
            a.to_bits().hash(&mut hasher);
        }
        prepared.warm_indices.hash(&mut hasher);
        let params_hash = hasher.finish();

        let mut cache_guard = self.param_cache.borrow_mut();
        let cache_hit = match cache_guard.as_ref() {
            Some(cache) => {
                cache.hash == params_hash
                    && cache.periods.len() == prepared.periods_i32.len()
                    && cache.alphas.len() == prepared.alphas_f32.len()
                    && cache.warm.len() == prepared.warm_indices.len()
            }
            None => false,
        };
        if !cache_hit {
            let periods =
                DeviceBuffer::from_slice(&prepared.periods_i32).map_err(CudaWildersError::from)?;
            let alphas =
                DeviceBuffer::from_slice(&prepared.alphas_f32).map_err(CudaWildersError::from)?;
            let warm =
                DeviceBuffer::from_slice(&prepared.warm_indices).map_err(CudaWildersError::from)?;
            *cache_guard = Some(ParamCache {
                hash: params_hash,
                periods,
                alphas,
                warm,
            });
        }
        let cache = cache_guard.as_ref().ok_or_else(|| {
            CudaWildersError::InvalidInput("failed to populate param cache".into())
        })?;
        let total = prepared
            .series_len
            .checked_mul(n_combos)
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total)? };

        self.launch_batch_kernel(
            &d_prices,
            &cache.periods,
            &cache.alphas,
            &cache.warm,
            prepared.series_len,
            prepared.first_valid,
            n_combos,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Wilders {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
            ctx: Arc::clone(&self._context),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wilders_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: i32,
        first_valid: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        if series_len <= 0 {
            return Err(CudaWildersError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if first_valid < 0 || first_valid >= series_len {
            return Err(CudaWildersError::InvalidInput(format!(
                "first_valid out of range: {} (len {})",
                first_valid, series_len
            )));
        }
        if n_combos <= 0 {
            return Err(CudaWildersError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        let expected = n_combos as usize;
        if d_periods.len() != expected || d_alphas.len() != expected || d_warm.len() != expected {
            return Err(CudaWildersError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_alphas,
            d_warm,
            series_len as usize,
            first_valid as usize,
            expected,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WildersParams,
    ) -> Result<(Vec<i32>, i32, f32), CudaWildersError> {
        if cols == 0 || rows == 0 {
            return Err(CudaWildersError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaWildersError::InvalidInput(format!(
                "time-major slice length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let period = params.period.unwrap_or(0) as i32;
        if period <= 0 {
            return Err(CudaWildersError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        if period as usize > rows {
            return Err(CudaWildersError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if v.is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaWildersError::InvalidInput(format!("series {} contains only NaNs", s))
            })?;
            let remain = rows - fv as usize;
            if remain < period as usize {
                return Err(CudaWildersError::InvalidInput(format!(
                    "series {} lacks enough valid data: need {}, have {}",
                    s, period, remain
                )));
            }
            first_valids[s] = fv;
        }

        let alpha = 1.0f32 / (period as f32);
        Ok((first_valids, period, alpha))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &WildersBatchRange,
    ) -> Result<PreparedWildersBatch, CudaWildersError> {
        if data_f32.is_empty() {
            return Err(CudaWildersError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaWildersError::InvalidInput(
                "no period combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaWildersError::InvalidInput("all values are NaN".into()))?;

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut alphas_f32 = Vec::with_capacity(combos.len());
        let mut warm_indices = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaWildersError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            if series_len - first_valid < period {
                return Err(CudaWildersError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, have {}",
                    period,
                    series_len - first_valid
                )));
            }
            for idx in 0..period {
                let sample = data_f32[first_valid + idx];
                if !sample.is_finite() {
                    return Err(CudaWildersError::InvalidInput(format!(
                        "non-finite value in warm window at offset {}",
                        idx
                    )));
                }
            }
            periods_i32.push(period as i32);
            alphas_f32.push(1.0f32 / (period as f32));
            warm_indices.push((first_valid + period - 1) as i32);
        }

        Ok(PreparedWildersBatch {
            first_valid,
            series_len,
            periods_i32,
            alphas_f32,
            warm_indices,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        if n_combos == 0 {
            return Ok(());
        }

        if matches!(self.policy.batch, BatchKernelPolicy::Auto) {
            if let Ok(mut func) = self.module.get_function("wilders_batch_warp_scan_f32") {
                let _ = func.set_cache_config(CacheConfig::PreferL1);

                let block_threads = 32u32;
                unsafe {
                    (*(self as *const _ as *mut CudaWilders)).last_batch =
                        Some(BatchKernelSelected::WarpScan {
                            block_x: block_threads,
                        });
                }
                self.maybe_log_batch_debug();

                let grid: GridSize = (n_combos as u32, 1, 1).into();
                let block: BlockSize = (block_threads, 1, 1).into();

                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                    let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
                    let mut warm_ptr = d_warm.as_device_ptr().as_raw();
                    let mut series_len_i = series_len as i32;
                    let mut first_valid_i = first_valid as i32;
                    let mut n_combos_i = n_combos as i32;
                    let mut out_ptr = d_out.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut alphas_ptr as *mut _ as *mut c_void,
                        &mut warm_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, 0, args)?;
                }

                return Ok(());
            }
        }

        let mut func = self.module.get_function("wilders_batch_f32").map_err(|_| {
            CudaWildersError::MissingKernelSymbol {
                name: "wilders_batch_f32",
            }
        })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x_user = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            BatchKernelPolicy::Auto => 256,
        };
        let block_threads = ((block_x_user / 32).max(1).min(32)) * 32;
        unsafe {
            (*(self as *const _ as *mut CudaWilders)).last_batch =
                Some(BatchKernelSelected::Plain {
                    block_x: block_threads,
                });
        }
        self.maybe_log_batch_debug();

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_threads, 1, 1).into();

        if block_threads > 1024 {
            return Err(CudaWildersError::LaunchConfigTooLarge {
                gx: n_combos as u32,
                gy: 1,
                gz: 1,
                bx: block_threads,
                by: 1,
                bz: 1,
            });
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
            let mut warm_ptr = d_warm.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        let mut func = self
            .module
            .get_function("wilders_many_series_one_param_f32")
            .map_err(|_| CudaWildersError::MissingKernelSymbol {
                name: "wilders_many_series_one_param_f32",
            })?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x_user = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
            ManySeriesKernelPolicy::Auto => {
                if num_series < 64 {
                    128
                } else {
                    256
                }
            }
        };
        let block_threads = ((block_x_user / 32).max(1).min(32)) * 32;
        let warps_per_block: u32 = (block_threads / 32) as u32;
        let grid_x: u32 = ((num_series as u32) + (warps_per_block - 1)) / warps_per_block;
        unsafe {
            (*(self as *const _ as *mut CudaWilders)).last_many =
                Some(ManySeriesKernelSelected::OneD {
                    block_x: block_threads,
                });
        }
        self.maybe_log_many_debug();

        let block: BlockSize = (block_threads, 1, 1).into();
        let grid: GridSize = (grid_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut alpha_f = alpha as f32;
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    pub fn wilders_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WildersParams,
    ) -> Result<DeviceArrayF32Wilders, CudaWildersError> {
        let (first_valids, period, alpha) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let in_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        let required = in_bytes
            .checked_add(first_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .and_then(|x| x.checked_add(16 * 1024 * 1024))
            .ok_or_else(|| CudaWildersError::InvalidInput("size overflow".into()))?;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            let (free, _tot) = mem_get_info().unwrap_or((0, 0));
            return Err(CudaWildersError::OutOfMemory {
                required,
                free,
                headroom: 64 * 1024 * 1024,
            });
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32).map_err(CudaWildersError::from)?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(CudaWildersError::from)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems)? };

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_first,
            period,
            alpha,
            cols,
            rows,
            &mut d_out_tm,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32Wilders {
            buf: d_out_tm,
            rows,
            cols,
            ctx: Arc::clone(&self._context),
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn wilders_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWildersError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaWildersError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if d_prices_tm.len() != num_series * series_len || d_out_tm.len() != num_series * series_len
        {
            return Err(CudaWildersError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaWildersError::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period,
            alpha,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if let Ok((free, _total)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                static ONCE: AtomicBool = AtomicBool::new(false);
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] WILDERS batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaWilders)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                static ONCE: AtomicBool = AtomicBool::new(false);
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] WILDERS many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaWilders)).debug_many_logged = true;
                }
            }
        }
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::wilders::WildersParams;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let param_bytes = PARAM_SWEEP * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + param_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + first_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct WildersBatchDevState {
        cuda: CudaWilders,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_alphas: DeviceBuffer<f32>,
        d_warm: DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for WildersBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_alphas,
                    &self.d_warm,
                    self.series_len,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("wilders batch kernel");
            self.cuda.stream.synchronize().expect("wilders sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaWilders::new(0).expect("cuda wilders");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = WildersBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let prep =
            CudaWilders::prepare_batch_inputs(&price, &sweep).expect("wilders prepare batch");
        let n_combos = prep.periods_i32.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&prep.periods_i32).expect("d_periods");
        let d_alphas = DeviceBuffer::from_slice(&prep.alphas_f32).expect("d_alphas");
        let d_warm = DeviceBuffer::from_slice(&prep.warm_indices).expect("d_warm");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prep.series_len * n_combos) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(WildersBatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_alphas,
            d_warm,
            series_len: prep.series_len,
            first_valid: prep.first_valid,
            n_combos,
            d_out,
        })
    }

    struct WildersManyDevState {
        cuda: CudaWilders,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: i32,
        alpha: f32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for WildersManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.period,
                    self.alpha,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                )
                .expect("wilders many-series kernel");
            self.cuda.stream.synchronize().expect("wilders sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaWilders::new(0).expect("cuda wilders");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = WildersParams { period: Some(64) };
        let (first_valids, period, alpha) =
            CudaWilders::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("wilders prepare many-series");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(WildersManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period: period as i32,
            alpha,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "wilders",
                "one_series_many_params",
                "wilders_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "wilders",
                "many_series_one_param",
                "wilders_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

fn expand_periods(range: &WildersBatchRange) -> Vec<WildersParams> {
    let (start, end, step) = range.period;

    if step == 0 || start == end {
        return vec![WildersParams {
            period: Some(start),
        }];
    }

    let mut out = Vec::new();
    if start < end {
        let mut v = start;
        while v <= end {
            out.push(WildersParams { period: Some(v) });
            match v.checked_add(step) {
                Some(n) => v = n,
                None => break,
            }
        }
    } else {
        let mut v = start;
        loop {
            if v < end {
                break;
            }
            out.push(WildersParams { period: Some(v) });
            if v < end + step {
                break;
            }
            v = v.saturating_sub(step);
            if v == 0 && end != 0 {
                break;
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaWildersPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    WarpScan { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}
