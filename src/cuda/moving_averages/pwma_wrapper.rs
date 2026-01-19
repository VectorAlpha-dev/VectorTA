#![cfg(feature = "cuda")]

use super::cwma_wrapper::{BatchKernelPolicy, BatchThreadsPerOutput, ManySeriesKernelPolicy};
use crate::indicators::moving_averages::pwma::{expand_grid, PwmaBatchRange, PwmaParams};
use cust::context::Context;
use cust::context::{CacheConfig, SharedMemoryConfig};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::{c_void, CStr};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

const PWMA_MAX_PERIOD_CONST: usize = 4096;
const BATCH_TX: u32 = 128;

#[derive(Debug, Error)]
pub enum CudaPwmaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch config too large: grid=({gx},{gy},{gz}), block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("out of memory: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct DeviceArrayF32Pwma {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub ctx: Arc<Context>,
    pub device_id: u32,
}
impl DeviceArrayF32Pwma {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaPwma {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaPwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,

    cmem_available: bool,
    cmem_scratch: [f32; PWMA_MAX_PERIOD_CONST],
}

impl CudaPwma {
    pub fn new(device_id: usize) -> Result<Self, CudaPwmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pwma_kernel.ptx"));
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

        let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"pwma_const_w\0") };
        let cmem_available = module
            .get_global::<[f32; PWMA_MAX_PERIOD_CONST]>(name)
            .is_ok();

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaPwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            cmem_available,
            cmem_scratch: [0.0f32; PWMA_MAX_PERIOD_CONST],
        })
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaPwmaError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn pascal_weights_f32(period: usize) -> Result<Vec<f32>, CudaPwmaError> {
        if period == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "period must be greater than zero".into(),
            ));
        }
        let n = period - 1;
        let mut row = Vec::with_capacity(period);
        let mut sum = 0.0f64;
        for r in 0..=n {
            let mut val = 1.0f64;
            for i in 0..r {
                val *= (n - i) as f64;
                val /= (i + 1) as f64;
            }
            row.push(val);
            sum += val;
        }
        if sum == 0.0 {
            return Err(CudaPwmaError::InvalidInput(format!(
                "Pascal weights sum to zero for period {}",
                period
            )));
        }
        let inv = 1.0 / sum;
        Ok(row.into_iter().map(|v| (v * inv) as f32).collect())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &PwmaBatchRange,
    ) -> Result<(Vec<PwmaParams>, usize, usize, usize, Vec<f32>), CudaPwmaError> {
        if data_f32.is_empty() {
            return Err(CudaPwmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaPwmaError::InvalidInput("all values are NaN".into()))?;
        let len = data_f32.len();

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaPwmaError::InvalidRange {
                start: sweep.period.0,
                end: sweep.period.1,
                step: sweep.period.2,
            });
        }

        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaPwmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaPwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaPwmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            if period > max_period {
                max_period = period;
            }
        }

        let n_combos = combos.len();
        let n_weights = n_combos
            .checked_mul(max_period)
            .ok_or_else(|| CudaPwmaError::InvalidInput("n_combos*max_period overflows".into()))?;
        let mut weights_flat = vec![0.0f32; n_weights];
        for (row, prm) in combos.iter().enumerate() {
            let weights = Self::pascal_weights_f32(prm.period.unwrap())?;
            let base = row * max_period;
            for (idx, w) in weights.iter().enumerate() {
                weights_flat[base + idx] = *w;
            }
        }

        Ok((combos, first_valid, len, max_period, weights_flat))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        if series_len == 0 || n_combos == 0 || max_period == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaPwmaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        let mut use_tiled = true;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { .. } => use_tiled = false,
            BatchKernelPolicy::Tiled { .. } => use_tiled = true,
        }

        if use_tiled {
            if let Ok(mut func) = self.module.get_function("pwma_batch_tiled_async_f32") {
                let tile_x: usize = BATCH_TX as usize;
                let align16 = |x: usize| (x + 15) & !15usize;
                let shared_bytes = (align16(max_period * std::mem::size_of::<f32>())
                    + 2 * (tile_x + max_period - 1) * std::mem::size_of::<f32>())
                    as u32;
                self.prefer_shared_and_optin_smem(&mut func, shared_bytes as usize);

                for (start, len) in Self::grid_y_chunks(n_combos) {
                    let grid_x = ((series_len as u32) + tile_x as u32 - 1) / (tile_x as u32);
                    let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
                    let block: BlockSize = (tile_x as u32, 1, 1).into();

                    unsafe {
                        let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                        let mut weights_ptr =
                            d_weights.as_device_ptr().add(start * max_period).as_raw();
                        let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                        let mut warms_ptr = d_warms.as_device_ptr().add(start).as_raw();
                        let mut series_len_i = series_len as i32;
                        let mut n_combos_i = len as i32;
                        let mut max_period_i = max_period as i32;
                        let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut weights_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut warms_ptr as *mut _ as *mut c_void,
                            &mut series_len_i as *mut _ as *mut c_void,
                            &mut n_combos_i as *mut _ as *mut c_void,
                            &mut max_period_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        if self
                            .stream
                            .launch(&func, grid, block, shared_bytes, args)
                            .is_err()
                        {
                            use_tiled = false;
                            break;
                        }
                    }
                }

                if use_tiled {
                    unsafe {
                        let this = self as *const _ as *mut CudaPwma;
                        (*this).last_batch = Some(BatchKernelSelected::Plain { block_x: 128 });
                    }
                    self.maybe_log_batch_debug();
                    return Ok(());
                }
            }
        }

        let func = self.module.get_function("pwma_batch_f32").map_err(|_| {
            CudaPwmaError::MissingKernelSymbol {
                name: "pwma_batch_f32",
            }
        })?;

        unsafe {
            let this = self as *const _ as *mut CudaPwma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x: 256 });
        }
        self.maybe_log_batch_debug();

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => 256,
        };
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

        for (start, len) in Self::grid_y_chunks(n_combos) {
            let grid_x = ((series_len as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut weights_ptr = d_weights.as_device_ptr().add(start * max_period).as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut warms_ptr = d_warms.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = len as i32;
                let mut max_period_i = max_period as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, shared_bytes, args)?;
            }
        }

        Ok(())
    }

    pub fn pwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        self.launch_batch_kernel(
            d_prices, d_weights, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn pwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PwmaBatchRange,
    ) -> Result<DeviceArrayF32Pwma, CudaPwmaError> {
        let (combos, first_valid, series_len, max_period, weights_flat) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let warms_i32: Vec<i32> = combos
            .iter()
            .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
            .collect();

        let szf = std::mem::size_of::<f32>();
        let szi = std::mem::size_of::<i32>();
        let prices_bytes = series_len
            .checked_mul(szf)
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        let weights_bytes = n_combos
            .checked_mul(max_period)
            .and_then(|v| v.checked_mul(szf))
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(szi)
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        let warms_bytes = n_combos
            .checked_mul(szi)
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = n_combos
            .checked_mul(series_len)
            .and_then(|v| v.checked_mul(szf))
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|v| v.checked_add(periods_bytes))
            .and_then(|v| v.checked_add(warms_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        Self::ensure_fit(required, 64 * 1024 * 1024)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_warms = DeviceBuffer::from_slice(&warms_i32)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }?;

        self.launch_batch_kernel(
            &d_prices, &d_weights, &d_periods, &d_warms, series_len, n_combos, max_period,
            &mut d_out,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32Pwma {
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
        params: &PwmaParams,
    ) -> Result<(Vec<i32>, Vec<f32>, usize), CudaPwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaPwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(CudaPwmaError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaPwmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let weights = Self::pascal_weights_f32(period)?;

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
                CudaPwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period {
                return Err(CudaPwmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaPwmaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, weights, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
        use_const: bool,
    ) -> Result<(), CudaPwmaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaPwmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let try_2d = |tx: u32, ty: u32| -> Option<()> {
            let fname = match (tx, ty) {
                (128, 4) => "pwma_ms1p_tiled_f32_tx128_ty4",
                (128, 2) => "pwma_ms1p_tiled_f32_tx128_ty2",
                _ => return None,
            };
            let mut func = match self.module.get_function(fname) {
                Ok(f) => f,
                Err(_) => return None,
            };
            let wlen = period;
            let align16 = |x: usize| (x + 15) & !15usize;
            let total = tx as usize + wlen - 1;

            let ty_pad = if (32 % (ty as usize)) == 0 {
                (ty + 1) as usize
            } else {
                ty as usize
            };
            let shared_bytes = (align16(wlen * std::mem::size_of::<f32>())
                + total * ty as usize * std::mem::size_of::<f32>())
                as u32;
            let grid_x = ((series_len as u32) + tx - 1) / tx;
            let grid_y = ((num_series as u32) + ty - 1) / ty;
            let grid: GridSize = (grid_x, grid_y, 1).into();
            let block: BlockSize = (tx, ty, 1).into();

            self.prefer_shared_and_optin_smem(&mut func, shared_bytes as usize);
            unsafe {
                let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
                let mut weights_ptr = d_weights.as_device_ptr().as_raw();
                let mut period_i = period as i32;
                let mut inv_norm = 1.0f32;
                let mut num_series_i = num_series as i32;
                let mut series_len_i = series_len as i32;
                let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut inv_norm as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_valids_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaPwmaError::Cuda(e))
                    .ok()?;
            }
            unsafe {
                let this = self as *const _ as *mut CudaPwma;
                (*this).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
            }
            self.maybe_log_many_debug();
            Some(())
        };

        match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                if try_2d(tx as u32, ty as u32).is_some() {
                    return Ok(());
                }
            }
            ManySeriesKernelPolicy::OneD { .. } => {}

            ManySeriesKernelPolicy::Auto => {}
        }

        if use_const {
            let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"pwma_const_w\0") };
            if let (Ok(_sym), Ok(func)) = (
                self.module.get_global::<[f32; PWMA_MAX_PERIOD_CONST]>(name),
                self.module.get_function("pwma_ms1p_const_f32"),
            ) {
                if period <= PWMA_MAX_PERIOD_CONST {
                    let block_x: u32 = match self.policy.many_series {
                        ManySeriesKernelPolicy::OneD { block_x } => block_x,
                        _ => 128,
                    };
                    let grid_x = ((series_len as u32) + block_x - 1) / block_x;
                    let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    let shared_bytes = 0u32;

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
                        self.stream.launch(&func, grid, block, shared_bytes, args)?;
                    }
                    unsafe {
                        let this = self as *const _ as *mut CudaPwma;
                        (*this).last_many = Some(ManySeriesKernelSelected::Const1D { block_x });
                    }
                    self.maybe_log_many_debug();
                    return Ok(());
                }
            }
        }

        let func = self
            .module
            .get_function("pwma_multi_series_one_param_f32")
            .map_err(|_| CudaPwmaError::MissingKernelSymbol {
                name: "pwma_multi_series_one_param_f32",
            })?;

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv_norm = 1.0f32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv_norm as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, shared_bytes, args)?;
        }
        unsafe {
            let this = self as *const _ as *mut CudaPwma;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn pwma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaPwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_weights,
            d_first_valids,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_out_tm,
            false,
        )
    }

    pub fn pwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PwmaParams,
    ) -> Result<DeviceArrayF32Pwma, CudaPwmaError> {
        let (first_valids, weights, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let prices_bytes = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        let use_const = self.cmem_available
            && self.module.get_function("pwma_ms1p_const_f32").is_ok()
            && period <= PWMA_MAX_PERIOD_CONST;
        let weights_bytes = if use_const {
            0
        } else {
            period
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?
        };
        let fv_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|v| v.checked_add(fv_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaPwmaError::InvalidInput("byte size overflow".into()))?;
        Self::ensure_fit(required, 64 * 1024 * 1024)?;

        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }?;
        let use_const = self.cmem_available
            && self.module.get_function("pwma_ms1p_const_f32").is_ok()
            && period <= PWMA_MAX_PERIOD_CONST;
        let d_weights = if use_const {
            unsafe { DeviceBuffer::uninitialized(0) }?
        } else {
            DeviceBuffer::from_slice(&weights)?
        };
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;

        if use_const {
            let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"pwma_const_w\0") };
            if let Ok(mut sym) = self.module.get_global::<[f32; PWMA_MAX_PERIOD_CONST]>(name) {
                let mut arr = self.cmem_scratch;
                for v in arr.iter_mut() {
                    *v = 0.0;
                }
                arr[..period].copy_from_slice(&weights);
                unsafe { sym.copy_from(&arr) }?;
            }
        }

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
            use_const,
        )?;
        self.stream.synchronize()?;

        Ok(DeviceArrayF32Pwma {
            buf: d_out_tm,
            rows,
            cols,
            ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn pwma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaPwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaPwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, weights, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }?;
        let use_const = self.cmem_available
            && self.module.get_function("pwma_ms1p_const_f32").is_ok()
            && period <= PWMA_MAX_PERIOD_CONST;
        let d_weights = if use_const {
            unsafe { DeviceBuffer::uninitialized(0) }?
        } else {
            DeviceBuffer::from_slice(&weights)?
        };
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;
        if use_const {
            let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"pwma_const_w\0") };
            if let Ok(mut sym) = self.module.get_global::<[f32; PWMA_MAX_PERIOD_CONST]>(name) {
                let mut arr = self.cmem_scratch;
                for v in arr.iter_mut() {
                    *v = 0.0;
                }
                arr[..period].copy_from_slice(&weights);
                unsafe { sym.copy_from(&arr) }?;
            }
        }

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
            use_const,
        )?;
        self.stream.synchronize()?;

        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(cols * rows) }?;
        unsafe {
            d_out_tm.async_copy_to(&mut pinned, &self.stream)?;
        }
        self.stream.synchronize()?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::pwma::{PwmaBatchRange, PwmaParams};

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

    struct PwmaBatchDevState {
        cuda: CudaPwma,
        d_prices: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_warms: DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for PwmaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .pwma_batch_device(
                    &self.d_prices,
                    &self.d_weights,
                    &self.d_periods,
                    &self.d_warms,
                    self.series_len,
                    self.n_combos,
                    self.max_period,
                    &mut self.d_out,
                )
                .expect("pwma batch kernel");
            self.cuda.stream.synchronize().expect("pwma sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaPwma::new(0).expect("cuda pwma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = PwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };

        let (combos, first_valid, series_len, max_period, weights_flat) =
            CudaPwma::prepare_batch_inputs(&price, &sweep).expect("pwma prepare batch inputs");
        let n_combos = combos.len();
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let warms_i32: Vec<i32> = combos
            .iter()
            .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
            .collect();

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("d_warms");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(PwmaBatchDevState {
            cuda,
            d_prices,
            d_weights,
            d_periods,
            d_warms,
            series_len,
            n_combos,
            max_period,
            d_out,
        })
    }

    struct PwmaManyDevState {
        cuda: CudaPwma,
        d_prices_tm: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        period: usize,
        use_const: bool,
        cols: usize,
        rows: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for PwmaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_weights,
                    &self.d_first_valids,
                    self.period,
                    self.cols,
                    self.rows,
                    &mut self.d_out_tm,
                    self.use_const,
                )
                .expect("pwma many-series kernel");
            self.cuda.stream.synchronize().expect("pwma sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaPwma::new(0).expect("cuda pwma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = PwmaParams { period: Some(64) };
        let (first_valids, weights, period) =
            CudaPwma::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("pwma prepare many-series inputs");

        let use_const = cuda.cmem_available
            && cuda.module.get_function("pwma_ms1p_const_f32").is_ok()
            && period <= PWMA_MAX_PERIOD_CONST;

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_weights = if use_const {
            unsafe { DeviceBuffer::uninitialized(0) }.expect("d_weights")
        } else {
            DeviceBuffer::from_slice(&weights).expect("d_weights")
        };
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");

        if use_const {
            let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"pwma_const_w\0") };
            if let Ok(mut sym) = cuda.module.get_global::<[f32; PWMA_MAX_PERIOD_CONST]>(name) {
                let mut arr = cuda.cmem_scratch;
                for v in arr.iter_mut() {
                    *v = 0.0;
                }
                arr[..period].copy_from_slice(&weights);
                unsafe { sym.copy_from(&arr) }.expect("pwma const copy");
            }
        }

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(PwmaManyDevState {
            cuda,
            d_prices_tm,
            d_weights,
            d_first_valids,
            period,
            use_const,
            cols,
            rows,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "pwma",
                "one_series_many_params",
                "pwma_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "pwma",
                "many_series_one_param",
                "pwma_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaPwmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaPwmaPolicy {
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
    Tiled2D { tx: u32, ty: u32 },
    Const1D { block_x: u32 },
}

impl CudaPwma {
    pub fn policy(&self) -> &CudaPwmaPolicy {
        &self.policy
    }
    pub fn set_policy(&mut self, policy: CudaPwmaPolicy) {
        self.policy = policy;
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
                    eprintln!("[DEBUG] PWMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaPwma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] PWMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaPwma)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
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
    fn ensure_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaPwmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaPwmaError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    #[inline]
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX: usize = 65_535;
        (0..n).step_by(MAX).map(move |start| {
            let len = (n - start).min(MAX);
            (start, len)
        })
    }

    #[inline]
    fn prefer_shared_and_optin_smem(&self, func: &mut Function, requested_dynamic_smem: usize) {
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let _ = func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize);
        unsafe {
            use cust::sys::{cuFuncSetAttribute, CUfunction_attribute_enum as Attr};
            let raw = func.to_raw();
            let _ = cuFuncSetAttribute(
                raw,
                Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                requested_dynamic_smem as i32,
            );
            let _ = cuFuncSetAttribute(
                raw,
                Attr::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            );
        }
    }
}
