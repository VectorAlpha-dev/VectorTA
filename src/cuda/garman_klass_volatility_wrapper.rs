#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::garman_klass_volatility::{
    GarmanKlassVolatilityBatchRange, GarmanKlassVolatilityParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::mem::size_of;
use std::sync::Arc;
use thiserror::Error;

const PREP_BLOCK_X: u32 = 256;
const BATCH_BLOCK_X: u32 = 256;
const MANY_BLOCK_X: u32 = 128;
const MAX_GRID_Y: usize = 65_535;

#[derive(Debug, Error)]
pub enum CudaGarmanKlassVolatilityError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
}

pub struct CudaGarmanKlassBatchResult {
    pub outputs: DeviceArrayF32,
    pub combos: Vec<GarmanKlassVolatilityParams>,
}

struct PreparedSeries {
    prefix_valid: DeviceBuffer<i32>,
    prefix_sum: DeviceBuffer<f32>,
    series_len: usize,
    first_valid: usize,
}

pub struct CudaGarmanKlassVolatility {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
}

impl CudaGarmanKlassVolatility {
    pub fn new(device_id: usize) -> Result<Self, CudaGarmanKlassVolatilityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(
            env!("OUT_DIR"),
            "/garman_klass_volatility_kernel.ptx"
        ));
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
        })
    }

    #[inline]
    pub fn context_arc_clone(&self) -> Arc<Context> {
        self._context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaGarmanKlassVolatilityError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn default_headroom() -> usize {
        env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024)
    }

    #[inline]
    fn will_fit(
        bytes_needed: usize,
        headroom: usize,
    ) -> Result<(), CudaGarmanKlassVolatilityError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _)) = mem_get_info() {
            if bytes_needed.saturating_add(headroom) <= free {
                Ok(())
            } else {
                Err(CudaGarmanKlassVolatilityError::OutOfMemory {
                    required: bytes_needed,
                    free,
                    headroom,
                })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn grid_x_for_len(len: usize, block_x: u32) -> u32 {
        let blocks = ((len as u64) + (block_x as u64) - 1) / (block_x as u64);
        blocks.max(1).min(u32::MAX as u64) as u32
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaGarmanKlassVolatilityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let step = step.max(1);
        if start < end {
            let mut out = Vec::new();
            let mut x = start;
            while x <= end {
                out.push(x);
                match x.checked_add(step) {
                    Some(next) if next != x => x = next,
                    _ => break,
                }
            }
            if out.is_empty() {
                return Err(CudaGarmanKlassVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        } else {
            let mut out = Vec::new();
            let mut x = start;
            loop {
                out.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(step);
                if next == x || next < end {
                    break;
                }
                x = next;
            }
            if out.is_empty() {
                return Err(CudaGarmanKlassVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        }
    }

    fn find_first_valid_ohlc(
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Option<usize> {
        (0..close.len()).find(|&i| {
            open[i].is_finite()
                && high[i].is_finite()
                && low[i].is_finite()
                && close[i].is_finite()
                && open[i] > 0.0
                && high[i] > 0.0
                && low[i] > 0.0
                && close[i] > 0.0
        })
    }

    fn count_valid_ohlc(open: &[f32], high: &[f32], low: &[f32], close: &[f32]) -> usize {
        (0..close.len())
            .filter(|&i| {
                open[i].is_finite()
                    && high[i].is_finite()
                    && low[i].is_finite()
                    && close[i].is_finite()
                    && open[i] > 0.0
                    && high[i] > 0.0
                    && low[i] > 0.0
                    && close[i] > 0.0
            })
            .count()
    }

    fn expand_combos(
        sweep: &GarmanKlassVolatilityBatchRange,
    ) -> Result<Vec<GarmanKlassVolatilityParams>, CudaGarmanKlassVolatilityError> {
        Ok(Self::axis_usize(sweep.lookback)?
            .into_iter()
            .map(|lookback| GarmanKlassVolatilityParams {
                lookback: Some(lookback),
            })
            .collect())
    }

    fn prepare_prefix_series_host(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<PreparedSeries, CudaGarmanKlassVolatilityError> {
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "OHLC length mismatch".to_string(),
            ));
        }
        let len = close.len();
        if len == 0 {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "empty input".to_string(),
            ));
        }
        let first_valid = Self::find_first_valid_ohlc(open, high, low, close).ok_or_else(|| {
            CudaGarmanKlassVolatilityError::InvalidInput("all values are invalid".to_string())
        })?;

        let d_open = unsafe { DeviceBuffer::from_slice_async(open, &self.stream) }?;
        let d_high = unsafe { DeviceBuffer::from_slice_async(high, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low, &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close, &self.stream) }?;
        self.prepare_prefix_series_device(&d_open, &d_high, &d_low, &d_close, first_valid)
    }

    fn prepare_prefix_series_device(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        first_valid: usize,
    ) -> Result<PreparedSeries, CudaGarmanKlassVolatilityError> {
        let len = d_close.len();
        if len == 0 {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "empty input".to_string(),
            ));
        }
        if d_open.len() != len || d_high.len() != len || d_low.len() != len {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "OHLC length mismatch".to_string(),
            ));
        }
        if first_valid >= len {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "first_valid out of range".to_string(),
            ));
        }

        let prefix_len = len.checked_add(1).ok_or_else(|| {
            CudaGarmanKlassVolatilityError::InvalidInput("series len overflow".to_string())
        })?;
        let required = len
            .checked_mul(size_of::<i32>() + size_of::<f32>())
            .and_then(|v| v.checked_add(prefix_len * (size_of::<i32>() + size_of::<f32>())))
            .ok_or_else(|| {
                CudaGarmanKlassVolatilityError::InvalidInput("byte size overflow".to_string())
            })?;
        Self::will_fit(required, Self::default_headroom())?;

        let mut d_valid: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        let mut d_term: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }?;
        let mut prefix_valid: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;
        let mut prefix_sum: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prefix_len, &self.stream) }?;

        let prep = self
            .module
            .get_function("garman_klass_precompute_terms_f32")
            .map_err(|_| CudaGarmanKlassVolatilityError::MissingKernelSymbol {
                name: "garman_klass_precompute_terms_f32",
            })?;
        let prep_grid = GridSize::x(Self::grid_x_for_len(len, PREP_BLOCK_X));
        let prep_block = BlockSize::x(PREP_BLOCK_X);
        let stream = &self.stream;
        unsafe {
            launch!(
                prep<<<prep_grid, prep_block, 0, stream>>>(
                    d_open.as_device_ptr(),
                    d_high.as_device_ptr(),
                    d_low.as_device_ptr(),
                    d_close.as_device_ptr(),
                    len as i32,
                    d_valid.as_device_ptr(),
                    d_term.as_device_ptr()
                )
            )?;
        }

        let prefix = self
            .module
            .get_function("garman_klass_prefix_terms_f32")
            .map_err(|_| CudaGarmanKlassVolatilityError::MissingKernelSymbol {
                name: "garman_klass_prefix_terms_f32",
            })?;
        unsafe {
            launch!(
                prefix<<<GridSize::x(1), BlockSize::x(1), 0, stream>>>(
                    d_valid.as_device_ptr(),
                    d_term.as_device_ptr(),
                    len as i32,
                    prefix_valid.as_device_ptr(),
                    prefix_sum.as_device_ptr()
                )
            )?;
        }
        self.synchronize()?;

        Ok(PreparedSeries {
            prefix_valid,
            prefix_sum,
            series_len: len,
            first_valid,
        })
    }

    fn launch_batch_kernel(
        &self,
        prepared: &PreparedSeries,
        d_lookbacks: &DeviceBuffer<i32>,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaGarmanKlassVolatilityError> {
        let func = self
            .module
            .get_function("garman_klass_volatility_batch_prefix_f32")
            .map_err(|_| CudaGarmanKlassVolatilityError::MissingKernelSymbol {
                name: "garman_klass_volatility_batch_prefix_f32",
            })?;

        let grid_x = Self::grid_x_for_len(prepared.series_len, BATCH_BLOCK_X);
        let grid_y = n_combos.min(MAX_GRID_Y).max(1) as u32;
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<(grid_x, grid_y, 1), (BATCH_BLOCK_X, 1, 1), 0, stream>>>(
                    d_lookbacks.as_device_ptr(),
                    prepared.series_len as i32,
                    prepared.first_valid as i32,
                    n_combos as i32,
                    prepared.prefix_valid.as_device_ptr(),
                    prepared.prefix_sum.as_device_ptr(),
                    d_out.as_device_ptr()
                )
            )?;
        }
        Ok(())
    }

    pub fn garman_klass_volatility_batch_dev(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &GarmanKlassVolatilityBatchRange,
    ) -> Result<CudaGarmanKlassBatchResult, CudaGarmanKlassVolatilityError> {
        let len = close.len();
        if open.len() != len || high.len() != len || low.len() != len {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "OHLC length mismatch".to_string(),
            ));
        }
        let valid = Self::count_valid_ohlc(open, high, low, close);
        let combos = Self::expand_combos(sweep)?;
        let max_lookback = combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(14))
            .max()
            .unwrap_or(0);
        if max_lookback == 0 || max_lookback > len || valid < max_lookback {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "invalid lookback or insufficient valid data".to_string(),
            ));
        }

        let prepared = self.prepare_prefix_series_host(open, high, low, close)?;
        let lookbacks: Vec<i32> = combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(14) as i32)
            .collect();
        let d_lookbacks = unsafe { DeviceBuffer::from_slice_async(&lookbacks, &self.stream) }?;
        let out_elems = combos
            .len()
            .checked_mul(prepared.series_len)
            .ok_or_else(|| {
                CudaGarmanKlassVolatilityError::InvalidInput("rows*cols overflow".to_string())
            })?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;
        self.launch_batch_kernel(&prepared, &d_lookbacks, combos.len(), &mut d_out)?;
        self.synchronize()?;

        Ok(CudaGarmanKlassBatchResult {
            outputs: DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: prepared.series_len,
            },
            combos,
        })
    }

    pub fn garman_klass_volatility_batch_from_device(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        first_valid: usize,
        sweep: &GarmanKlassVolatilityBatchRange,
    ) -> Result<CudaGarmanKlassBatchResult, CudaGarmanKlassVolatilityError> {
        let len = d_close.len();
        let combos = Self::expand_combos(sweep)?;
        let max_lookback = combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(14))
            .max()
            .unwrap_or(0);
        if max_lookback == 0 || max_lookback > len {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "invalid lookback".to_string(),
            ));
        }

        let prepared =
            self.prepare_prefix_series_device(d_open, d_high, d_low, d_close, first_valid)?;
        let lookbacks: Vec<i32> = combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(14) as i32)
            .collect();
        let d_lookbacks = unsafe { DeviceBuffer::from_slice_async(&lookbacks, &self.stream) }?;
        let out_elems = combos
            .len()
            .checked_mul(prepared.series_len)
            .ok_or_else(|| {
                CudaGarmanKlassVolatilityError::InvalidInput("rows*cols overflow".to_string())
            })?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;
        self.launch_batch_kernel(&prepared, &d_lookbacks, combos.len(), &mut d_out)?;
        self.synchronize()?;

        Ok(CudaGarmanKlassBatchResult {
            outputs: DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: prepared.series_len,
            },
            combos,
        })
    }

    pub fn garman_klass_volatility_many_series_one_param_time_major_dev(
        &self,
        open_tm: &[f32],
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        lookback: usize,
    ) -> Result<DeviceArrayF32, CudaGarmanKlassVolatilityError> {
        if cols == 0 || rows == 0 {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "invalid matrix dims".to_string(),
            ));
        }
        let total = cols.checked_mul(rows).ok_or_else(|| {
            CudaGarmanKlassVolatilityError::InvalidInput("rows*cols overflow".to_string())
        })?;
        if open_tm.len() != total
            || high_tm.len() != total
            || low_tm.len() != total
            || close_tm.len() != total
        {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "matrix input length mismatch".to_string(),
            ));
        }
        if lookback == 0 || lookback > rows {
            return Err(CudaGarmanKlassVolatilityError::InvalidInput(
                "invalid lookback".to_string(),
            ));
        }

        let required = total.checked_mul(5 * size_of::<f32>()).ok_or_else(|| {
            CudaGarmanKlassVolatilityError::InvalidInput("byte size overflow".to_string())
        })?;
        Self::will_fit(required, Self::default_headroom())?;

        let d_open = unsafe { DeviceBuffer::from_slice_async(open_tm, &self.stream) }?;
        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }?;
        let d_close = unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;

        let func = self
            .module
            .get_function("garman_klass_volatility_many_series_one_param_f32")
            .map_err(|_| CudaGarmanKlassVolatilityError::MissingKernelSymbol {
                name: "garman_klass_volatility_many_series_one_param_f32",
            })?;
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<GridSize::x(Self::grid_x_for_len(cols, MANY_BLOCK_X)), BlockSize::x(MANY_BLOCK_X), 0, stream>>>(
                    d_open.as_device_ptr(),
                    d_high.as_device_ptr(),
                    d_low.as_device_ptr(),
                    d_close.as_device_ptr(),
                    cols as i32,
                    rows as i32,
                    lookback as i32,
                    d_out.as_device_ptr()
                )
            )?;
        }
        self.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}
