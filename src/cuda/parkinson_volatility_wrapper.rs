#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::parkinson_volatility::{
    expand_grid_parkinson, ParkinsonVolatilityBatchRange, ParkinsonVolatilityParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::sync::Arc;
use thiserror::Error;

const BATCH_BLOCK_X: u32 = 256;
const MANY_SERIES_BLOCK_X: u32 = 256;
const MAX_GRID_Y: usize = 65_535;

#[derive(Debug, Error)]
pub enum CudaParkinsonVolatilityError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
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

pub struct ParkinsonDeviceArrayF32Pair {
    pub volatility: DeviceArrayF32,
    pub variance: DeviceArrayF32,
}

impl ParkinsonDeviceArrayF32Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.volatility.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.volatility.cols
    }
}

pub struct CudaParkinsonVolatilityBatchResult {
    pub outputs: ParkinsonDeviceArrayF32Pair,
    pub combos: Vec<ParkinsonVolatilityParams>,
}

pub struct CudaParkinsonVolatility {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaParkinsonVolatility {
    pub fn new(device_id: usize) -> Result<Self, CudaParkinsonVolatilityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/parkinson_volatility_kernel.ptx"));
        let module = crate::load_cuda_embedded_module!("parkinson_volatility_kernel")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
        })
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
    pub fn synchronize(&self) -> Result<(), CudaParkinsonVolatilityError> {
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
    fn bytes_for<T>(elems: usize) -> Result<usize, CudaParkinsonVolatilityError> {
        elems
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("byte size overflow".into()))
    }

    #[inline]
    fn will_fit(
        required_bytes: usize,
        headroom_bytes: usize,
    ) -> Result<(), CudaParkinsonVolatilityError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        match mem_get_info() {
            Ok((free, _total)) => {
                if required_bytes.saturating_add(headroom_bytes) <= free {
                    Ok(())
                } else {
                    Err(CudaParkinsonVolatilityError::OutOfMemory {
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
    fn grid_x_for_len(len: usize, block_x: u32) -> u32 {
        let blocks = ((len as u64) + (block_x as u64) - 1) / (block_x as u64);
        blocks.max(1).min(u32::MAX as u64) as u32
    }

    #[inline]
    fn valid_high_low(high: f32, low: f32) -> bool {
        high.is_finite() && low.is_finite() && high > 0.0 && low > 0.0
    }

    fn first_valid_host(high: &[f32], low: &[f32]) -> Option<usize> {
        high.iter()
            .zip(low.iter())
            .position(|(&h, &l)| Self::valid_high_low(h, l))
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaParkinsonVolatilityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut out = Vec::new();
            let mut x = start;
            while x <= end {
                out.push(x);
                match x.checked_add(step) {
                    Some(next) if next > x => x = next,
                    _ => break,
                }
            }
            if out.is_empty() {
                return Err(CudaParkinsonVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        } else {
            let mut out = Vec::new();
            let mut x = start;
            let st = step.max(1);
            while x >= end {
                out.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(st);
                if next == x || next < end {
                    break;
                }
                x = next;
            }
            if out.is_empty() {
                return Err(CudaParkinsonVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        }
    }

    fn build_combo_buffers(
        sweep: &ParkinsonVolatilityBatchRange,
        len: usize,
        first_valid: usize,
    ) -> Result<(Vec<ParkinsonVolatilityParams>, Vec<i32>, usize), CudaParkinsonVolatilityError>
    {
        let combos = expand_grid_parkinson(sweep)
            .map_err(|e| CudaParkinsonVolatilityError::InvalidInput(e.to_string()))?;
        let periods = Self::axis_usize(sweep.period)?;
        let mut max_period = 0usize;
        for &period in &periods {
            if period == 0 || period > len {
                return Err(CudaParkinsonVolatilityError::InvalidInput(
                    "invalid period".into(),
                ));
            }
            if len.saturating_sub(first_valid) < period {
                return Err(CudaParkinsonVolatilityError::InvalidInput(
                    "not enough valid data".into(),
                ));
            }
            max_period = max_period.max(period);
        }
        Ok((
            combos,
            periods.into_iter().map(|p| p as i32).collect(),
            max_period,
        ))
    }

    fn launch_prefix_kernel(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_prefix_sum: &mut DeviceBuffer<f64>,
        d_prefix_invalid: &mut DeviceBuffer<i32>,
    ) -> Result<(), CudaParkinsonVolatilityError> {
        let func = self
            .module
            .get_function("parkinson_volatility_build_prefix_f64")
            .map_err(|_| CudaParkinsonVolatilityError::MissingKernelSymbol {
                name: "parkinson_volatility_build_prefix_f64",
            })?;
        let grid: GridSize = (1, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        let stream = &self.stream;
        unsafe {
            cust::launch!(
                func<<<grid, block, 0, stream>>>(
                    d_high.as_device_ptr(),
                    d_low.as_device_ptr(),
                    len as i32,
                    first_valid as i32,
                    d_prefix_sum.as_device_ptr(),
                    d_prefix_invalid.as_device_ptr()
                )
            )?;
        }
        Ok(())
    }

    fn launch_batch_kernel(
        &self,
        len: usize,
        first_valid: usize,
        d_prefix_sum: &DeviceBuffer<f64>,
        d_prefix_invalid: &DeviceBuffer<i32>,
        d_periods: &DeviceBuffer<i32>,
        combos: usize,
        d_volatility: &mut DeviceBuffer<f32>,
        d_variance: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaParkinsonVolatilityError> {
        let func = self
            .module
            .get_function("parkinson_volatility_batch_f32")
            .map_err(|_| CudaParkinsonVolatilityError::MissingKernelSymbol {
                name: "parkinson_volatility_batch_f32",
            })?;

        let grid_x = Self::grid_x_for_len(len, BATCH_BLOCK_X);
        let block: BlockSize = (BATCH_BLOCK_X, 1, 1).into();
        let stream = &self.stream;

        let mut launched = 0usize;
        while launched < combos {
            let chunk = (combos - launched).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x, chunk as u32, 1).into();
            unsafe {
                let periods_ptr = (d_periods.as_device_ptr().as_raw()
                    + (launched as u64) * std::mem::size_of::<i32>() as u64)
                    as u64;
                let vol_ptr = (d_volatility.as_device_ptr().as_raw()
                    + (launched as u64) * (len as u64) * std::mem::size_of::<f32>() as u64)
                    as u64;
                let var_ptr = (d_variance.as_device_ptr().as_raw()
                    + (launched as u64) * (len as u64) * std::mem::size_of::<f32>() as u64)
                    as u64;
                cust::launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_prefix_sum.as_device_ptr(),
                        d_prefix_invalid.as_device_ptr(),
                        len as i32,
                        first_valid as i32,
                        periods_ptr,
                        chunk as i32,
                        vol_ptr,
                        var_ptr
                    )
                )?;
            }
            launched += chunk;
        }

        Ok(())
    }

    pub fn parkinson_volatility_batch_dev(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &ParkinsonVolatilityBatchRange,
    ) -> Result<CudaParkinsonVolatilityBatchResult, CudaParkinsonVolatilityError> {
        if high_f32.is_empty() || low_f32.is_empty() {
            return Err(CudaParkinsonVolatilityError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high_f32.len() != low_f32.len() {
            return Err(CudaParkinsonVolatilityError::InvalidInput(
                "length mismatch".into(),
            ));
        }
        let len = high_f32.len();
        let first_valid = Self::first_valid_host(high_f32, low_f32).ok_or_else(|| {
            CudaParkinsonVolatilityError::InvalidInput("all values are invalid".into())
        })?;

        let (combos, periods_i32, _) = Self::build_combo_buffers(sweep, len, first_valid)?;
        let len1 = len
            .checked_add(1)
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("len+1 overflow".into()))?;
        let input_bytes = Self::bytes_for::<f32>(len)?
            .checked_mul(2)
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        let prefix_bytes = Self::bytes_for::<f64>(len1)?
            .checked_add(Self::bytes_for::<i32>(len1)?)
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        let periods_bytes = Self::bytes_for::<i32>(periods_i32.len())?;
        let out_elems = combos.len().checked_mul(len).ok_or_else(|| {
            CudaParkinsonVolatilityError::InvalidInput("rows*cols overflow".into())
        })?;
        let out_bytes = Self::bytes_for::<f32>(out_elems)?
            .checked_mul(2)
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        let required = input_bytes
            .checked_add(prefix_bytes)
            .and_then(|v| v.checked_add(periods_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, Self::default_headroom())?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_f32, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_f32, &self.stream) }?;
        let result = self.parkinson_volatility_batch_dev_from_device_inputs(
            &d_high,
            &d_low,
            len,
            first_valid,
            sweep,
        )?;
        self.synchronize()?;
        Ok(result)
    }

    pub fn parkinson_volatility_batch_dev_from_device_inputs(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        sweep: &ParkinsonVolatilityBatchRange,
    ) -> Result<CudaParkinsonVolatilityBatchResult, CudaParkinsonVolatilityError> {
        if len == 0 || d_high.len() != len || d_low.len() != len {
            return Err(CudaParkinsonVolatilityError::InvalidInput(
                "device input length mismatch".into(),
            ));
        }

        let (combos, periods_i32, _) = Self::build_combo_buffers(sweep, len, first_valid)?;
        let len1 = len
            .checked_add(1)
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("len+1 overflow".into()))?;
        let prefix_bytes = Self::bytes_for::<f64>(len1)?
            .checked_add(Self::bytes_for::<i32>(len1)?)
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        let periods_bytes = Self::bytes_for::<i32>(periods_i32.len())?;
        let out_elems = combos.len().checked_mul(len).ok_or_else(|| {
            CudaParkinsonVolatilityError::InvalidInput("rows*cols overflow".into())
        })?;
        let out_bytes = Self::bytes_for::<f32>(out_elems)?
            .checked_mul(2)
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        let required = prefix_bytes
            .checked_add(periods_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, Self::default_headroom())?;

        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }?;
        let mut d_prefix_sum: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::uninitialized_async(len1, &self.stream) }?;
        let mut d_prefix_invalid: DeviceBuffer<i32> =
            unsafe { DeviceBuffer::uninitialized_async(len1, &self.stream) }?;
        let mut d_volatility: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;
        let mut d_variance: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        self.launch_prefix_kernel(
            d_high,
            d_low,
            len,
            first_valid,
            &mut d_prefix_sum,
            &mut d_prefix_invalid,
        )?;
        self.launch_batch_kernel(
            len,
            first_valid,
            &d_prefix_sum,
            &d_prefix_invalid,
            &d_periods,
            combos.len(),
            &mut d_volatility,
            &mut d_variance,
        )?;

        Ok(CudaParkinsonVolatilityBatchResult {
            outputs: ParkinsonDeviceArrayF32Pair {
                volatility: DeviceArrayF32 {
                    buf: d_volatility,
                    rows: combos.len(),
                    cols: len,
                },
                variance: DeviceArrayF32 {
                    buf: d_variance,
                    rows: combos.len(),
                    cols: len,
                },
            },
            combos,
        })
    }

    pub fn parkinson_volatility_many_series_one_param_time_major_dev(
        &self,
        high_tm_f32: &[f32],
        low_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<ParkinsonDeviceArrayF32Pair, CudaParkinsonVolatilityError> {
        if cols == 0 || rows == 0 {
            return Err(CudaParkinsonVolatilityError::InvalidInput(
                "invalid matrix dims".into(),
            ));
        }
        let total = cols.checked_mul(rows).ok_or_else(|| {
            CudaParkinsonVolatilityError::InvalidInput("rows*cols overflow".into())
        })?;
        if high_tm_f32.len() != total || low_tm_f32.len() != total {
            return Err(CudaParkinsonVolatilityError::InvalidInput(
                "matrix input length mismatch".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaParkinsonVolatilityError::InvalidInput(
                "invalid period".into(),
            ));
        }

        let mut first_valids = vec![-1i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if Self::valid_high_low(high_tm_f32[idx], low_tm_f32[idx]) {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let input_bytes = Self::bytes_for::<f32>(total)?
            .checked_mul(2)
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        let first_bytes = Self::bytes_for::<i32>(cols)?;
        let out_bytes = Self::bytes_for::<f32>(total)?
            .checked_mul(2)
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        let required = input_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaParkinsonVolatilityError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, Self::default_headroom())?;

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm_f32, &self.stream) }?;
        let d_low = unsafe { DeviceBuffer::from_slice_async(low_tm_f32, &self.stream) }?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_volatility: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;
        let mut d_variance: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(total, &self.stream) }?;

        let func = self
            .module
            .get_function("parkinson_volatility_many_series_one_param_f32")
            .map_err(|_| CudaParkinsonVolatilityError::MissingKernelSymbol {
                name: "parkinson_volatility_many_series_one_param_f32",
            })?;
        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (MANY_SERIES_BLOCK_X, 1, 1).into();
        let stream = &self.stream;
        unsafe {
            cust::launch!(
                func<<<grid, block, 0, stream>>>(
                    d_high.as_device_ptr(),
                    d_low.as_device_ptr(),
                    d_first.as_device_ptr(),
                    period as i32,
                    cols as i32,
                    rows as i32,
                    d_volatility.as_device_ptr(),
                    d_variance.as_device_ptr()
                )
            )?;
        }

        self.synchronize()?;

        Ok(ParkinsonDeviceArrayF32Pair {
            volatility: DeviceArrayF32 {
                buf: d_volatility,
                rows,
                cols,
            },
            variance: DeviceArrayF32 {
                buf: d_variance,
                rows,
                cols,
            },
        })
    }

    pub fn parkinson_volatility_batch_into_pinned_host_f32(
        &self,
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &ParkinsonVolatilityBatchRange,
    ) -> Result<
        (
            LockedBuffer<f32>,
            LockedBuffer<f32>,
            usize,
            usize,
            Vec<ParkinsonVolatilityParams>,
        ),
        CudaParkinsonVolatilityError,
    > {
        let result = self.parkinson_volatility_batch_dev(high_f32, low_f32, sweep)?;
        let rows = result.outputs.rows();
        let cols = result.outputs.cols();
        let total = rows.checked_mul(cols).ok_or_else(|| {
            CudaParkinsonVolatilityError::InvalidInput("rows*cols overflow".into())
        })?;
        let mut pinned_volatility: LockedBuffer<f32> =
            unsafe { LockedBuffer::uninitialized(total)? };
        let mut pinned_variance: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(total)? };
        unsafe {
            result
                .outputs
                .volatility
                .buf
                .async_copy_to(pinned_volatility.as_mut_slice(), &self.stream)?;
            result
                .outputs
                .variance
                .buf
                .async_copy_to(pinned_variance.as_mut_slice(), &self.stream)?;
        }
        self.synchronize()?;
        Ok((
            pinned_volatility,
            pinned_variance,
            rows,
            cols,
            result.combos,
        ))
    }
}
