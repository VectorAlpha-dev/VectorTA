#![cfg(feature = "cuda")]

use crate::indicators::trend_trigger_factor::{
    expand_grid, TrendTriggerFactorBatchRange, TrendTriggerFactorParams,
};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

const TREND_TRIGGER_FACTOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaTrendTriggerFactorError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
}

pub struct TrendTriggerFactorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl TrendTriggerFactorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaTrendTriggerFactorBatchResult {
    pub outputs: TrendTriggerFactorDeviceArrayF64,
    pub combos: Vec<TrendTriggerFactorParams>,
}

pub struct CudaTrendTriggerFactor {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaTrendTriggerFactor {
    pub fn new(device_id: usize) -> Result<Self, CudaTrendTriggerFactorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("trend_trigger_factor_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaTrendTriggerFactorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaTrendTriggerFactorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaTrendTriggerFactorError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaTrendTriggerFactorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaTrendTriggerFactorError::LaunchConfigTooLarge {
                gx: grid.x,
                gy: grid.y,
                gz: grid.z,
                bx: block.x,
                by: block.y,
                bz: block.z,
            });
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        sweep: &TrendTriggerFactorBatchRange,
    ) -> Result<CudaTrendTriggerFactorBatchResult, CudaTrendTriggerFactorError> {
        if high.is_empty() || low.is_empty() {
            return Err(CudaTrendTriggerFactorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() {
            return Err(CudaTrendTriggerFactorError::InvalidInput(format!(
                "input length mismatch: high={}, low={}",
                high.len(),
                low.len()
            )));
        }

        let first_valid = (0..high.len())
            .find(|&idx| high[idx].is_finite() && low[idx].is_finite())
            .ok_or_else(|| {
                CudaTrendTriggerFactorError::InvalidInput("all values are NaN".into())
            })?;
        let combos = expand_grid(sweep)
            .map_err(|err| CudaTrendTriggerFactorError::InvalidInput(err.to_string()))?;
        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(15))
            .max()
            .unwrap_or(0);
        let valid = high.len().saturating_sub(first_valid);
        if valid < max_length {
            return Err(CudaTrendTriggerFactorError::InvalidInput(format!(
                "not enough valid data: needed={max_length}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = high.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(15))
            .map(|length| {
                if length == 0 {
                    Err(CudaTrendTriggerFactorError::InvalidInput(
                        "length must be > 0".into(),
                    ))
                } else {
                    Ok(length as i32)
                }
            })
            .collect::<Result<_, _>>()?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaTrendTriggerFactorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaTrendTriggerFactorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaTrendTriggerFactorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaTrendTriggerFactorError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_len = rows.checked_mul(max_length).ok_or_else(|| {
            CudaTrendTriggerFactorError::InvalidInput("rows*max_length overflow".into())
        })?;
        let scratch_bytes = scratch_len
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| {
                scratch_len
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| other.checked_mul(2))
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaTrendTriggerFactorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaTrendTriggerFactorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let mut d_maxq_idx = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_len)? };
        let mut d_minq_idx = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_len)? };
        let mut d_hh_history = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len)? };
        let mut d_ll_history = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len)? };
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("trend_trigger_factor_batch_f64")
            .map_err(|_| CudaTrendTriggerFactorError::MissingKernelSymbol {
                name: "trend_trigger_factor_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + TREND_TRIGGER_FACTOR_BLOCK_X - 1) / TREND_TRIGGER_FACTOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(TREND_TRIGGER_FACTOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                cols as i32,
                first_valid as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                d_maxq_idx.as_device_ptr(),
                d_minq_idx.as_device_ptr(),
                d_hh_history.as_device_ptr(),
                d_ll_history.as_device_ptr(),
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaTrendTriggerFactorBatchResult {
            outputs: TrendTriggerFactorDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
