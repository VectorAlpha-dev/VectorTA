#![cfg(feature = "cuda")]

use crate::indicators::trend_continuation_factor::{
    expand_grid_trend_continuation_factor, TrendContinuationFactorBatchRange,
    TrendContinuationFactorParams,
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

const TREND_CONTINUATION_FACTOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaTrendContinuationFactorError {
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

pub struct TrendContinuationFactorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl TrendContinuationFactorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct TrendContinuationFactorDeviceArrayF64Pair {
    pub plus_tcf: TrendContinuationFactorDeviceArrayF64,
    pub minus_tcf: TrendContinuationFactorDeviceArrayF64,
}

impl TrendContinuationFactorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.plus_tcf.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.plus_tcf.cols
    }
}

pub struct CudaTrendContinuationFactorBatchResult {
    pub outputs: TrendContinuationFactorDeviceArrayF64Pair,
    pub combos: Vec<TrendContinuationFactorParams>,
}

pub struct CudaTrendContinuationFactor {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaTrendContinuationFactor {
    pub fn new(device_id: usize) -> Result<Self, CudaTrendContinuationFactorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("trend_continuation_factor_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaTrendContinuationFactorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_index(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaTrendContinuationFactorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaTrendContinuationFactorError::OutOfMemory {
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
    ) -> Result<(), CudaTrendContinuationFactorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaTrendContinuationFactorError::LaunchConfigTooLarge {
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
        data: &[f64],
        sweep: &TrendContinuationFactorBatchRange,
    ) -> Result<CudaTrendContinuationFactorBatchResult, CudaTrendContinuationFactorError> {
        if data.is_empty() {
            return Err(CudaTrendContinuationFactorError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = expand_grid_trend_continuation_factor(sweep)
            .map_err(|err| CudaTrendContinuationFactorError::InvalidInput(err.to_string()))?;
        let first_valid = Self::first_valid_index(data).ok_or_else(|| {
            CudaTrendContinuationFactorError::InvalidInput("all values are NaN".into())
        })?;
        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(35))
            .max()
            .unwrap_or(0);
        let valid = data.len().saturating_sub(first_valid);
        if valid <= max_length {
            return Err(CudaTrendContinuationFactorError::InvalidInput(format!(
                "not enough valid data: needed={}, valid={valid}",
                max_length + 1
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(35))
            .map(|length| {
                if length == 0 {
                    Err(CudaTrendContinuationFactorError::InvalidInput(
                        "length must be > 0".into(),
                    ))
                } else {
                    Ok(length as i32)
                }
            })
            .collect::<Result<_, _>>()?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaTrendContinuationFactorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaTrendContinuationFactorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaTrendContinuationFactorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaTrendContinuationFactorError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_len = rows.checked_mul(max_length).ok_or_else(|| {
            CudaTrendContinuationFactorError::InvalidInput("rows*max_length overflow".into())
        })?;
        let scratch_bytes = scratch_len
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaTrendContinuationFactorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaTrendContinuationFactorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_plus_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len)? };
        let d_minus_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len)? };
        let d_out_plus = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_minus = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("trend_continuation_factor_batch_f64")
            .map_err(|_| CudaTrendContinuationFactorError::MissingKernelSymbol {
                name: "trend_continuation_factor_batch_f64",
            })?;
        let grid_x = ((rows as u32) + TREND_CONTINUATION_FACTOR_BLOCK_X - 1)
            / TREND_CONTINUATION_FACTOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(TREND_CONTINUATION_FACTOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                d_plus_buffer.as_device_ptr(),
                d_minus_buffer.as_device_ptr(),
                d_out_plus.as_device_ptr(),
                d_out_minus.as_device_ptr()
            ))?;
        }

        Ok(CudaTrendContinuationFactorBatchResult {
            outputs: TrendContinuationFactorDeviceArrayF64Pair {
                plus_tcf: TrendContinuationFactorDeviceArrayF64 {
                    buf: d_out_plus,
                    rows,
                    cols,
                },
                minus_tcf: TrendContinuationFactorDeviceArrayF64 {
                    buf: d_out_minus,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
