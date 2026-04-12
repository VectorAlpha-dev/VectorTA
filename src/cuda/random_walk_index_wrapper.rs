#![cfg(feature = "cuda")]

use crate::indicators::random_walk_index::{
    expand_grid, RandomWalkIndexBatchRange, RandomWalkIndexParams,
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

const RANDOM_WALK_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaRandomWalkIndexError {
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

pub struct RandomWalkIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl RandomWalkIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct RandomWalkIndexDeviceArrayF64Pair {
    pub high: RandomWalkIndexDeviceArrayF64,
    pub low: RandomWalkIndexDeviceArrayF64,
}

impl RandomWalkIndexDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.high.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.high.cols
    }
}

pub struct CudaRandomWalkIndexBatchResult {
    pub outputs: RandomWalkIndexDeviceArrayF64Pair,
    pub combos: Vec<RandomWalkIndexParams>,
}

pub struct CudaRandomWalkIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaRandomWalkIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaRandomWalkIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("random_walk_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaRandomWalkIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_hlc(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len())
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaRandomWalkIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaRandomWalkIndexError::OutOfMemory {
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
    ) -> Result<(), CudaRandomWalkIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaRandomWalkIndexError::LaunchConfigTooLarge {
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
        close: &[f64],
        sweep: &RandomWalkIndexBatchRange,
    ) -> Result<CudaRandomWalkIndexBatchResult, CudaRandomWalkIndexError> {
        let len = close.len();
        if len == 0 {
            return Err(CudaRandomWalkIndexError::InvalidInput("empty input".into()));
        }
        if high.len() != len || low.len() != len {
            return Err(CudaRandomWalkIndexError::InvalidInput(format!(
                "inconsistent slice lengths: high={}, low={}, close={len}",
                high.len(),
                low.len()
            )));
        }

        let combos = expand_grid(sweep)
            .map_err(|err| CudaRandomWalkIndexError::InvalidInput(err.to_string()))?;
        let first_valid = Self::first_valid_hlc(high, low, close)
            .ok_or_else(|| CudaRandomWalkIndexError::InvalidInput("all values are NaN".into()))?;
        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(14))
            .max()
            .unwrap_or(0);
        let valid = len.saturating_sub(first_valid);
        if valid < max_length {
            return Err(CudaRandomWalkIndexError::InvalidInput(format!(
                "not enough valid data: needed={max_length}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = len;
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(14))
            .map(|length| {
                if length == 0 || length > len {
                    Err(CudaRandomWalkIndexError::InvalidInput(format!(
                        "invalid length: length={length}, data_len={len}"
                    )))
                } else {
                    Ok(length as i32)
                }
            })
            .collect::<Result<_, _>>()?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| CudaRandomWalkIndexError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaRandomWalkIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaRandomWalkIndexError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaRandomWalkIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaRandomWalkIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_out_high = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_low = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("random_walk_index_batch_f64")
            .map_err(|_| CudaRandomWalkIndexError::MissingKernelSymbol {
                name: "random_walk_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + RANDOM_WALK_INDEX_BLOCK_X - 1) / RANDOM_WALK_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(RANDOM_WALK_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                first_valid as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                d_out_high.as_device_ptr(),
                d_out_low.as_device_ptr()
            ))?;
        }

        Ok(CudaRandomWalkIndexBatchResult {
            outputs: RandomWalkIndexDeviceArrayF64Pair {
                high: RandomWalkIndexDeviceArrayF64 {
                    buf: d_out_high,
                    rows,
                    cols,
                },
                low: RandomWalkIndexDeviceArrayF64 {
                    buf: d_out_low,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
