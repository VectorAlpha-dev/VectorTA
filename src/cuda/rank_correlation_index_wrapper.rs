#![cfg(feature = "cuda")]

use crate::indicators::rank_correlation_index::{
    RankCorrelationIndexBatchRange, RankCorrelationIndexParams,
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

const RANK_CORRELATION_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaRankCorrelationIndexError {
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

pub struct RankCorrelationIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl RankCorrelationIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaRankCorrelationIndexBatchResult {
    pub outputs: RankCorrelationIndexDeviceArrayF64,
    pub combos: Vec<RankCorrelationIndexParams>,
}

pub struct CudaRankCorrelationIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaRankCorrelationIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaRankCorrelationIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("rank_correlation_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaRankCorrelationIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaRankCorrelationIndexError> {
        if start < 2 || end < 2 {
            return Err(CudaRankCorrelationIndexError::InvalidInput(format!(
                "invalid length range: start={start}, end={end}, step={step}"
            )));
        }
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            while value <= end {
                out.push(value);
                match value.checked_add(step) {
                    Some(next) if next > value => value = next,
                    _ => break,
                }
            }
        } else {
            let mut value = start;
            while value >= end {
                out.push(value);
                if value < end.saturating_add(step) {
                    break;
                }
                value = value.saturating_sub(step);
                if value == 0 {
                    break;
                }
            }
        }

        if out.is_empty() {
            return Err(CudaRankCorrelationIndexError::InvalidInput(format!(
                "invalid length range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &RankCorrelationIndexBatchRange,
    ) -> Result<Vec<RankCorrelationIndexParams>, CudaRankCorrelationIndexError> {
        Ok(Self::axis_usize(sweep.length)?
            .into_iter()
            .map(|length| RankCorrelationIndexParams {
                length: Some(length),
            })
            .collect())
    }

    fn first_valid_index(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaRankCorrelationIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaRankCorrelationIndexError::OutOfMemory {
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
    ) -> Result<(), CudaRankCorrelationIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaRankCorrelationIndexError::LaunchConfigTooLarge {
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
        sweep: &RankCorrelationIndexBatchRange,
    ) -> Result<CudaRankCorrelationIndexBatchResult, CudaRankCorrelationIndexError> {
        if data.is_empty() {
            return Err(CudaRankCorrelationIndexError::InvalidInput(
                "empty data".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let first = Self::first_valid_index(data).ok_or_else(|| {
            CudaRankCorrelationIndexError::InvalidInput("all values are NaN".into())
        })?;
        let max_length = combos
            .iter()
            .map(|params| params.length.unwrap_or(12))
            .max()
            .unwrap_or(12);
        if max_length > data.len() {
            return Err(CudaRankCorrelationIndexError::InvalidInput(format!(
                "invalid length: length={max_length}, data_len={}",
                data.len()
            )));
        }
        let valid = data.len() - first;
        if valid < max_length {
            return Err(CudaRankCorrelationIndexError::InvalidInput(format!(
                "not enough valid data: needed={max_length}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.length.unwrap_or(12) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaRankCorrelationIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaRankCorrelationIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaRankCorrelationIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaRankCorrelationIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaRankCorrelationIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("rank_correlation_index_batch_f64")
            .map_err(|_| CudaRankCorrelationIndexError::MissingKernelSymbol {
                name: "rank_correlation_index_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + RANK_CORRELATION_INDEX_BLOCK_X - 1) / RANK_CORRELATION_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(RANK_CORRELATION_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaRankCorrelationIndexBatchResult {
            outputs: RankCorrelationIndexDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
