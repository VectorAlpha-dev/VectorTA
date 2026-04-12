#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::gopalakrishnan_range_index::{
    GopalakrishnanRangeIndexBatchRange, GopalakrishnanRangeIndexParams,
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

const GAPO_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaGopalakrishnanRangeIndexError {
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

pub struct CudaGopalakrishnanRangeIndexBatchResult {
    pub outputs: DeviceArrayF32,
    pub combos: Vec<GopalakrishnanRangeIndexParams>,
}

pub struct CudaGopalakrishnanRangeIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaGopalakrishnanRangeIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaGopalakrishnanRangeIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("gopalakrishnan_range_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaGopalakrishnanRangeIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn expand_grid(
        range: &GopalakrishnanRangeIndexBatchRange,
    ) -> Result<Vec<GopalakrishnanRangeIndexParams>, CudaGopalakrishnanRangeIndexError> {
        let (start, end, step) = range.length;
        let lengths = if step == 0 || start == end {
            vec![start]
        } else if start < end {
            let mut out = Vec::new();
            let mut x = start;
            let st = step.max(1);
            while x <= end {
                out.push(x);
                let next = x.saturating_add(st);
                if next == x {
                    break;
                }
                x = next;
            }
            out
        } else {
            let mut out = Vec::new();
            let mut x = start;
            let st = step.max(1);
            while x >= end {
                out.push(x);
                let next = x.saturating_sub(st);
                if next == x || next < end {
                    break;
                }
                x = next;
            }
            out
        };

        if lengths.is_empty() {
            return Err(CudaGopalakrishnanRangeIndexError::InvalidInput(format!(
                "invalid length range: start={start}, end={end}, step={step}"
            )));
        }

        Ok(lengths
            .into_iter()
            .map(|length| GopalakrishnanRangeIndexParams {
                length: Some(length),
            })
            .collect())
    }

    fn first_valid_high_low(high: &[f32], low: &[f32]) -> Option<usize> {
        (0..high.len()).find(|&i| high[i].is_finite() && low[i].is_finite())
    }

    fn count_valid_high_low(high: &[f32], low: &[f32]) -> usize {
        let mut count = 0usize;
        for i in 0..high.len() {
            if high[i].is_finite() && low[i].is_finite() {
                count += 1;
            }
        }
        count
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaGopalakrishnanRangeIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaGopalakrishnanRangeIndexError::OutOfMemory {
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
    ) -> Result<(), CudaGopalakrishnanRangeIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaGopalakrishnanRangeIndexError::LaunchConfigTooLarge {
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
        high_f32: &[f32],
        low_f32: &[f32],
        sweep: &GopalakrishnanRangeIndexBatchRange,
    ) -> Result<CudaGopalakrishnanRangeIndexBatchResult, CudaGopalakrishnanRangeIndexError> {
        if high_f32.is_empty() || low_f32.is_empty() {
            return Err(CudaGopalakrishnanRangeIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high_f32.len() != low_f32.len() {
            return Err(CudaGopalakrishnanRangeIndexError::InvalidInput(
                "high/low length mismatch".into(),
            ));
        }
        Self::first_valid_high_low(high_f32, low_f32).ok_or_else(|| {
            CudaGopalakrishnanRangeIndexError::InvalidInput("all values are NaN".into())
        })?;

        let combos = Self::expand_grid(sweep)?;
        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(5))
            .max()
            .unwrap_or(0);
        let valid = Self::count_valid_high_low(high_f32, low_f32);
        if max_length <= 1 || valid < max_length {
            return Err(CudaGopalakrishnanRangeIndexError::InvalidInput(format!(
                "not enough valid data: needed={max_length}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = high_f32.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(5) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaGopalakrishnanRangeIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaGopalakrishnanRangeIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaGopalakrishnanRangeIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaGopalakrishnanRangeIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaGopalakrishnanRangeIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high_f32)?;
        let d_low = DeviceBuffer::from_slice(low_f32)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("gopalakrishnan_range_index_batch_f32")
            .map_err(|_| CudaGopalakrishnanRangeIndexError::MissingKernelSymbol {
                name: "gopalakrishnan_range_index_batch_f32",
            })?;
        let grid_x = ((rows as u32) + GAPO_BLOCK_X - 1) / GAPO_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(GAPO_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaGopalakrishnanRangeIndexBatchResult {
            outputs: DeviceArrayF32 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
