#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::psychological_line::{PsychologicalLineBatchRange, PsychologicalLineParams};
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

const PSYCHOLOGICAL_LINE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaPsychologicalLineError {
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

pub struct CudaPsychologicalLineBatchResult {
    pub outputs: DeviceArrayF32,
    pub combos: Vec<PsychologicalLineParams>,
}

pub struct CudaPsychologicalLine {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaPsychologicalLine {
    pub fn new(device_id: usize) -> Result<Self, CudaPsychologicalLineError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("psychological_line_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaPsychologicalLineError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaPsychologicalLineError> {
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
                if value < end + step {
                    break;
                }
                value = value.saturating_sub(step);
                if value == 0 {
                    break;
                }
            }
        }

        if out.is_empty() {
            return Err(CudaPsychologicalLineError::InvalidInput(format!(
                "invalid length range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &PsychologicalLineBatchRange,
    ) -> Result<Vec<PsychologicalLineParams>, CudaPsychologicalLineError> {
        Ok(Self::axis_usize(sweep.length)?
            .into_iter()
            .map(|length| PsychologicalLineParams {
                length: Some(length),
            })
            .collect())
    }

    fn first_valid_index(data: &[f32]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaPsychologicalLineError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaPsychologicalLineError::OutOfMemory {
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
    ) -> Result<(), CudaPsychologicalLineError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaPsychologicalLineError::LaunchConfigTooLarge {
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
        data_f32: &[f32],
        sweep: &PsychologicalLineBatchRange,
    ) -> Result<CudaPsychologicalLineBatchResult, CudaPsychologicalLineError> {
        if data_f32.is_empty() {
            return Err(CudaPsychologicalLineError::InvalidInput(
                "empty data".into(),
            ));
        }

        let first_valid = Self::first_valid_index(data_f32)
            .ok_or_else(|| CudaPsychologicalLineError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep)?;
        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(20))
            .max()
            .unwrap_or(0);
        if max_length == 0 {
            return Err(CudaPsychologicalLineError::InvalidInput(
                "length sweep produced no valid values".into(),
            ));
        }
        let valid = data_f32.len().saturating_sub(first_valid);
        if valid <= max_length {
            return Err(CudaPsychologicalLineError::InvalidInput(format!(
                "not enough valid data: needed={}, valid={valid}",
                max_length + 1
            )));
        }

        let rows = combos.len();
        let cols = data_f32.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(20) as i32)
            .collect();

        let input_bytes = data_f32
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaPsychologicalLineError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaPsychologicalLineError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaPsychologicalLineError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaPsychologicalLineError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaPsychologicalLineError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data_f32)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("psychological_line_batch_f32")
            .map_err(|_| CudaPsychologicalLineError::MissingKernelSymbol {
                name: "psychological_line_batch_f32",
            })?;
        let grid_x = ((rows as u32) + PSYCHOLOGICAL_LINE_BLOCK_X - 1) / PSYCHOLOGICAL_LINE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(PSYCHOLOGICAL_LINE_BLOCK_X);
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

        Ok(CudaPsychologicalLineBatchResult {
            outputs: DeviceArrayF32 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
