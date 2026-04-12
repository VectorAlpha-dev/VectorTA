#![cfg(feature = "cuda")]

use crate::indicators::fractal_dimension_index::{
    FractalDimensionIndexBatchRange, FractalDimensionIndexParams,
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

const FRACTAL_DIMENSION_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaFractalDimensionIndexError {
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

pub struct FractalDimensionIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl FractalDimensionIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaFractalDimensionIndexBatchResult {
    pub outputs: FractalDimensionIndexDeviceArrayF64,
    pub combos: Vec<FractalDimensionIndexParams>,
}

pub struct CudaFractalDimensionIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaFractalDimensionIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaFractalDimensionIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("fractal_dimension_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaFractalDimensionIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn longest_valid_run(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for &value in data {
            if value.is_finite() {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        best
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaFractalDimensionIndexError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut cur = start;
            loop {
                out.push(cur);
                if cur == end {
                    break;
                }
                let next = cur.saturating_add(step.max(1));
                if next == cur || next > end {
                    break;
                }
                cur = next;
            }
        } else {
            let mut cur = start;
            loop {
                out.push(cur);
                if cur == end {
                    break;
                }
                let next = cur.saturating_sub(step.max(1));
                if next == cur || next < end {
                    break;
                }
                cur = next;
            }
        }

        if out.is_empty() {
            return Err(CudaFractalDimensionIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &FractalDimensionIndexBatchRange,
    ) -> Result<Vec<FractalDimensionIndexParams>, CudaFractalDimensionIndexError> {
        let lengths = Self::axis_usize(sweep.length)?;
        if lengths.iter().any(|&length| length < 2) {
            return Err(CudaFractalDimensionIndexError::InvalidInput(
                "invalid length: length must be >= 2".into(),
            ));
        }
        Ok(lengths
            .into_iter()
            .map(|length| FractalDimensionIndexParams {
                length: Some(length),
            })
            .collect())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaFractalDimensionIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaFractalDimensionIndexError::OutOfMemory {
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
    ) -> Result<(), CudaFractalDimensionIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaFractalDimensionIndexError::LaunchConfigTooLarge {
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
        sweep: &FractalDimensionIndexBatchRange,
    ) -> Result<CudaFractalDimensionIndexBatchResult, CudaFractalDimensionIndexError> {
        if data.is_empty() {
            return Err(CudaFractalDimensionIndexError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let len = data.len();
        let max_run = Self::longest_valid_run(data);
        if max_run == 0 {
            return Err(CudaFractalDimensionIndexError::InvalidInput(
                "all values are NaN".into(),
            ));
        }
        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(30))
            .max()
            .unwrap_or(0);
        if max_length > len {
            return Err(CudaFractalDimensionIndexError::InvalidInput(format!(
                "invalid length: length={max_length}, data_len={len}"
            )));
        }
        if max_run < max_length {
            return Err(CudaFractalDimensionIndexError::InvalidInput(format!(
                "not enough valid data: needed={max_length}, valid={max_run}"
            )));
        }

        let rows = combos.len();
        let cols = len;
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(30) as i32)
            .collect();
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaFractalDimensionIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaFractalDimensionIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaFractalDimensionIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaFractalDimensionIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaFractalDimensionIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("fractal_dimension_index_batch_f64")
            .map_err(|_| CudaFractalDimensionIndexError::MissingKernelSymbol {
                name: "fractal_dimension_index_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + FRACTAL_DIMENSION_INDEX_BLOCK_X - 1) / FRACTAL_DIMENSION_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(FRACTAL_DIMENSION_INDEX_BLOCK_X);
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

        Ok(CudaFractalDimensionIndexBatchResult {
            outputs: FractalDimensionIndexDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
