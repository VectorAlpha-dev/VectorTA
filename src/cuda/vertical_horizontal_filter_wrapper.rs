#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::vertical_horizontal_filter::{
    VerticalHorizontalFilterBatchRange, VerticalHorizontalFilterParams,
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

const VHF_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaVerticalHorizontalFilterError {
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

pub struct CudaVerticalHorizontalFilterBatchResult {
    pub outputs: DeviceArrayF32,
    pub combos: Vec<VerticalHorizontalFilterParams>,
}

pub struct CudaVerticalHorizontalFilter {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVerticalHorizontalFilter {
    pub fn new(device_id: usize) -> Result<Self, CudaVerticalHorizontalFilterError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("vertical_horizontal_filter_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVerticalHorizontalFilterError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn expand_grid(
        range: &VerticalHorizontalFilterBatchRange,
    ) -> Result<Vec<VerticalHorizontalFilterParams>, CudaVerticalHorizontalFilterError> {
        let (start, end, step) = range.length;
        let lengths = if start == end {
            vec![start]
        } else {
            if step == 0 {
                return Err(CudaVerticalHorizontalFilterError::InvalidInput(format!(
                    "invalid length range: start={start}, end={end}, step={step}"
                )));
            }
            let mut out = Vec::new();
            if start < end {
                let mut x = start;
                while x <= end {
                    out.push(x);
                    let next = x.saturating_add(step);
                    if next == x {
                        break;
                    }
                    x = next;
                }
            } else {
                let mut x = start;
                while x >= end {
                    out.push(x);
                    let next = x.saturating_sub(step);
                    if next == x || next < end {
                        break;
                    }
                    x = next;
                }
            }
            out
        };

        if lengths.is_empty() {
            return Err(CudaVerticalHorizontalFilterError::InvalidInput(format!(
                "invalid length range: start={start}, end={end}, step={step}"
            )));
        }

        Ok(lengths
            .into_iter()
            .map(|length| VerticalHorizontalFilterParams {
                length: Some(length),
            })
            .collect())
    }

    fn count_valid_changes(data: &[f32]) -> usize {
        if data.len() < 2 {
            return 0;
        }
        let mut count = 0usize;
        for i in 1..data.len() {
            if data[i - 1].is_finite() && data[i].is_finite() {
                count += 1;
            }
        }
        count
    }

    fn first_valid_value(data: &[f32]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaVerticalHorizontalFilterError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVerticalHorizontalFilterError::OutOfMemory {
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
    ) -> Result<(), CudaVerticalHorizontalFilterError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVerticalHorizontalFilterError::LaunchConfigTooLarge {
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
        sweep: &VerticalHorizontalFilterBatchRange,
    ) -> Result<CudaVerticalHorizontalFilterBatchResult, CudaVerticalHorizontalFilterError> {
        if data_f32.is_empty() {
            return Err(CudaVerticalHorizontalFilterError::InvalidInput(
                "empty data".into(),
            ));
        }
        Self::first_valid_value(data_f32).ok_or_else(|| {
            CudaVerticalHorizontalFilterError::InvalidInput("all values are NaN".into())
        })?;

        let combos = Self::expand_grid(sweep)?;
        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(28))
            .max()
            .unwrap_or(0);
        let valid = Self::count_valid_changes(data_f32);
        if max_length == 0 || valid < max_length {
            return Err(CudaVerticalHorizontalFilterError::InvalidInput(format!(
                "not enough valid changes: needed={max_length}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = data_f32.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(28) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaVerticalHorizontalFilterError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaVerticalHorizontalFilterError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVerticalHorizontalFilterError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                CudaVerticalHorizontalFilterError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVerticalHorizontalFilterError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data_f32)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("vertical_horizontal_filter_batch_f32")
            .map_err(|_| CudaVerticalHorizontalFilterError::MissingKernelSymbol {
                name: "vertical_horizontal_filter_batch_f32",
            })?;
        let grid_x = ((rows as u32) + VHF_BLOCK_X - 1) / VHF_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VHF_BLOCK_X);
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

        Ok(CudaVerticalHorizontalFilterBatchResult {
            outputs: DeviceArrayF32 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
