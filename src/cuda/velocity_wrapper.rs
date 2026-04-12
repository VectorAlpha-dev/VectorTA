#![cfg(feature = "cuda")]

use crate::indicators::velocity::{VelocityBatchRange, VelocityParams};
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

const VELOCITY_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 21;
const DEFAULT_SMOOTH_LENGTH: usize = 5;
const MIN_LENGTH: usize = 2;
const MAX_LENGTH: usize = 60;
const MIN_SMOOTH_LENGTH: usize = 1;
const MAX_SMOOTH_LENGTH: usize = 9;

#[derive(Debug, Error)]
pub enum CudaVelocityError {
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

pub struct VelocityDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VelocityDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaVelocityBatchResult {
    pub outputs: VelocityDeviceArrayF64,
    pub combos: Vec<VelocityParams>,
}

pub struct CudaVelocity {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVelocity {
    pub fn new(device_id: usize) -> Result<Self, CudaVelocityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("velocity_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVelocityError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn validate_params(length: usize, smooth_length: usize) -> Result<(), CudaVelocityError> {
        if !(MIN_LENGTH..=MAX_LENGTH).contains(&length) {
            return Err(CudaVelocityError::InvalidInput(format!(
                "invalid length: {length}. Expected 2..=60."
            )));
        }
        if !(MIN_SMOOTH_LENGTH..=MAX_SMOOTH_LENGTH).contains(&smooth_length) {
            return Err(CudaVelocityError::InvalidInput(format!(
                "invalid smoothing length: {smooth_length}. Expected 1..=9."
            )));
        }
        Ok(())
    }

    fn expand_axis(
        (start, end, step): (usize, usize, usize),
        is_smooth: bool,
    ) -> Result<Vec<usize>, CudaVelocityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut values = Vec::new();
        if start < end {
            let mut current = start;
            while current <= end {
                values.push(current);
                match current.checked_add(step) {
                    Some(next) if next > current => current = next,
                    _ => break,
                }
            }
        } else {
            let mut current = start;
            while current >= end {
                values.push(current);
                if current < end.saturating_add(step) {
                    break;
                }
                current = current.saturating_sub(step);
            }
        }

        if values.is_empty() {
            return Err(CudaVelocityError::InvalidInput(if is_smooth {
                format!("invalid smoothing length range: start={start}, end={end}, step={step}")
            } else {
                format!("invalid length range: start={start}, end={end}, step={step}")
            }));
        }

        Ok(values)
    }

    fn expand_grid(sweep: &VelocityBatchRange) -> Result<Vec<VelocityParams>, CudaVelocityError> {
        let lengths = Self::expand_axis(sweep.length, false)?;
        let smooth_lengths = Self::expand_axis(sweep.smooth_length, true)?;
        let mut combos = Vec::with_capacity(lengths.len().saturating_mul(smooth_lengths.len()));
        for &length in &lengths {
            for &smooth_length in &smooth_lengths {
                Self::validate_params(length, smooth_length)?;
                combos.push(VelocityParams {
                    length: Some(length),
                    smooth_length: Some(smooth_length),
                });
            }
        }
        Ok(combos)
    }

    fn first_valid_index(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| !value.is_nan())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaVelocityError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVelocityError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    fn validate_launch(&self, grid: GridSize, block: BlockSize) -> Result<(), CudaVelocityError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVelocityError::LaunchConfigTooLarge {
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
        sweep: &VelocityBatchRange,
    ) -> Result<CudaVelocityBatchResult, CudaVelocityError> {
        if data.is_empty() {
            return Err(CudaVelocityError::InvalidInput("empty data".into()));
        }

        let combos = Self::expand_grid(sweep)?;
        let first_valid = Self::first_valid_index(data)
            .ok_or_else(|| CudaVelocityError::InvalidInput("all values are NaN".into()))?;
        let valid = data.len() - first_valid;
        let max_smooth = combos
            .iter()
            .map(|combo| combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH))
            .max()
            .unwrap_or(DEFAULT_SMOOTH_LENGTH);
        if valid < max_smooth {
            return Err(CudaVelocityError::InvalidInput(format!(
                "not enough valid data: needed={max_smooth}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as i32)
            .collect();
        let smooth_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaVelocityError::InvalidInput("input bytes overflow".into()))?;
        let length_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaVelocityError::InvalidInput("length bytes overflow".into()))?;
        let smooth_length_bytes = smooth_lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaVelocityError::InvalidInput("smooth length bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaVelocityError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaVelocityError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(length_bytes)
            .and_then(|v| v.checked_add(smooth_length_bytes))
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| CudaVelocityError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_smooth_lengths = DeviceBuffer::from_slice(&smooth_lengths)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("velocity_batch_f64")
            .map_err(|_| CudaVelocityError::MissingKernelSymbol {
                name: "velocity_batch_f64",
            })?;
        let grid_x = ((rows as u32) + VELOCITY_BLOCK_X - 1) / VELOCITY_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VELOCITY_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_smooth_lengths.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVelocityBatchResult {
            outputs: VelocityDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
