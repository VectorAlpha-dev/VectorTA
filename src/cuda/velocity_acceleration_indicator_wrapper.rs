#![cfg(feature = "cuda")]

use crate::indicators::velocity_acceleration_indicator::{
    VelocityAccelerationIndicatorBatchRange, VelocityAccelerationIndicatorParams,
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

const VELOCITY_ACCELERATION_INDICATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 21;
const DEFAULT_SMOOTH_LENGTH: usize = 5;

#[derive(Debug, Error)]
pub enum CudaVelocityAccelerationIndicatorError {
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

pub struct VelocityAccelerationIndicatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VelocityAccelerationIndicatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaVelocityAccelerationIndicatorBatchResult {
    pub outputs: VelocityAccelerationIndicatorDeviceArrayF64,
    pub combos: Vec<VelocityAccelerationIndicatorParams>,
}

pub struct CudaVelocityAccelerationIndicator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVelocityAccelerationIndicator {
    pub fn new(device_id: usize) -> Result<Self, CudaVelocityAccelerationIndicatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("velocity_acceleration_indicator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVelocityAccelerationIndicatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn expand_axis(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaVelocityAccelerationIndicatorError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            while value <= end {
                out.push(value);
                let next = value.saturating_add(step);
                if next == value {
                    break;
                }
                value = next;
            }
        } else {
            let mut value = start;
            loop {
                out.push(value);
                if value == end {
                    break;
                }
                let next = value.saturating_sub(step);
                if next == value || next < end {
                    break;
                }
                value = next;
            }
        }

        if out.is_empty() {
            return Err(CudaVelocityAccelerationIndicatorError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &VelocityAccelerationIndicatorBatchRange,
    ) -> Result<Vec<VelocityAccelerationIndicatorParams>, CudaVelocityAccelerationIndicatorError>
    {
        let lengths = Self::expand_axis(sweep.length)?;
        let smooth_lengths = Self::expand_axis(sweep.smooth_length)?;
        let mut combos = Vec::with_capacity(lengths.len().saturating_mul(smooth_lengths.len()));
        for length in lengths {
            if length < 2 {
                return Err(CudaVelocityAccelerationIndicatorError::InvalidInput(
                    format!("invalid length: {length}"),
                ));
            }
            for smooth_length in smooth_lengths.iter().copied() {
                if smooth_length == 0 {
                    return Err(CudaVelocityAccelerationIndicatorError::InvalidInput(
                        "invalid smooth_length: 0".into(),
                    ));
                }
                combos.push(VelocityAccelerationIndicatorParams {
                    length: Some(length),
                    smooth_length: Some(smooth_length),
                });
            }
        }
        Ok(combos)
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn max_consecutive_valid_values(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut run = 0usize;
        for &value in data {
            if value.is_finite() {
                run += 1;
                if run > best {
                    best = run;
                }
            } else {
                run = 0;
            }
        }
        best
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaVelocityAccelerationIndicatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVelocityAccelerationIndicatorError::OutOfMemory {
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
    ) -> Result<(), CudaVelocityAccelerationIndicatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(
                CudaVelocityAccelerationIndicatorError::LaunchConfigTooLarge {
                    gx: grid.x,
                    gy: grid.y,
                    gz: grid.z,
                    bx: block.x,
                    by: block.y,
                    bz: block.z,
                },
            );
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &VelocityAccelerationIndicatorBatchRange,
    ) -> Result<CudaVelocityAccelerationIndicatorBatchResult, CudaVelocityAccelerationIndicatorError>
    {
        if data.is_empty() {
            return Err(CudaVelocityAccelerationIndicatorError::InvalidInput(
                "empty input".into(),
            ));
        }

        Self::first_valid_value(data).ok_or_else(|| {
            CudaVelocityAccelerationIndicatorError::InvalidInput("all values are NaN".into())
        })?;
        let combos = Self::expand_grid(sweep)?;
        let max_smooth_length = combos
            .iter()
            .map(|combo| combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH))
            .max()
            .unwrap_or(DEFAULT_SMOOTH_LENGTH);
        let valid = Self::max_consecutive_valid_values(data);
        if valid < max_smooth_length {
            return Err(CudaVelocityAccelerationIndicatorError::InvalidInput(
                format!("not enough valid data: needed={max_smooth_length}, valid={valid}"),
            ));
        }

        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .max()
            .unwrap_or(DEFAULT_LENGTH);
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
            .ok_or_else(|| {
                CudaVelocityAccelerationIndicatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                smooth_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaVelocityAccelerationIndicatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVelocityAccelerationIndicatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaVelocityAccelerationIndicatorError::InvalidInput("output bytes overflow".into())
            })?;
        let history_len = rows.checked_mul(max_length).ok_or_else(|| {
            CudaVelocityAccelerationIndicatorError::InvalidInput("rows*max_length overflow".into())
        })?;
        let wma_len = rows.checked_mul(max_smooth_length).ok_or_else(|| {
            CudaVelocityAccelerationIndicatorError::InvalidInput(
                "rows*max_smooth_length overflow".into(),
            )
        })?;
        let scratch_bytes = history_len
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| {
                wma_len
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaVelocityAccelerationIndicatorError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaVelocityAccelerationIndicatorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_smooth_lengths = DeviceBuffer::from_slice(&smooth_lengths)?;
        let mut d_source_histories = unsafe { DeviceBuffer::<f64>::uninitialized(history_len)? };
        let mut d_acceleration_histories =
            unsafe { DeviceBuffer::<f64>::uninitialized(history_len)? };
        let mut d_wma_values = unsafe { DeviceBuffer::<f64>::uninitialized(wma_len)? };
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("velocity_acceleration_indicator_batch_f64")
            .map_err(
                |_| CudaVelocityAccelerationIndicatorError::MissingKernelSymbol {
                    name: "velocity_acceleration_indicator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + VELOCITY_ACCELERATION_INDICATOR_BLOCK_X - 1)
            / VELOCITY_ACCELERATION_INDICATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VELOCITY_ACCELERATION_INDICATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_smooth_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                max_smooth_length as i32,
                d_source_histories.as_device_ptr(),
                d_acceleration_histories.as_device_ptr(),
                d_wma_values.as_device_ptr(),
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVelocityAccelerationIndicatorBatchResult {
            outputs: VelocityAccelerationIndicatorDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
