#![cfg(feature = "cuda")]

use crate::indicators::leavitt_convolution_acceleration::{
    expand_grid_leavitt_convolution_acceleration, LeavittConvolutionAccelerationBatchRange,
    LeavittConvolutionAccelerationParams,
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

const LEAVITT_CONVOLUTION_ACCELERATION_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 70;
const DEFAULT_NORM_LENGTH: usize = 150;

#[derive(Debug, Error)]
pub enum CudaLeavittConvolutionAccelerationError {
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

pub struct LeavittConvolutionAccelerationDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl LeavittConvolutionAccelerationDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct LeavittConvolutionAccelerationDeviceArrayF64Pair {
    pub conv_acceleration: LeavittConvolutionAccelerationDeviceArrayF64,
    pub signal: LeavittConvolutionAccelerationDeviceArrayF64,
}

impl LeavittConvolutionAccelerationDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.conv_acceleration.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.conv_acceleration.cols
    }
}

pub struct CudaLeavittConvolutionAccelerationBatchResult {
    pub outputs: LeavittConvolutionAccelerationDeviceArrayF64Pair,
    pub combos: Vec<LeavittConvolutionAccelerationParams>,
}

pub struct CudaLeavittConvolutionAcceleration {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn sqrt_length(length: usize) -> usize {
    ((length as f64).sqrt().floor() as usize).max(1)
}

impl CudaLeavittConvolutionAcceleration {
    pub fn new(device_id: usize) -> Result<Self, CudaLeavittConvolutionAccelerationError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("leavitt_convolution_acceleration_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaLeavittConvolutionAccelerationError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaLeavittConvolutionAccelerationError> {
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
                CudaLeavittConvolutionAccelerationError::LaunchConfigTooLarge {
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

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaLeavittConvolutionAccelerationError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaLeavittConvolutionAccelerationError::OutOfMemory {
                    required,
                    free,
                    headroom,
                });
            }
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        sweep: &LeavittConvolutionAccelerationBatchRange,
    ) -> Result<
        CudaLeavittConvolutionAccelerationBatchResult,
        CudaLeavittConvolutionAccelerationError,
    > {
        if data.is_empty() {
            return Err(CudaLeavittConvolutionAccelerationError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = expand_grid_leavitt_convolution_acceleration(sweep).map_err(|err| {
            CudaLeavittConvolutionAccelerationError::InvalidInput(err.to_string())
        })?;
        if combos.is_empty() {
            return Err(CudaLeavittConvolutionAccelerationError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let first = data
            .iter()
            .position(|value| value.is_finite())
            .ok_or_else(|| {
                CudaLeavittConvolutionAccelerationError::InvalidInput("all values are NaN".into())
            })?;
        let valid = data[first..]
            .iter()
            .filter(|value| value.is_finite())
            .count();

        let rows = combos.len();
        let cols = data.len();
        let mut max_length = 0usize;
        let mut max_norm_length = 0usize;
        let mut use_norm_hyperbolic = Vec::with_capacity(rows);
        let mut lengths = Vec::with_capacity(rows);
        let mut norm_lengths = Vec::with_capacity(rows);

        for params in &combos {
            let length = params.length.unwrap_or(DEFAULT_LENGTH);
            let norm_length = params.norm_length.unwrap_or(DEFAULT_NORM_LENGTH);
            let needed = length + sqrt_length(length) + norm_length - 2;
            if valid < needed {
                return Err(CudaLeavittConvolutionAccelerationError::InvalidInput(
                    format!("not enough valid data: needed={needed}, valid={valid}"),
                ));
            }
            max_length = max_length.max(length);
            max_norm_length = max_norm_length.max(norm_length);
            lengths.push(length as i32);
            norm_lengths.push(norm_length as i32);
            use_norm_hyperbolic.push(i32::from(params.use_norm_hyperbolic.unwrap_or(true)));
        }

        let max_sqrt_length = sqrt_length(max_length);
        let scratch_cap = max_length
            .checked_add(max_sqrt_length)
            .and_then(|value| value.checked_add(max_norm_length))
            .ok_or_else(|| {
                CudaLeavittConvolutionAccelerationError::InvalidInput(
                    "scratch capacity overflow".into(),
                )
            })?;

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaLeavittConvolutionAccelerationError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaLeavittConvolutionAccelerationError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaLeavittConvolutionAccelerationError::InvalidInput("param bytes overflow".into())
            })?;
        let scratch_bytes = output_elems
            .checked_div(cols.max(1))
            .and_then(|_| rows.checked_mul(scratch_cap))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                CudaLeavittConvolutionAccelerationError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaLeavittConvolutionAccelerationError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaLeavittConvolutionAccelerationError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_norm_lengths = DeviceBuffer::from_slice(&norm_lengths)?;
        let d_use_norm_hyperbolic = DeviceBuffer::from_slice(&use_norm_hyperbolic)?;
        let mut d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(rows * scratch_cap)? };
        let mut d_conv_acceleration = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("leavitt_convolution_acceleration_batch_f64")
            .map_err(
                |_| CudaLeavittConvolutionAccelerationError::MissingKernelSymbol {
                    name: "leavitt_convolution_acceleration_batch_f64",
                },
            )?;

        let grid_x = ((rows as u32) + LEAVITT_CONVOLUTION_ACCELERATION_BLOCK_X - 1)
            / LEAVITT_CONVOLUTION_ACCELERATION_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(LEAVITT_CONVOLUTION_ACCELERATION_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_norm_lengths.as_device_ptr(),
                d_use_norm_hyperbolic.as_device_ptr(),
                rows as i32,
                scratch_cap as i32,
                d_scratch.as_device_ptr(),
                d_conv_acceleration.as_device_ptr(),
                d_signal.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaLeavittConvolutionAccelerationBatchResult {
            outputs: LeavittConvolutionAccelerationDeviceArrayF64Pair {
                conv_acceleration: LeavittConvolutionAccelerationDeviceArrayF64 {
                    buf: d_conv_acceleration,
                    rows,
                    cols,
                },
                signal: LeavittConvolutionAccelerationDeviceArrayF64 {
                    buf: d_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
