#![cfg(feature = "cuda")]

use crate::indicators::linear_correlation_oscillator::{
    expand_grid, LinearCorrelationOscillatorBatchRange, LinearCorrelationOscillatorParams,
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

const LINEAR_CORRELATION_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaLinearCorrelationOscillatorError {
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

pub struct LinearCorrelationOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl LinearCorrelationOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaLinearCorrelationOscillatorBatchResult {
    pub outputs: LinearCorrelationOscillatorDeviceArrayF64,
    pub combos: Vec<LinearCorrelationOscillatorParams>,
}

pub struct CudaLinearCorrelationOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaLinearCorrelationOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaLinearCorrelationOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("linear_correlation_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaLinearCorrelationOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_source(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| !value.is_nan())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaLinearCorrelationOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaLinearCorrelationOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaLinearCorrelationOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaLinearCorrelationOscillatorError::LaunchConfigTooLarge {
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
        sweep: &LinearCorrelationOscillatorBatchRange,
    ) -> Result<CudaLinearCorrelationOscillatorBatchResult, CudaLinearCorrelationOscillatorError>
    {
        if data.is_empty() {
            return Err(CudaLinearCorrelationOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = expand_grid(sweep)
            .map_err(|err| CudaLinearCorrelationOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaLinearCorrelationOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let first = Self::first_valid_source(data).ok_or_else(|| {
            CudaLinearCorrelationOscillatorError::InvalidInput("all values are NaN".into())
        })?;
        let max_period = combos
            .iter()
            .map(|params| params.period.unwrap_or(14))
            .max()
            .unwrap_or(0);
        if max_period == 0 || max_period > data.len() {
            return Err(CudaLinearCorrelationOscillatorError::InvalidInput(format!(
                "invalid period: period={max_period}, data_len={}",
                data.len()
            )));
        }

        let valid = data.len() - first;
        if valid <= max_period + 1 {
            return Err(CudaLinearCorrelationOscillatorError::InvalidInput(format!(
                "not enough valid data: needed={}, valid={valid}",
                max_period + 2
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let periods: Vec<i32> = combos
            .iter()
            .map(|params| params.period.unwrap_or(14) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaLinearCorrelationOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let period_bytes = periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaLinearCorrelationOscillatorError::InvalidInput("period bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaLinearCorrelationOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaLinearCorrelationOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(period_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaLinearCorrelationOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("linear_correlation_oscillator_batch_f64")
            .map_err(
                |_| CudaLinearCorrelationOscillatorError::MissingKernelSymbol {
                    name: "linear_correlation_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + LINEAR_CORRELATION_OSCILLATOR_BLOCK_X - 1)
            / LINEAR_CORRELATION_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(LINEAR_CORRELATION_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaLinearCorrelationOscillatorBatchResult {
            outputs: LinearCorrelationOscillatorDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
