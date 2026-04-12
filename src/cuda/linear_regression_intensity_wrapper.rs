#![cfg(feature = "cuda")]

use crate::indicators::linear_regression_intensity::{
    expand_grid_linear_regression_intensity, LinearRegressionIntensityBatchRange,
    LinearRegressionIntensityParams,
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

const LINEAR_REGRESSION_INTENSITY_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaLinearRegressionIntensityError {
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

pub struct LinearRegressionIntensityDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl LinearRegressionIntensityDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaLinearRegressionIntensityBatchResult {
    pub outputs: LinearRegressionIntensityDeviceArrayF64,
    pub combos: Vec<LinearRegressionIntensityParams>,
}

pub struct CudaLinearRegressionIntensity {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaLinearRegressionIntensity {
    pub fn new(device_id: usize) -> Result<Self, CudaLinearRegressionIntensityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("linear_regression_intensity_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaLinearRegressionIntensityError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_source(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaLinearRegressionIntensityError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaLinearRegressionIntensityError::OutOfMemory {
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
    ) -> Result<(), CudaLinearRegressionIntensityError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaLinearRegressionIntensityError::LaunchConfigTooLarge {
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
        sweep: &LinearRegressionIntensityBatchRange,
    ) -> Result<CudaLinearRegressionIntensityBatchResult, CudaLinearRegressionIntensityError> {
        if data.is_empty() {
            return Err(CudaLinearRegressionIntensityError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = expand_grid_linear_regression_intensity(sweep)
            .map_err(|err| CudaLinearRegressionIntensityError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaLinearRegressionIntensityError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let first = Self::first_valid_source(data).ok_or_else(|| {
            CudaLinearRegressionIntensityError::InvalidInput("all values are NaN".into())
        })?;
        let valid = data.len() - first;

        let mut max_lookback_period = 0usize;
        let mut max_linreg_length = 0usize;
        let mut lookback_periods = Vec::with_capacity(combos.len());
        let mut linreg_lengths = Vec::with_capacity(combos.len());
        for combo in &combos {
            let lookback_period = combo.lookback_period.unwrap_or(12);
            let range_tolerance = combo.range_tolerance.unwrap_or(90.0);
            let linreg_length = combo.linreg_length.unwrap_or(90);
            if lookback_period == 0 || lookback_period > data.len() {
                return Err(CudaLinearRegressionIntensityError::InvalidInput(format!(
                    "invalid lookback_period: lookback_period={lookback_period}, data_len={}",
                    data.len()
                )));
            }
            if !range_tolerance.is_finite() || !(0.0..=100.0).contains(&range_tolerance) {
                return Err(CudaLinearRegressionIntensityError::InvalidInput(format!(
                    "invalid range_tolerance: range_tolerance={range_tolerance}"
                )));
            }
            if linreg_length == 0 || linreg_length > data.len() {
                return Err(CudaLinearRegressionIntensityError::InvalidInput(format!(
                    "invalid linreg_length: linreg_length={linreg_length}, data_len={}",
                    data.len()
                )));
            }
            let needed = linreg_length
                .saturating_add(lookback_period)
                .saturating_sub(1);
            if valid < needed {
                return Err(CudaLinearRegressionIntensityError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            max_lookback_period = max_lookback_period.max(lookback_period);
            max_linreg_length = max_linreg_length.max(linreg_length);
            lookback_periods.push(lookback_period as i32);
            linreg_lengths.push(linreg_length as i32);
        }

        let rows = combos.len();
        let cols = data.len();
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaLinearRegressionIntensityError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lookback_periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                linreg_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaLinearRegressionIntensityError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaLinearRegressionIntensityError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaLinearRegressionIntensityError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_elems = rows
            .checked_mul(max_linreg_length)
            .and_then(|value| {
                rows.checked_mul(max_lookback_period)
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaLinearRegressionIntensityError::InvalidInput("scratch elems overflow".into())
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaLinearRegressionIntensityError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaLinearRegressionIntensityError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lookback_periods = DeviceBuffer::from_slice(&lookback_periods)?;
        let d_linreg_lengths = DeviceBuffer::from_slice(&linreg_lengths)?;
        let d_linreg_input =
            unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_linreg_length)? };
        let d_linreg_window =
            unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_lookback_period)? };
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("linear_regression_intensity_batch_f64")
            .map_err(
                |_| CudaLinearRegressionIntensityError::MissingKernelSymbol {
                    name: "linear_regression_intensity_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + LINEAR_REGRESSION_INTENSITY_BLOCK_X - 1)
            / LINEAR_REGRESSION_INTENSITY_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(LINEAR_REGRESSION_INTENSITY_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lookback_periods.as_device_ptr(),
                d_linreg_lengths.as_device_ptr(),
                rows as i32,
                max_lookback_period as i32,
                max_linreg_length as i32,
                d_linreg_input.as_device_ptr(),
                d_linreg_window.as_device_ptr(),
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaLinearRegressionIntensityBatchResult {
            outputs: LinearRegressionIntensityDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
