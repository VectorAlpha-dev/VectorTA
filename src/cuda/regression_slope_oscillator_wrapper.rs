#![cfg(feature = "cuda")]

use crate::indicators::regression_slope_oscillator::{
    regression_slope_oscillator_expand_grid, RegressionSlopeOscillatorBatchRange,
    RegressionSlopeOscillatorParams,
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

const REGRESSION_SLOPE_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaRegressionSlopeOscillatorError {
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

pub struct RegressionSlopeOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl RegressionSlopeOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct RegressionSlopeOscillatorDeviceArrayF64Quad {
    pub value: RegressionSlopeOscillatorDeviceArrayF64,
    pub signal: RegressionSlopeOscillatorDeviceArrayF64,
    pub bullish_reversal: RegressionSlopeOscillatorDeviceArrayF64,
    pub bearish_reversal: RegressionSlopeOscillatorDeviceArrayF64,
}

impl RegressionSlopeOscillatorDeviceArrayF64Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.value.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.value.cols
    }
}

pub struct CudaRegressionSlopeOscillatorBatchResult {
    pub outputs: RegressionSlopeOscillatorDeviceArrayF64Quad,
    pub combos: Vec<RegressionSlopeOscillatorParams>,
}

pub struct CudaRegressionSlopeOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaRegressionSlopeOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaRegressionSlopeOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("regression_slope_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaRegressionSlopeOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_positive(data: &[f64]) -> Option<usize> {
        data.iter()
            .position(|value| value.is_finite() && *value > 0.0)
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaRegressionSlopeOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaRegressionSlopeOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaRegressionSlopeOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaRegressionSlopeOscillatorError::LaunchConfigTooLarge {
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
        sweep: &RegressionSlopeOscillatorBatchRange,
    ) -> Result<CudaRegressionSlopeOscillatorBatchResult, CudaRegressionSlopeOscillatorError> {
        if data.is_empty() {
            return Err(CudaRegressionSlopeOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        Self::first_valid_positive(data).ok_or_else(|| {
            CudaRegressionSlopeOscillatorError::InvalidInput(
                "all values are NaN, non-finite, or non-positive".into(),
            )
        })?;

        let combos = regression_slope_oscillator_expand_grid(sweep)
            .map_err(|err| CudaRegressionSlopeOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaRegressionSlopeOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut min_ranges = Vec::with_capacity(rows);
        let mut max_ranges = Vec::with_capacity(rows);
        let mut steps = Vec::with_capacity(rows);
        let mut signal_lines = Vec::with_capacity(rows);

        for combo in &combos {
            let min_range = combo.min_range.unwrap_or(10);
            let max_range = combo.max_range.unwrap_or(100);
            let step = combo.step.unwrap_or(5);
            let signal_line = combo.signal_line.unwrap_or(7);
            if min_range < 2
                || max_range < 2
                || step == 0
                || signal_line == 0
                || min_range > max_range
            {
                return Err(CudaRegressionSlopeOscillatorError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            min_ranges.push(min_range as i32);
            max_ranges.push(max_range as i32);
            steps.push(step as i32);
            signal_lines.push(signal_line as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaRegressionSlopeOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaRegressionSlopeOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaRegressionSlopeOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaRegressionSlopeOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaRegressionSlopeOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_min_ranges = DeviceBuffer::from_slice(&min_ranges)?;
        let d_max_ranges = DeviceBuffer::from_slice(&max_ranges)?;
        let d_steps = DeviceBuffer::from_slice(&steps)?;
        let d_signal_lines = DeviceBuffer::from_slice(&signal_lines)?;
        let d_out_value = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bullish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bearish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("regression_slope_oscillator_batch_f64")
            .map_err(
                |_| CudaRegressionSlopeOscillatorError::MissingKernelSymbol {
                    name: "regression_slope_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + REGRESSION_SLOPE_OSCILLATOR_BLOCK_X - 1)
            / REGRESSION_SLOPE_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(REGRESSION_SLOPE_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_min_ranges.as_device_ptr(),
                d_max_ranges.as_device_ptr(),
                d_steps.as_device_ptr(),
                d_signal_lines.as_device_ptr(),
                rows as i32,
                d_out_value.as_device_ptr(),
                d_out_signal.as_device_ptr(),
                d_out_bullish.as_device_ptr(),
                d_out_bearish.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaRegressionSlopeOscillatorBatchResult {
            outputs: RegressionSlopeOscillatorDeviceArrayF64Quad {
                value: RegressionSlopeOscillatorDeviceArrayF64 {
                    buf: d_out_value,
                    rows,
                    cols,
                },
                signal: RegressionSlopeOscillatorDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
                bullish_reversal: RegressionSlopeOscillatorDeviceArrayF64 {
                    buf: d_out_bullish,
                    rows,
                    cols,
                },
                bearish_reversal: RegressionSlopeOscillatorDeviceArrayF64 {
                    buf: d_out_bearish,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
