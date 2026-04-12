#![cfg(feature = "cuda")]

use crate::indicators::nonlinear_regression_zero_lag_moving_average::{
    expand_grid_nonlinear_regression_zero_lag_moving_average,
    NonlinearRegressionZeroLagMovingAverageBatchRange,
    NonlinearRegressionZeroLagMovingAverageParams,
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

const NONLINEAR_REGRESSION_ZERO_LAG_MOVING_AVERAGE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_ZLMA_PERIOD: usize = 15;
const DEFAULT_REGRESSION_PERIOD: usize = 15;

#[derive(Debug, Error)]
pub enum CudaNonlinearRegressionZeroLagMovingAverageError {
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

pub struct NonlinearRegressionZeroLagMovingAverageDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl NonlinearRegressionZeroLagMovingAverageDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct NonlinearRegressionZeroLagMovingAverageDeviceArrayF64Quad {
    pub value: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64,
    pub signal: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64,
    pub long_signal: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64,
    pub short_signal: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64,
}

impl NonlinearRegressionZeroLagMovingAverageDeviceArrayF64Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.value.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.value.cols
    }
}

pub struct CudaNonlinearRegressionZeroLagMovingAverageBatchResult {
    pub outputs: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64Quad,
    pub combos: Vec<NonlinearRegressionZeroLagMovingAverageParams>,
}

pub struct CudaNonlinearRegressionZeroLagMovingAverage {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for &value in data {
        if value.is_nan() {
            current = 0;
        } else {
            current += 1;
            best = best.max(current);
        }
    }
    best
}

fn required_valid_count(
    zlma_period: usize,
    regression_period: usize,
) -> Result<usize, CudaNonlinearRegressionZeroLagMovingAverageError> {
    zlma_period
        .checked_mul(2)
        .and_then(|value| value.checked_add(regression_period))
        .and_then(|value| value.checked_sub(2))
        .ok_or_else(|| {
            CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                "required valid count overflow".into(),
            )
        })
}

impl CudaNonlinearRegressionZeroLagMovingAverage {
    pub fn new(device_id: usize) -> Result<Self, CudaNonlinearRegressionZeroLagMovingAverageError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!(
            "nonlinear_regression_zero_lag_moving_average_kernel"
        )?;
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

    pub fn synchronize(&self) -> Result<(), CudaNonlinearRegressionZeroLagMovingAverageError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaNonlinearRegressionZeroLagMovingAverageError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(
                    CudaNonlinearRegressionZeroLagMovingAverageError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    },
                );
            }
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaNonlinearRegressionZeroLagMovingAverageError> {
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
                CudaNonlinearRegressionZeroLagMovingAverageError::LaunchConfigTooLarge {
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
        sweep: &NonlinearRegressionZeroLagMovingAverageBatchRange,
    ) -> Result<
        CudaNonlinearRegressionZeroLagMovingAverageBatchResult,
        CudaNonlinearRegressionZeroLagMovingAverageError,
    > {
        if data.is_empty() {
            return Err(
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                    "empty input".into(),
                ),
            );
        }

        let valid = longest_valid_run(data);
        if valid == 0 {
            return Err(
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                    "all values are NaN".into(),
                ),
            );
        }

        let combos =
            expand_grid_nonlinear_regression_zero_lag_moving_average(sweep).map_err(|err| {
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(err.to_string())
            })?;
        if combos.is_empty() {
            return Err(
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                    "empty parameter grid".into(),
                ),
            );
        }

        let rows = combos.len();
        let cols = data.len();
        let mut zlma_periods = Vec::with_capacity(rows);
        let mut regression_periods = Vec::with_capacity(rows);
        let mut max_zlma_period = 0usize;
        let mut max_regression_period = 0usize;
        let mut max_needed = 0usize;
        for combo in &combos {
            let zlma_period = combo.zlma_period.unwrap_or(DEFAULT_ZLMA_PERIOD);
            let regression_period = combo.regression_period.unwrap_or(DEFAULT_REGRESSION_PERIOD);
            if zlma_period == 0 {
                return Err(
                    CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(format!(
                        "invalid zlma_period: {zlma_period}"
                    )),
                );
            }
            if regression_period == 0 {
                return Err(
                    CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(format!(
                        "invalid regression_period: {regression_period}"
                    )),
                );
            }
            let needed = required_valid_count(zlma_period, regression_period)?;
            max_needed = max_needed.max(needed);
            max_zlma_period = max_zlma_period.max(zlma_period);
            max_regression_period = max_regression_period.max(regression_period);
            zlma_periods.push(i32::try_from(zlma_period).map_err(|_| {
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(format!(
                    "zlma_period out of range: {zlma_period}"
                ))
            })?);
            regression_periods.push(i32::try_from(regression_period).map_err(|_| {
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(format!(
                    "regression_period out of range: {regression_period}"
                ))
            })?);
        }
        if valid < max_needed {
            return Err(
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(format!(
                    "not enough valid data: needed={max_needed}, valid={valid}"
                )),
            );
        }

        let total = rows.checked_mul(cols).ok_or_else(|| {
            CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                "rows*cols overflow".into(),
            )
        })?;
        let first_scratch = rows.checked_mul(max_zlma_period).ok_or_else(|| {
            CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                "first scratch rows*period overflow".into(),
            )
        })?;
        let second_scratch = rows.checked_mul(max_zlma_period).ok_or_else(|| {
            CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                "second scratch rows*period overflow".into(),
            )
        })?;
        let regression_scratch = rows.checked_mul(max_regression_period).ok_or_else(|| {
            CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                "regression scratch rows*period overflow".into(),
            )
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_bytes = total
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let scratch_bytes = first_scratch
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                second_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                regression_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaNonlinearRegressionZeroLagMovingAverageError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_zlma_periods = DeviceBuffer::from_slice(&zlma_periods)?;
        let d_regression_periods = DeviceBuffer::from_slice(&regression_periods)?;
        let d_first_wma_rings = unsafe { DeviceBuffer::<f64>::uninitialized(first_scratch)? };
        let d_second_wma_rings = unsafe { DeviceBuffer::<f64>::uninitialized(second_scratch)? };
        let d_regression_rings = unsafe { DeviceBuffer::<f64>::uninitialized(regression_scratch)? };
        let d_out_value = unsafe { DeviceBuffer::<f64>::uninitialized(total)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(total)? };
        let d_out_long_signal = unsafe { DeviceBuffer::<f64>::uninitialized(total)? };
        let d_out_short_signal = unsafe { DeviceBuffer::<f64>::uninitialized(total)? };

        let func = self
            .module
            .get_function("nonlinear_regression_zero_lag_moving_average_batch_f64")
            .map_err(
                |_| CudaNonlinearRegressionZeroLagMovingAverageError::MissingKernelSymbol {
                    name: "nonlinear_regression_zero_lag_moving_average_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + NONLINEAR_REGRESSION_ZERO_LAG_MOVING_AVERAGE_BLOCK_X - 1)
            / NONLINEAR_REGRESSION_ZERO_LAG_MOVING_AVERAGE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(NONLINEAR_REGRESSION_ZERO_LAG_MOVING_AVERAGE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_zlma_periods.as_device_ptr(),
                d_regression_periods.as_device_ptr(),
                rows as i32,
                max_zlma_period as i32,
                max_regression_period as i32,
                d_first_wma_rings.as_device_ptr(),
                d_second_wma_rings.as_device_ptr(),
                d_regression_rings.as_device_ptr(),
                d_out_value.as_device_ptr(),
                d_out_signal.as_device_ptr(),
                d_out_long_signal.as_device_ptr(),
                d_out_short_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaNonlinearRegressionZeroLagMovingAverageBatchResult {
            outputs: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64Quad {
                value: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64 {
                    buf: d_out_value,
                    rows,
                    cols,
                },
                signal: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
                long_signal: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64 {
                    buf: d_out_long_signal,
                    rows,
                    cols,
                },
                short_signal: NonlinearRegressionZeroLagMovingAverageDeviceArrayF64 {
                    buf: d_out_short_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
