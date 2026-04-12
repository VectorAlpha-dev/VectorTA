#![cfg(feature = "cuda")]

use crate::indicators::moving_average_cross_probability::{
    moving_average_cross_probability_expand_grid, MovingAverageCrossProbabilityBatchRange,
    MovingAverageCrossProbabilityMaType, MovingAverageCrossProbabilityParams,
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

const MOVING_AVERAGE_CROSS_PROBABILITY_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_SMOOTHING_WINDOW: usize = 7;
const DEFAULT_SLOW_LENGTH: usize = 30;
const DEFAULT_FAST_LENGTH: usize = 14;
const DEFAULT_RESOLUTION: usize = 50;
const MA_EMA: i32 = 0;
const MA_SMA: i32 = 1;

#[derive(Debug, Error)]
pub enum CudaMovingAverageCrossProbabilityError {
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

pub struct MovingAverageCrossProbabilityDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MovingAverageCrossProbabilityDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct MovingAverageCrossProbabilityDeviceArrayF64Seven {
    pub value: MovingAverageCrossProbabilityDeviceArrayF64,
    pub slow_ma: MovingAverageCrossProbabilityDeviceArrayF64,
    pub fast_ma: MovingAverageCrossProbabilityDeviceArrayF64,
    pub forecast: MovingAverageCrossProbabilityDeviceArrayF64,
    pub upper: MovingAverageCrossProbabilityDeviceArrayF64,
    pub lower: MovingAverageCrossProbabilityDeviceArrayF64,
    pub direction: MovingAverageCrossProbabilityDeviceArrayF64,
}

impl MovingAverageCrossProbabilityDeviceArrayF64Seven {
    #[inline]
    pub fn rows(&self) -> usize {
        self.value.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.value.cols
    }
}

pub struct CudaMovingAverageCrossProbabilityBatchResult {
    pub outputs: MovingAverageCrossProbabilityDeviceArrayF64Seven,
    pub combos: Vec<MovingAverageCrossProbabilityParams>,
}

pub struct CudaMovingAverageCrossProbability {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn parse_ma_type(value: MovingAverageCrossProbabilityMaType) -> i32 {
    match value {
        MovingAverageCrossProbabilityMaType::Ema => MA_EMA,
        MovingAverageCrossProbabilityMaType::Sma => MA_SMA,
    }
}

impl CudaMovingAverageCrossProbability {
    pub fn new(device_id: usize) -> Result<Self, CudaMovingAverageCrossProbabilityError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("moving_average_cross_probability_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMovingAverageCrossProbabilityError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaMovingAverageCrossProbabilityError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaMovingAverageCrossProbabilityError::OutOfMemory {
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
    ) -> Result<(), CudaMovingAverageCrossProbabilityError> {
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
                CudaMovingAverageCrossProbabilityError::LaunchConfigTooLarge {
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
        sweep: &MovingAverageCrossProbabilityBatchRange,
    ) -> Result<CudaMovingAverageCrossProbabilityBatchResult, CudaMovingAverageCrossProbabilityError>
    {
        if data.is_empty() {
            return Err(CudaMovingAverageCrossProbabilityError::InvalidInput(
                "empty input".into(),
            ));
        }
        if !data.iter().any(|value| value.is_finite()) {
            return Err(CudaMovingAverageCrossProbabilityError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = moving_average_cross_probability_expand_grid(sweep)
            .map_err(|e| CudaMovingAverageCrossProbabilityError::InvalidInput(e.to_string()))?;
        if combos.is_empty() {
            return Err(CudaMovingAverageCrossProbabilityError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut smoothing_windows = Vec::with_capacity(rows);
        let mut slow_lengths = Vec::with_capacity(rows);
        let mut fast_lengths = Vec::with_capacity(rows);
        let mut resolutions = Vec::with_capacity(rows);
        let mut ma_codes = Vec::with_capacity(rows);
        let mut max_smoothing_window = 0usize;
        let mut max_slow_length = 0usize;
        let mut max_fast_length = 0usize;

        for combo in &combos {
            let smoothing_window = combo.smoothing_window.unwrap_or(DEFAULT_SMOOTHING_WINDOW);
            let slow_length = combo.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH);
            let fast_length = combo.fast_length.unwrap_or(DEFAULT_FAST_LENGTH);
            let resolution = combo.resolution.unwrap_or(DEFAULT_RESOLUTION);
            let ma_type = combo
                .ma_type
                .unwrap_or(MovingAverageCrossProbabilityMaType::Ema);

            if smoothing_window < 2 {
                return Err(CudaMovingAverageCrossProbabilityError::InvalidInput(
                    format!("invalid smoothing_window: {smoothing_window}"),
                ));
            }
            if slow_length < 2 {
                return Err(CudaMovingAverageCrossProbabilityError::InvalidInput(
                    format!("invalid slow_length: {slow_length}"),
                ));
            }
            if fast_length == 0 {
                return Err(CudaMovingAverageCrossProbabilityError::InvalidInput(
                    format!("invalid fast_length: {fast_length}"),
                ));
            }
            if slow_length <= fast_length {
                return Err(CudaMovingAverageCrossProbabilityError::InvalidInput(
                    format!(
                    "invalid length order: fast_length={fast_length}, slow_length={slow_length}"
                ),
                ));
            }
            if resolution < 2 {
                return Err(CudaMovingAverageCrossProbabilityError::InvalidInput(
                    format!("invalid resolution: {resolution}"),
                ));
            }

            max_smoothing_window = max_smoothing_window.max(smoothing_window);
            max_slow_length = max_slow_length.max(slow_length);
            max_fast_length = max_fast_length.max(fast_length);
            smoothing_windows.push(i32::try_from(smoothing_window).map_err(|_| {
                CudaMovingAverageCrossProbabilityError::InvalidInput(format!(
                    "smoothing_window out of range: {smoothing_window}"
                ))
            })?);
            slow_lengths.push(i32::try_from(slow_length).map_err(|_| {
                CudaMovingAverageCrossProbabilityError::InvalidInput(format!(
                    "slow_length out of range: {slow_length}"
                ))
            })?);
            fast_lengths.push(i32::try_from(fast_length).map_err(|_| {
                CudaMovingAverageCrossProbabilityError::InvalidInput(format!(
                    "fast_length out of range: {fast_length}"
                ))
            })?);
            resolutions.push(i32::try_from(resolution).map_err(|_| {
                CudaMovingAverageCrossProbabilityError::InvalidInput(format!(
                    "resolution out of range: {resolution}"
                ))
            })?);
            ma_codes.push(parse_ma_type(ma_type));
        }

        let scratch_row_stride = max_smoothing_window
            .checked_mul(4)
            .and_then(|value| value.checked_add(max_slow_length))
            .and_then(|value| value.checked_add(max_fast_length))
            .ok_or_else(|| {
                CudaMovingAverageCrossProbabilityError::InvalidInput(
                    "scratch row stride overflow".into(),
                )
            })?;
        let scratch_elems = rows.checked_mul(scratch_row_stride).ok_or_else(|| {
            CudaMovingAverageCrossProbabilityError::InvalidInput("scratch size overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaMovingAverageCrossProbabilityError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| {
                CudaMovingAverageCrossProbabilityError::InvalidInput("params bytes overflow".into())
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaMovingAverageCrossProbabilityError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaMovingAverageCrossProbabilityError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(7))
            .ok_or_else(|| {
                CudaMovingAverageCrossProbabilityError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaMovingAverageCrossProbabilityError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_smoothing_windows = DeviceBuffer::from_slice(&smoothing_windows)?;
        let d_slow_lengths = DeviceBuffer::from_slice(&slow_lengths)?;
        let d_fast_lengths = DeviceBuffer::from_slice(&fast_lengths)?;
        let d_resolutions = DeviceBuffer::from_slice(&resolutions)?;
        let d_ma_codes = DeviceBuffer::from_slice(&ma_codes)?;
        let d_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_out_value = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_slow_ma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_fast_ma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_forecast = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_direction = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("moving_average_cross_probability_batch_f64")
            .map_err(
                |_| CudaMovingAverageCrossProbabilityError::MissingKernelSymbol {
                    name: "moving_average_cross_probability_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + MOVING_AVERAGE_CROSS_PROBABILITY_BLOCK_X - 1)
            / MOVING_AVERAGE_CROSS_PROBABILITY_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(MOVING_AVERAGE_CROSS_PROBABILITY_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_smoothing_windows.as_device_ptr(),
                d_slow_lengths.as_device_ptr(),
                d_fast_lengths.as_device_ptr(),
                d_resolutions.as_device_ptr(),
                d_ma_codes.as_device_ptr(),
                rows as i32,
                max_smoothing_window as i32,
                max_slow_length as i32,
                max_fast_length as i32,
                d_scratch.as_device_ptr(),
                d_out_value.as_device_ptr(),
                d_out_slow_ma.as_device_ptr(),
                d_out_fast_ma.as_device_ptr(),
                d_out_forecast.as_device_ptr(),
                d_out_upper.as_device_ptr(),
                d_out_lower.as_device_ptr(),
                d_out_direction.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaMovingAverageCrossProbabilityBatchResult {
            outputs: MovingAverageCrossProbabilityDeviceArrayF64Seven {
                value: MovingAverageCrossProbabilityDeviceArrayF64 {
                    buf: d_out_value,
                    rows,
                    cols,
                },
                slow_ma: MovingAverageCrossProbabilityDeviceArrayF64 {
                    buf: d_out_slow_ma,
                    rows,
                    cols,
                },
                fast_ma: MovingAverageCrossProbabilityDeviceArrayF64 {
                    buf: d_out_fast_ma,
                    rows,
                    cols,
                },
                forecast: MovingAverageCrossProbabilityDeviceArrayF64 {
                    buf: d_out_forecast,
                    rows,
                    cols,
                },
                upper: MovingAverageCrossProbabilityDeviceArrayF64 {
                    buf: d_out_upper,
                    rows,
                    cols,
                },
                lower: MovingAverageCrossProbabilityDeviceArrayF64 {
                    buf: d_out_lower,
                    rows,
                    cols,
                },
                direction: MovingAverageCrossProbabilityDeviceArrayF64 {
                    buf: d_out_direction,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
