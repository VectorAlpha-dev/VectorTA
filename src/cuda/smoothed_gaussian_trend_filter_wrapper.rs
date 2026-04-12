#![cfg(feature = "cuda")]

use crate::indicators::smoothed_gaussian_trend_filter::{
    expand_grid_smoothed_gaussian_trend_filter, SmoothedGaussianTrendFilterBatchRange,
    SmoothedGaussianTrendFilterParams,
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

const SMOOTHED_GAUSSIAN_TREND_FILTER_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaSmoothedGaussianTrendFilterError {
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

pub struct SmoothedGaussianTrendFilterDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl SmoothedGaussianTrendFilterDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct SmoothedGaussianTrendFilterDeviceArrayF64Quad {
    pub filter: SmoothedGaussianTrendFilterDeviceArrayF64,
    pub supertrend: SmoothedGaussianTrendFilterDeviceArrayF64,
    pub trend: SmoothedGaussianTrendFilterDeviceArrayF64,
    pub ranging: SmoothedGaussianTrendFilterDeviceArrayF64,
}

impl SmoothedGaussianTrendFilterDeviceArrayF64Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.filter.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.filter.cols
    }
}

pub struct CudaSmoothedGaussianTrendFilterBatchResult {
    pub outputs: SmoothedGaussianTrendFilterDeviceArrayF64Quad,
    pub combos: Vec<SmoothedGaussianTrendFilterParams>,
}

pub struct CudaSmoothedGaussianTrendFilter {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaSmoothedGaussianTrendFilter {
    pub fn new(device_id: usize) -> Result<Self, CudaSmoothedGaussianTrendFilterError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("smoothed_gaussian_trend_filter_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaSmoothedGaussianTrendFilterError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len()).find(|&i| {
            high[i].is_finite() && low[i].is_finite() && close[i].is_finite() && high[i] >= low[i]
        })
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaSmoothedGaussianTrendFilterError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaSmoothedGaussianTrendFilterError::OutOfMemory {
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
    ) -> Result<(), CudaSmoothedGaussianTrendFilterError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaSmoothedGaussianTrendFilterError::LaunchConfigTooLarge {
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
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &SmoothedGaussianTrendFilterBatchRange,
    ) -> Result<CudaSmoothedGaussianTrendFilterBatchResult, CudaSmoothedGaussianTrendFilterError>
    {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaSmoothedGaussianTrendFilterError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaSmoothedGaussianTrendFilterError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = expand_grid_smoothed_gaussian_trend_filter(sweep)
            .map_err(|err| CudaSmoothedGaussianTrendFilterError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaSmoothedGaussianTrendFilterError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let first_valid = Self::first_valid_bar(high, low, close).ok_or_else(|| {
            CudaSmoothedGaussianTrendFilterError::InvalidInput("all values are NaN".into())
        })?;
        let valid = close.len() - first_valid;
        let rows = combos.len();
        let cols = close.len();
        let mut max_needed = 0usize;
        let mut gaussian_lengths = Vec::with_capacity(rows);
        let mut poles_values = Vec::with_capacity(rows);
        let mut smoothing_lengths = Vec::with_capacity(rows);
        let mut linreg_offsets = Vec::with_capacity(rows);

        for combo in &combos {
            let gaussian_length = combo.gaussian_length.unwrap_or(15);
            let poles = combo.poles.unwrap_or(3);
            let smoothing_length = combo.smoothing_length.unwrap_or(22);
            let linreg_offset = combo.linreg_offset.unwrap_or(7);
            if gaussian_length == 0
                || gaussian_length > cols
                || smoothing_length == 0
                || smoothing_length > cols
                || !(1..=4).contains(&poles)
            {
                return Err(CudaSmoothedGaussianTrendFilterError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            max_needed = max_needed.max(smoothing_length.max(21));
            gaussian_lengths.push(gaussian_length as i32);
            poles_values.push(poles as i32);
            smoothing_lengths.push(smoothing_length as i32);
            linreg_offsets.push(linreg_offset as i32);
        }

        if valid < max_needed {
            return Err(CudaSmoothedGaussianTrendFilterError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={valid}"
            )));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaSmoothedGaussianTrendFilterError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaSmoothedGaussianTrendFilterError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaSmoothedGaussianTrendFilterError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaSmoothedGaussianTrendFilterError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaSmoothedGaussianTrendFilterError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_gaussian_lengths = DeviceBuffer::from_slice(&gaussian_lengths)?;
        let d_poles_values = DeviceBuffer::from_slice(&poles_values)?;
        let d_smoothing_lengths = DeviceBuffer::from_slice(&smoothing_lengths)?;
        let d_linreg_offsets = DeviceBuffer::from_slice(&linreg_offsets)?;
        let d_out_filter = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_supertrend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_trend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ranging = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("smoothed_gaussian_trend_filter_batch_f64")
            .map_err(
                |_| CudaSmoothedGaussianTrendFilterError::MissingKernelSymbol {
                    name: "smoothed_gaussian_trend_filter_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + SMOOTHED_GAUSSIAN_TREND_FILTER_BLOCK_X - 1)
            / SMOOTHED_GAUSSIAN_TREND_FILTER_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(SMOOTHED_GAUSSIAN_TREND_FILTER_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_gaussian_lengths.as_device_ptr(),
                d_poles_values.as_device_ptr(),
                d_smoothing_lengths.as_device_ptr(),
                d_linreg_offsets.as_device_ptr(),
                rows as i32,
                d_out_filter.as_device_ptr(),
                d_out_supertrend.as_device_ptr(),
                d_out_trend.as_device_ptr(),
                d_out_ranging.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaSmoothedGaussianTrendFilterBatchResult {
            outputs: SmoothedGaussianTrendFilterDeviceArrayF64Quad {
                filter: SmoothedGaussianTrendFilterDeviceArrayF64 {
                    buf: d_out_filter,
                    rows,
                    cols,
                },
                supertrend: SmoothedGaussianTrendFilterDeviceArrayF64 {
                    buf: d_out_supertrend,
                    rows,
                    cols,
                },
                trend: SmoothedGaussianTrendFilterDeviceArrayF64 {
                    buf: d_out_trend,
                    rows,
                    cols,
                },
                ranging: SmoothedGaussianTrendFilterDeviceArrayF64 {
                    buf: d_out_ranging,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
