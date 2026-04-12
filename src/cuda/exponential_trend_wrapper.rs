#![cfg(feature = "cuda")]

use crate::indicators::exponential_trend::{
    expand_grid_exponential_trend, ExponentialTrendBatchRange, ExponentialTrendParams,
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

const EXPONENTIAL_TREND_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaExponentialTrendError {
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

pub struct ExponentialTrendDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl ExponentialTrendDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct ExponentialTrendDeviceArrayF64Sextet {
    pub uptrend_base: ExponentialTrendDeviceArrayF64,
    pub downtrend_base: ExponentialTrendDeviceArrayF64,
    pub uptrend_extension: ExponentialTrendDeviceArrayF64,
    pub downtrend_extension: ExponentialTrendDeviceArrayF64,
    pub bullish_change: ExponentialTrendDeviceArrayF64,
    pub bearish_change: ExponentialTrendDeviceArrayF64,
}

impl ExponentialTrendDeviceArrayF64Sextet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.uptrend_base.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.uptrend_base.cols
    }
}

pub struct CudaExponentialTrendBatchResult {
    pub outputs: ExponentialTrendDeviceArrayF64Sextet,
    pub combos: Vec<ExponentialTrendParams>,
}

pub struct CudaExponentialTrend {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaExponentialTrend {
    pub fn new(device_id: usize) -> Result<Self, CudaExponentialTrendError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("exponential_trend_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaExponentialTrendError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn analyze_valid_segments(
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<(usize, usize), CudaExponentialTrendError> {
        if high.is_empty() {
            return Err(CudaExponentialTrendError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaExponentialTrendError::InvalidInput(format!(
                "length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let mut valid = 0usize;
        let mut run = 0usize;
        let mut max_run = 0usize;
        for i in 0..high.len() {
            if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
                valid += 1;
                run += 1;
                max_run = max_run.max(run);
            } else {
                run = 0;
            }
        }

        if valid == 0 {
            return Err(CudaExponentialTrendError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        Ok((valid, max_run))
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaExponentialTrendError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaExponentialTrendError::OutOfMemory {
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
    ) -> Result<(), CudaExponentialTrendError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaExponentialTrendError::LaunchConfigTooLarge {
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
        sweep: &ExponentialTrendBatchRange,
    ) -> Result<CudaExponentialTrendBatchResult, CudaExponentialTrendError> {
        let (_, max_run) = Self::analyze_valid_segments(high, low, close)?;
        if max_run < 101 {
            return Err(CudaExponentialTrendError::InvalidInput(format!(
                "not enough valid data: needed=101, valid={max_run}"
            )));
        }

        let combos = expand_grid_exponential_trend(sweep)
            .map_err(|err| CudaExponentialTrendError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaExponentialTrendError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut exp_rates = Vec::with_capacity(rows);
        let mut initial_distances = Vec::with_capacity(rows);
        let mut width_multipliers = Vec::with_capacity(rows);
        for combo in &combos {
            let exp_rate = combo.exp_rate.unwrap_or(0.00003);
            let initial_distance = combo.initial_distance.unwrap_or(4.0);
            let width_multiplier = combo.width_multiplier.unwrap_or(1.0);
            if !exp_rate.is_finite() || !(0.0..=0.5).contains(&exp_rate) {
                return Err(CudaExponentialTrendError::InvalidInput(format!(
                    "invalid exp_rate: {exp_rate}"
                )));
            }
            if !initial_distance.is_finite() || initial_distance < 0.1 {
                return Err(CudaExponentialTrendError::InvalidInput(format!(
                    "invalid initial_distance: {initial_distance}"
                )));
            }
            if !width_multiplier.is_finite() || width_multiplier < 0.1 {
                return Err(CudaExponentialTrendError::InvalidInput(format!(
                    "invalid width_multiplier: {width_multiplier}"
                )));
            }
            exp_rates.push(exp_rate);
            initial_distances.push(initial_distance);
            width_multipliers.push(width_multiplier);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaExponentialTrendError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaExponentialTrendError::InvalidInput("parameter bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaExponentialTrendError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(6))
            .ok_or_else(|| {
                CudaExponentialTrendError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaExponentialTrendError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_exp_rates = DeviceBuffer::from_slice(&exp_rates)?;
        let d_initial_distances = DeviceBuffer::from_slice(&initial_distances)?;
        let d_width_multipliers = DeviceBuffer::from_slice(&width_multipliers)?;
        let d_out_uptrend_base = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_downtrend_base = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_uptrend_extension = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_downtrend_extension =
            unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bullish_change = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bearish_change = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("exponential_trend_batch_f64")
            .map_err(|_| CudaExponentialTrendError::MissingKernelSymbol {
                name: "exponential_trend_batch_f64",
            })?;
        let grid_x = ((rows as u32) + EXPONENTIAL_TREND_BLOCK_X - 1) / EXPONENTIAL_TREND_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EXPONENTIAL_TREND_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_exp_rates.as_device_ptr(),
                d_initial_distances.as_device_ptr(),
                d_width_multipliers.as_device_ptr(),
                rows as i32,
                d_out_uptrend_base.as_device_ptr(),
                d_out_downtrend_base.as_device_ptr(),
                d_out_uptrend_extension.as_device_ptr(),
                d_out_downtrend_extension.as_device_ptr(),
                d_out_bullish_change.as_device_ptr(),
                d_out_bearish_change.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaExponentialTrendBatchResult {
            outputs: ExponentialTrendDeviceArrayF64Sextet {
                uptrend_base: ExponentialTrendDeviceArrayF64 {
                    buf: d_out_uptrend_base,
                    rows,
                    cols,
                },
                downtrend_base: ExponentialTrendDeviceArrayF64 {
                    buf: d_out_downtrend_base,
                    rows,
                    cols,
                },
                uptrend_extension: ExponentialTrendDeviceArrayF64 {
                    buf: d_out_uptrend_extension,
                    rows,
                    cols,
                },
                downtrend_extension: ExponentialTrendDeviceArrayF64 {
                    buf: d_out_downtrend_extension,
                    rows,
                    cols,
                },
                bullish_change: ExponentialTrendDeviceArrayF64 {
                    buf: d_out_bullish_change,
                    rows,
                    cols,
                },
                bearish_change: ExponentialTrendDeviceArrayF64 {
                    buf: d_out_bearish_change,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
