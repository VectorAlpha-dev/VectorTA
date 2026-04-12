#![cfg(feature = "cuda")]

use crate::indicators::price_density_market_noise::{
    expand_grid_price_density_market_noise, PriceDensityMarketNoiseBatchRange,
    PriceDensityMarketNoiseParams,
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

const PRICE_DENSITY_MARKET_NOISE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaPriceDensityMarketNoiseError {
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

pub struct PriceDensityMarketNoiseDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl PriceDensityMarketNoiseDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct PriceDensityMarketNoiseDeviceArrayF64Pair {
    pub price_density: PriceDensityMarketNoiseDeviceArrayF64,
    pub price_density_percent: PriceDensityMarketNoiseDeviceArrayF64,
}

impl PriceDensityMarketNoiseDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.price_density.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.price_density.cols
    }
}

pub struct CudaPriceDensityMarketNoiseBatchResult {
    pub outputs: PriceDensityMarketNoiseDeviceArrayF64Pair,
    pub combos: Vec<PriceDensityMarketNoiseParams>,
}

pub struct CudaPriceDensityMarketNoise {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaPriceDensityMarketNoise {
    pub fn new(device_id: usize) -> Result<Self, CudaPriceDensityMarketNoiseError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("price_density_market_noise_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaPriceDensityMarketNoiseError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len())
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
    }

    fn count_valid_from(high: &[f64], low: &[f64], close: &[f64], start: usize) -> usize {
        (start..close.len())
            .filter(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
            .count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaPriceDensityMarketNoiseError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaPriceDensityMarketNoiseError::OutOfMemory {
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
    ) -> Result<(), CudaPriceDensityMarketNoiseError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaPriceDensityMarketNoiseError::LaunchConfigTooLarge {
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
        sweep: &PriceDensityMarketNoiseBatchRange,
    ) -> Result<CudaPriceDensityMarketNoiseBatchResult, CudaPriceDensityMarketNoiseError> {
        let len = close.len();
        if len == 0 || high.is_empty() || low.is_empty() {
            return Err(CudaPriceDensityMarketNoiseError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != len || low.len() != len {
            return Err(CudaPriceDensityMarketNoiseError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={len}",
                high.len(),
                low.len(),
            )));
        }

        let combos = expand_grid_price_density_market_noise(sweep)
            .map_err(|err| CudaPriceDensityMarketNoiseError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaPriceDensityMarketNoiseError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let first = Self::first_valid_bar(high, low, close).ok_or_else(|| {
            CudaPriceDensityMarketNoiseError::InvalidInput("all values are NaN".into())
        })?;
        let valid = Self::count_valid_from(high, low, close, first);

        let mut max_length = 0usize;
        let mut max_eval_period = 0usize;
        let mut lengths = Vec::with_capacity(combos.len());
        let mut eval_periods = Vec::with_capacity(combos.len());
        for combo in &combos {
            let length = combo.length.unwrap_or(14);
            let eval_period = combo.eval_period.unwrap_or(200);
            if length == 0 || length > len {
                return Err(CudaPriceDensityMarketNoiseError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={len}"
                )));
            }
            if eval_period == 0 {
                return Err(CudaPriceDensityMarketNoiseError::InvalidInput(format!(
                    "invalid eval_period: eval_period={eval_period}, data_len={len}"
                )));
            }
            if valid < length {
                return Err(CudaPriceDensityMarketNoiseError::InvalidInput(format!(
                    "not enough valid data: needed={length}, valid={valid}"
                )));
            }
            max_length = max_length.max(length);
            max_eval_period = max_eval_period.max(eval_period);
            lengths.push(length as i32);
            eval_periods.push(eval_period as i32);
        }

        let rows = combos.len();
        let cols = len;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaPriceDensityMarketNoiseError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                eval_periods
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaPriceDensityMarketNoiseError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaPriceDensityMarketNoiseError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaPriceDensityMarketNoiseError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_elems = rows
            .checked_mul(max_length)
            .and_then(|value| value.checked_mul(3))
            .and_then(|value| {
                rows.checked_mul(max_eval_period)
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaPriceDensityMarketNoiseError::InvalidInput("scratch elems overflow".into())
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaPriceDensityMarketNoiseError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaPriceDensityMarketNoiseError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_eval_periods = DeviceBuffer::from_slice(&eval_periods)?;
        let d_high_ring = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_length)? };
        let d_low_ring = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_length)? };
        let d_tr_ring = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_length)? };
        let d_pd_ring = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_eval_period)? };
        let d_out_price_density = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_price_density_percent =
            unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("price_density_market_noise_batch_f64")
            .map_err(|_| CudaPriceDensityMarketNoiseError::MissingKernelSymbol {
                name: "price_density_market_noise_batch_f64",
            })?;
        let grid_x = ((rows as u32) + PRICE_DENSITY_MARKET_NOISE_BLOCK_X - 1)
            / PRICE_DENSITY_MARKET_NOISE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(PRICE_DENSITY_MARKET_NOISE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_eval_periods.as_device_ptr(),
                rows as i32,
                max_length as i32,
                max_eval_period as i32,
                d_high_ring.as_device_ptr(),
                d_low_ring.as_device_ptr(),
                d_tr_ring.as_device_ptr(),
                d_pd_ring.as_device_ptr(),
                d_out_price_density.as_device_ptr(),
                d_out_price_density_percent.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaPriceDensityMarketNoiseBatchResult {
            outputs: PriceDensityMarketNoiseDeviceArrayF64Pair {
                price_density: PriceDensityMarketNoiseDeviceArrayF64 {
                    buf: d_out_price_density,
                    rows,
                    cols,
                },
                price_density_percent: PriceDensityMarketNoiseDeviceArrayF64 {
                    buf: d_out_price_density_percent,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
