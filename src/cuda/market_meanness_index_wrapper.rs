#![cfg(feature = "cuda")]

use crate::indicators::market_meanness_index::{
    expand_grid_market_meanness_index, MarketMeannessIndexBatchRange, MarketMeannessIndexParams,
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

const MARKET_MEANNESS_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const MIN_LENGTH: usize = 6;

#[derive(Debug, Error)]
pub enum CudaMarketMeannessIndexError {
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

pub struct MarketMeannessIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MarketMeannessIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct MarketMeannessIndexDeviceArrayF64Pair {
    pub mmi: MarketMeannessIndexDeviceArrayF64,
    pub mmi_smoothed: MarketMeannessIndexDeviceArrayF64,
}

impl MarketMeannessIndexDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.mmi.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.mmi.cols
    }
}

pub struct CudaMarketMeannessIndexBatchResult {
    pub outputs: MarketMeannessIndexDeviceArrayF64Pair,
    pub combos: Vec<MarketMeannessIndexParams>,
}

pub struct CudaMarketMeannessIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaMarketMeannessIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaMarketMeannessIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("market_meanness_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMarketMeannessIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn parse_mode_flag(value: &str) -> Result<i32, CudaMarketMeannessIndexError> {
        match value.trim().to_ascii_uppercase().as_str() {
            "PRICE" => Ok(0),
            "CHANGE" => Ok(1),
            _ => Err(CudaMarketMeannessIndexError::InvalidInput(format!(
                "invalid source mode: {value}"
            ))),
        }
    }

    fn is_valid_bar(open: f64, close: f64, mode_flag: i32) -> bool {
        if mode_flag == 0 {
            close.is_finite()
        } else {
            open.is_finite() && close.is_finite()
        }
    }

    fn first_valid_bar(open: &[f64], close: &[f64], mode_flag: i32) -> Option<usize> {
        (0..close.len()).find(|&i| Self::is_valid_bar(open[i], close[i], mode_flag))
    }

    fn count_valid_from(open: &[f64], close: &[f64], start: usize, mode_flag: i32) -> usize {
        (start..close.len())
            .filter(|&i| Self::is_valid_bar(open[i], close[i], mode_flag))
            .count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaMarketMeannessIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaMarketMeannessIndexError::OutOfMemory {
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
    ) -> Result<(), CudaMarketMeannessIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaMarketMeannessIndexError::LaunchConfigTooLarge {
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
        open: &[f64],
        close: &[f64],
        sweep: &MarketMeannessIndexBatchRange,
    ) -> Result<CudaMarketMeannessIndexBatchResult, CudaMarketMeannessIndexError> {
        let len = close.len();
        if len == 0 {
            return Err(CudaMarketMeannessIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != len {
            return Err(CudaMarketMeannessIndexError::InvalidInput(format!(
                "input length mismatch: open={}, close={len}",
                open.len()
            )));
        }

        let combos = expand_grid_market_meanness_index(sweep)
            .map_err(|err| CudaMarketMeannessIndexError::InvalidInput(err.to_string()))?;
        let mut lengths = Vec::with_capacity(combos.len());
        let mut mode_flags = Vec::with_capacity(combos.len());
        let mut max_length = 0usize;

        for combo in &combos {
            let length = combo.length.unwrap_or(300);
            if length < MIN_LENGTH || length > len {
                return Err(CudaMarketMeannessIndexError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={len}"
                )));
            }
            let mode_flag = Self::parse_mode_flag(combo.source_mode.as_deref().unwrap_or("Price"))?;
            let first = Self::first_valid_bar(open, close, mode_flag).ok_or_else(|| {
                CudaMarketMeannessIndexError::InvalidInput("all values are NaN".into())
            })?;
            let valid = Self::count_valid_from(open, close, first, mode_flag);
            if valid < length {
                return Err(CudaMarketMeannessIndexError::InvalidInput(format!(
                    "not enough valid data: needed={length}, valid={valid}"
                )));
            }
            max_length = max_length.max(length);
            lengths.push(length as i32);
            mode_flags.push(mode_flag);
        }

        let rows = combos.len();
        let cols = len;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaMarketMeannessIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                mode_flags
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaMarketMeannessIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaMarketMeannessIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaMarketMeannessIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_len = rows.checked_mul(max_length).ok_or_else(|| {
            CudaMarketMeannessIndexError::InvalidInput("rows*max_length overflow".into())
        })?;
        let scratch_bytes = scratch_len
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaMarketMeannessIndexError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaMarketMeannessIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_mode_flags = DeviceBuffer::from_slice(&mode_flags)?;
        let d_source_ring = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len)? };
        let d_window_buf = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len)? };
        let d_median_buf = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len)? };
        let d_smoothing_buf = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len)? };
        let d_out_mmi = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_smoothed = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("market_meanness_index_batch_f64")
            .map_err(|_| CudaMarketMeannessIndexError::MissingKernelSymbol {
                name: "market_meanness_index_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + MARKET_MEANNESS_INDEX_BLOCK_X - 1) / MARKET_MEANNESS_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(MARKET_MEANNESS_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_mode_flags.as_device_ptr(),
                rows as i32,
                max_length as i32,
                d_source_ring.as_device_ptr(),
                d_window_buf.as_device_ptr(),
                d_median_buf.as_device_ptr(),
                d_smoothing_buf.as_device_ptr(),
                d_out_mmi.as_device_ptr(),
                d_out_smoothed.as_device_ptr()
            ))?;
        }

        Ok(CudaMarketMeannessIndexBatchResult {
            outputs: MarketMeannessIndexDeviceArrayF64Pair {
                mmi: MarketMeannessIndexDeviceArrayF64 {
                    buf: d_out_mmi,
                    rows,
                    cols,
                },
                mmi_smoothed: MarketMeannessIndexDeviceArrayF64 {
                    buf: d_out_smoothed,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
