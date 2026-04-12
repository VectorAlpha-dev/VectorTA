#![cfg(feature = "cuda")]

use crate::indicators::accumulation_swing_index::{
    expand_grid, AccumulationSwingIndexBatchRange, AccumulationSwingIndexParams,
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

const ACCUMULATION_SWING_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_DAILY_LIMIT: f64 = 10_000.0;

#[derive(Debug, Error)]
pub enum CudaAccumulationSwingIndexError {
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

pub struct AccumulationSwingIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AccumulationSwingIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaAccumulationSwingIndexBatchResult {
    pub outputs: AccumulationSwingIndexDeviceArrayF64,
    pub combos: Vec<AccumulationSwingIndexParams>,
}

pub struct CudaAccumulationSwingIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAccumulationSwingIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaAccumulationSwingIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("accumulation_swing_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAccumulationSwingIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len()).find(|&i| {
            open[i].is_finite() && high[i].is_finite() && low[i].is_finite() && close[i].is_finite()
        })
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaAccumulationSwingIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAccumulationSwingIndexError::OutOfMemory {
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
    ) -> Result<(), CudaAccumulationSwingIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaAccumulationSwingIndexError::LaunchConfigTooLarge {
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
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &AccumulationSwingIndexBatchRange,
    ) -> Result<CudaAccumulationSwingIndexBatchResult, CudaAccumulationSwingIndexError> {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaAccumulationSwingIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaAccumulationSwingIndexError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }
        Self::first_valid_ohlc(open, high, low, close).ok_or_else(|| {
            CudaAccumulationSwingIndexError::InvalidInput("all values are NaN".into())
        })?;

        let combos = expand_grid(sweep)
            .map_err(|err| CudaAccumulationSwingIndexError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaAccumulationSwingIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut daily_limits = Vec::with_capacity(rows);
        for combo in &combos {
            let daily_limit = combo.daily_limit.unwrap_or(DEFAULT_DAILY_LIMIT);
            if !daily_limit.is_finite() || daily_limit <= 0.0 {
                return Err(CudaAccumulationSwingIndexError::InvalidInput(
                    "invalid daily_limit".into(),
                ));
            }
            daily_limits.push(daily_limit);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaAccumulationSwingIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaAccumulationSwingIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaAccumulationSwingIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaAccumulationSwingIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaAccumulationSwingIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_daily_limits = DeviceBuffer::from_slice(&daily_limits)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("accumulation_swing_index_batch_f64")
            .map_err(|_| CudaAccumulationSwingIndexError::MissingKernelSymbol {
                name: "accumulation_swing_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + ACCUMULATION_SWING_INDEX_BLOCK_X - 1)
            / ACCUMULATION_SWING_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ACCUMULATION_SWING_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_daily_limits.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaAccumulationSwingIndexBatchResult {
            outputs: AccumulationSwingIndexDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
