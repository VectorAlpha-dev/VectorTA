#![cfg(feature = "cuda")]

use crate::indicators::impulse_macd::{
    expand_grid_impulse_macd, ImpulseMacdBatchRange, ImpulseMacdParams,
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

const IMPULSE_MACD_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH_MA: usize = 34;
const DEFAULT_LENGTH_SIGNAL: usize = 9;

#[derive(Debug, Error)]
pub enum CudaImpulseMacdError {
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

pub struct ImpulseMacdDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl ImpulseMacdDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct ImpulseMacdDeviceArrayF64Triple {
    pub impulse_macd: ImpulseMacdDeviceArrayF64,
    pub impulse_histo: ImpulseMacdDeviceArrayF64,
    pub signal: ImpulseMacdDeviceArrayF64,
}

impl ImpulseMacdDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.impulse_macd.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.impulse_macd.cols
    }
}

pub struct CudaImpulseMacdBatchResult {
    pub outputs: ImpulseMacdDeviceArrayF64Triple,
    pub combos: Vec<ImpulseMacdParams>,
}

pub struct CudaImpulseMacd {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaImpulseMacd {
    pub fn new(device_id: usize) -> Result<Self, CudaImpulseMacdError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("impulse_macd_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaImpulseMacdError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn valid_bar(high: f64, low: f64, close: f64) -> bool {
        high.is_finite() && low.is_finite() && close.is_finite() && high >= low
    }

    fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len()).find(|&i| Self::valid_bar(high[i], low[i], close[i]))
    }

    fn count_valid_from(high: &[f64], low: &[f64], close: &[f64], start: usize) -> usize {
        (start..close.len())
            .filter(|&i| Self::valid_bar(high[i], low[i], close[i]))
            .count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaImpulseMacdError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaImpulseMacdError::OutOfMemory {
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
    ) -> Result<(), CudaImpulseMacdError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaImpulseMacdError::LaunchConfigTooLarge {
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
        sweep: &ImpulseMacdBatchRange,
    ) -> Result<CudaImpulseMacdBatchResult, CudaImpulseMacdError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaImpulseMacdError::InvalidInput("empty input".into()));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaImpulseMacdError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let first = Self::first_valid_bar(high, low, close)
            .ok_or_else(|| CudaImpulseMacdError::InvalidInput("all values are NaN".into()))?;
        let valid = Self::count_valid_from(high, low, close, first);

        let combos = expand_grid_impulse_macd(sweep)
            .map_err(|err| CudaImpulseMacdError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaImpulseMacdError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut length_mas = Vec::with_capacity(rows);
        let mut length_signals = Vec::with_capacity(rows);
        let mut max_signal_length = 0usize;
        for combo in &combos {
            let length_ma = combo.length_ma.unwrap_or(DEFAULT_LENGTH_MA);
            let length_signal = combo.length_signal.unwrap_or(DEFAULT_LENGTH_SIGNAL);
            if length_ma == 0 || length_ma > cols {
                return Err(CudaImpulseMacdError::InvalidInput(format!(
                    "invalid length_ma: length_ma={length_ma}, data_len={cols}"
                )));
            }
            if length_signal == 0 || length_signal > cols {
                return Err(CudaImpulseMacdError::InvalidInput(format!(
                    "invalid length_signal: length_signal={length_signal}, data_len={cols}"
                )));
            }
            let needed = length_ma.max(length_signal);
            if valid < needed {
                return Err(CudaImpulseMacdError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            max_signal_length = max_signal_length.max(length_signal);
            length_mas.push(i32::try_from(length_ma).map_err(|_| {
                CudaImpulseMacdError::InvalidInput(format!("length_ma out of range: {length_ma}"))
            })?);
            length_signals.push(i32::try_from(length_signal).map_err(|_| {
                CudaImpulseMacdError::InvalidInput(format!(
                    "length_signal out of range: {length_signal}"
                ))
            })?);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| CudaImpulseMacdError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| CudaImpulseMacdError::InvalidInput("params bytes overflow".into()))?;
        let scratch_elems = rows.checked_mul(max_signal_length).ok_or_else(|| {
            CudaImpulseMacdError::InvalidInput("scratch rows*cols overflow".into())
        })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaImpulseMacdError::InvalidInput("scratch bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaImpulseMacdError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| CudaImpulseMacdError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| CudaImpulseMacdError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_length_mas = DeviceBuffer::from_slice(&length_mas)?;
        let d_length_signals = DeviceBuffer::from_slice(&length_signals)?;
        let d_signal_buf = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_out_md = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_hist = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("impulse_macd_batch_f64")
            .map_err(|_| CudaImpulseMacdError::MissingKernelSymbol {
                name: "impulse_macd_batch_f64",
            })?;
        let grid_x = ((rows as u32) + IMPULSE_MACD_BLOCK_X - 1) / IMPULSE_MACD_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(IMPULSE_MACD_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_length_mas.as_device_ptr(),
                d_length_signals.as_device_ptr(),
                rows as i32,
                max_signal_length as i32,
                d_signal_buf.as_device_ptr(),
                d_out_md.as_device_ptr(),
                d_out_hist.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaImpulseMacdBatchResult {
            outputs: ImpulseMacdDeviceArrayF64Triple {
                impulse_macd: ImpulseMacdDeviceArrayF64 {
                    buf: d_out_md,
                    rows,
                    cols,
                },
                impulse_histo: ImpulseMacdDeviceArrayF64 {
                    buf: d_out_hist,
                    rows,
                    cols,
                },
                signal: ImpulseMacdDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
