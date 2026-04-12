#![cfg(feature = "cuda")]

use crate::indicators::adaptive_macd::{expand_grid, AdaptiveMacdBatchRange, AdaptiveMacdParams};
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

const ADAPTIVE_MACD_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaAdaptiveMacdError {
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

pub struct AdaptiveMacdDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AdaptiveMacdDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct AdaptiveMacdDeviceArrayF64Triple {
    pub macd: AdaptiveMacdDeviceArrayF64,
    pub signal: AdaptiveMacdDeviceArrayF64,
    pub hist: AdaptiveMacdDeviceArrayF64,
}

impl AdaptiveMacdDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.macd.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.macd.cols
    }
}

pub struct CudaAdaptiveMacdBatchResult {
    pub outputs: AdaptiveMacdDeviceArrayF64Triple,
    pub combos: Vec<AdaptiveMacdParams>,
}

pub struct CudaAdaptiveMacd {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAdaptiveMacd {
    pub fn new(device_id: usize) -> Result<Self, CudaAdaptiveMacdError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("adaptive_macd_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAdaptiveMacdError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_index(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| !value.is_nan())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaAdaptiveMacdError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAdaptiveMacdError::OutOfMemory {
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
    ) -> Result<(), CudaAdaptiveMacdError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaAdaptiveMacdError::LaunchConfigTooLarge {
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
        sweep: &AdaptiveMacdBatchRange,
    ) -> Result<CudaAdaptiveMacdBatchResult, CudaAdaptiveMacdError> {
        if data.is_empty() {
            return Err(CudaAdaptiveMacdError::InvalidInput("empty input".into()));
        }

        let combos = expand_grid(sweep)
            .map_err(|err| CudaAdaptiveMacdError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaAdaptiveMacdError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let first_valid = Self::first_valid_index(data)
            .ok_or_else(|| CudaAdaptiveMacdError::InvalidInput("all values are NaN".into()))?;
        let valid = data.len().saturating_sub(first_valid);

        let rows = combos.len();
        let cols = data.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut fast_periods = Vec::with_capacity(rows);
        let mut slow_periods = Vec::with_capacity(rows);
        let mut signal_periods = Vec::with_capacity(rows);
        for combo in &combos {
            let length = combo.length.unwrap_or(20);
            let fast_period = combo.fast_period.unwrap_or(10);
            let slow_period = combo.slow_period.unwrap_or(20);
            let signal_period = combo.signal_period.unwrap_or(9);
            if length < 2
                || fast_period < 2
                || slow_period < 2
                || signal_period < 2
                || length > cols
                || fast_period > cols
                || slow_period > cols
                || signal_period > cols
            {
                return Err(CudaAdaptiveMacdError::InvalidInput(format!(
                    "invalid period: length={length}, fast={fast_period}, slow={slow_period}, signal={signal_period}, data_len={cols}"
                )));
            }
            if valid < length {
                return Err(CudaAdaptiveMacdError::InvalidInput(format!(
                    "not enough valid data: needed={length}, valid={valid}"
                )));
            }
            lengths.push(length as i32);
            fast_periods.push(fast_period as i32);
            slow_periods.push(slow_period as i32);
            signal_periods.push(signal_period as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaAdaptiveMacdError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                fast_periods
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                slow_periods
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                signal_periods
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| CudaAdaptiveMacdError::InvalidInput("params bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAdaptiveMacdError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| CudaAdaptiveMacdError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| CudaAdaptiveMacdError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_fast_periods = DeviceBuffer::from_slice(&fast_periods)?;
        let d_slow_periods = DeviceBuffer::from_slice(&slow_periods)?;
        let d_signal_periods = DeviceBuffer::from_slice(&signal_periods)?;
        let d_out_macd = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_hist = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("adaptive_macd_batch_f64")
            .map_err(|_| CudaAdaptiveMacdError::MissingKernelSymbol {
                name: "adaptive_macd_batch_f64",
            })?;
        let grid_x = ((rows as u32) + ADAPTIVE_MACD_BLOCK_X - 1) / ADAPTIVE_MACD_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ADAPTIVE_MACD_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_fast_periods.as_device_ptr(),
                d_slow_periods.as_device_ptr(),
                d_signal_periods.as_device_ptr(),
                rows as i32,
                d_out_macd.as_device_ptr(),
                d_out_signal.as_device_ptr(),
                d_out_hist.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaAdaptiveMacdBatchResult {
            outputs: AdaptiveMacdDeviceArrayF64Triple {
                macd: AdaptiveMacdDeviceArrayF64 {
                    buf: d_out_macd,
                    rows,
                    cols,
                },
                signal: AdaptiveMacdDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
                hist: AdaptiveMacdDeviceArrayF64 {
                    buf: d_out_hist,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
