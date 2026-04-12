#![cfg(feature = "cuda")]

use crate::indicators::andean_oscillator::{
    expand_grid, AndeanOscillatorBatchRange, AndeanOscillatorParams,
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

const ANDEAN_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 50;
const DEFAULT_SIGNAL_LENGTH: usize = 9;

#[derive(Debug, Error)]
pub enum CudaAndeanOscillatorError {
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

pub struct AndeanOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AndeanOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct AndeanOscillatorDeviceArrayF64Trio {
    pub bull: AndeanOscillatorDeviceArrayF64,
    pub bear: AndeanOscillatorDeviceArrayF64,
    pub signal: AndeanOscillatorDeviceArrayF64,
}

impl AndeanOscillatorDeviceArrayF64Trio {
    #[inline]
    pub fn rows(&self) -> usize {
        self.bull.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.bull.cols
    }
}

pub struct CudaAndeanOscillatorBatchResult {
    pub outputs: AndeanOscillatorDeviceArrayF64Trio,
    pub combos: Vec<AndeanOscillatorParams>,
}

pub struct CudaAndeanOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAndeanOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaAndeanOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("andean_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAndeanOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_pair(open: &[f64], close: &[f64]) -> Option<usize> {
        (0..open.len()).find(|&i| open[i].is_finite() && close[i].is_finite())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaAndeanOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAndeanOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaAndeanOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaAndeanOscillatorError::LaunchConfigTooLarge {
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
        sweep: &AndeanOscillatorBatchRange,
    ) -> Result<CudaAndeanOscillatorBatchResult, CudaAndeanOscillatorError> {
        if open.is_empty() || close.is_empty() {
            return Err(CudaAndeanOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != close.len() {
            return Err(CudaAndeanOscillatorError::InvalidInput(format!(
                "input length mismatch: open={}, close={}",
                open.len(),
                close.len()
            )));
        }
        Self::first_valid_pair(open, close)
            .ok_or_else(|| CudaAndeanOscillatorError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep)
            .map_err(|err| CudaAndeanOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaAndeanOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = open.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut signal_lengths = Vec::with_capacity(rows);
        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let signal_length = combo.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH);
            if length == 0 || signal_length == 0 {
                return Err(CudaAndeanOscillatorError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            lengths.push(length as i32);
            signal_lengths.push(signal_length as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaAndeanOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaAndeanOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAndeanOscillatorError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaAndeanOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaAndeanOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_signal_lengths = DeviceBuffer::from_slice(&signal_lengths)?;
        let d_out_bull = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bear = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("andean_oscillator_batch_f64")
            .map_err(|_| CudaAndeanOscillatorError::MissingKernelSymbol {
                name: "andean_oscillator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + ANDEAN_OSCILLATOR_BLOCK_X - 1) / ANDEAN_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ANDEAN_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_signal_lengths.as_device_ptr(),
                rows as i32,
                d_out_bull.as_device_ptr(),
                d_out_bear.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaAndeanOscillatorBatchResult {
            outputs: AndeanOscillatorDeviceArrayF64Trio {
                bull: AndeanOscillatorDeviceArrayF64 {
                    buf: d_out_bull,
                    rows,
                    cols,
                },
                bear: AndeanOscillatorDeviceArrayF64 {
                    buf: d_out_bear,
                    rows,
                    cols,
                },
                signal: AndeanOscillatorDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
