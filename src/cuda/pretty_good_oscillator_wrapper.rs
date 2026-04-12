#![cfg(feature = "cuda")]

use crate::indicators::pretty_good_oscillator::{
    PrettyGoodOscillatorBatchRange, PrettyGoodOscillatorParams,
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

const PRETTY_GOOD_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaPrettyGoodOscillatorError {
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

pub struct PrettyGoodOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl PrettyGoodOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaPrettyGoodOscillatorBatchResult {
    pub outputs: PrettyGoodOscillatorDeviceArrayF64,
    pub combos: Vec<PrettyGoodOscillatorParams>,
}

pub struct CudaPrettyGoodOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaPrettyGoodOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaPrettyGoodOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("pretty_good_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaPrettyGoodOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn expand_grid(
        sweep: &PrettyGoodOscillatorBatchRange,
    ) -> Result<Vec<PrettyGoodOscillatorParams>, CudaPrettyGoodOscillatorError> {
        let (start, end, step) = sweep.length;
        if start == 0 || end == 0 || start > end || (start != end && step == 0) {
            return Err(CudaPrettyGoodOscillatorError::InvalidInput(format!(
                "invalid length range: start={start}, end={end}, step={step}"
            )));
        }

        let mut combos = Vec::new();
        let mut value = start;
        loop {
            combos.push(PrettyGoodOscillatorParams {
                length: Some(value),
            });
            if value == end {
                break;
            }
            value = value.saturating_add(step);
            if value > end {
                break;
            }
        }

        Ok(combos)
    }

    fn is_valid_bar(high: f64, low: f64, close: f64, source: f64) -> bool {
        high.is_finite()
            && low.is_finite()
            && close.is_finite()
            && source.is_finite()
            && high >= low
    }

    fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64], source: &[f64]) -> Option<usize> {
        (0..source.len()).find(|&i| Self::is_valid_bar(high[i], low[i], close[i], source[i]))
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaPrettyGoodOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaPrettyGoodOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaPrettyGoodOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaPrettyGoodOscillatorError::LaunchConfigTooLarge {
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
        source: &[f64],
        sweep: &PrettyGoodOscillatorBatchRange,
    ) -> Result<CudaPrettyGoodOscillatorBatchResult, CudaPrettyGoodOscillatorError> {
        if high.is_empty() {
            return Err(CudaPrettyGoodOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || low.len() != close.len() || close.len() != source.len() {
            return Err(CudaPrettyGoodOscillatorError::InvalidInput(
                "data length mismatch across high, low, close, and source".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(14))
            .max()
            .unwrap_or(0);
        if max_length == 0 || max_length > source.len() {
            return Err(CudaPrettyGoodOscillatorError::InvalidInput(format!(
                "invalid length: length={max_length}, data_len={}",
                source.len()
            )));
        }

        let first = Self::first_valid_bar(high, low, close, source).ok_or_else(|| {
            CudaPrettyGoodOscillatorError::InvalidInput("all OHLC/source values are invalid".into())
        })?;
        let valid = source.len().saturating_sub(first);
        if valid < max_length {
            return Err(CudaPrettyGoodOscillatorError::InvalidInput(format!(
                "not enough valid data: needed={max_length}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = source.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(14) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| {
                CudaPrettyGoodOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaPrettyGoodOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaPrettyGoodOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaPrettyGoodOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaPrettyGoodOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_source = DeviceBuffer::from_slice(source)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("pretty_good_oscillator_batch_f64")
            .map_err(|_| CudaPrettyGoodOscillatorError::MissingKernelSymbol {
                name: "pretty_good_oscillator_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + PRETTY_GOOD_OSCILLATOR_BLOCK_X - 1) / PRETTY_GOOD_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(PRETTY_GOOD_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_source.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaPrettyGoodOscillatorBatchResult {
            outputs: PrettyGoodOscillatorDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
