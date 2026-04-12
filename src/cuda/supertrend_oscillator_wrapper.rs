#![cfg(feature = "cuda")]

use crate::indicators::supertrend_oscillator::{
    expand_grid_supertrend_oscillator, SuperTrendOscillatorBatchRange, SuperTrendOscillatorParams,
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

const SUPERTREND_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 10;
const DEFAULT_MULT: f64 = 2.0;
const DEFAULT_SMOOTH: usize = 72;

#[derive(Debug, Error)]
pub enum CudaSupertrendOscillatorError {
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

pub struct SupertrendOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl SupertrendOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct SupertrendOscillatorDeviceArrayF64Triple {
    pub oscillator: SupertrendOscillatorDeviceArrayF64,
    pub signal: SupertrendOscillatorDeviceArrayF64,
    pub histogram: SupertrendOscillatorDeviceArrayF64,
}

impl SupertrendOscillatorDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaSupertrendOscillatorBatchResult {
    pub outputs: SupertrendOscillatorDeviceArrayF64Triple,
    pub combos: Vec<SuperTrendOscillatorParams>,
}

pub struct CudaSupertrendOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaSupertrendOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaSupertrendOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("supertrend_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaSupertrendOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn valid_bar(high: f64, low: f64, source: f64) -> bool {
        high.is_finite() && low.is_finite() && source.is_finite() && high >= low
    }

    fn first_valid_bar(high: &[f64], low: &[f64], source: &[f64]) -> Option<usize> {
        (0..source.len()).find(|&i| Self::valid_bar(high[i], low[i], source[i]))
    }

    fn max_valid_run(high: &[f64], low: &[f64], source: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for i in 0..source.len() {
            if Self::valid_bar(high[i], low[i], source[i]) {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        best
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaSupertrendOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaSupertrendOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaSupertrendOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaSupertrendOscillatorError::LaunchConfigTooLarge {
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
        source: &[f64],
        sweep: &SuperTrendOscillatorBatchRange,
    ) -> Result<CudaSupertrendOscillatorBatchResult, CudaSupertrendOscillatorError> {
        if high.is_empty() || low.is_empty() || source.is_empty() {
            return Err(CudaSupertrendOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || low.len() != source.len() {
            return Err(CudaSupertrendOscillatorError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, source={}",
                high.len(),
                low.len(),
                source.len()
            )));
        }
        Self::first_valid_bar(high, low, source).ok_or_else(|| {
            CudaSupertrendOscillatorError::InvalidInput("all values are NaN".into())
        })?;

        let combos = expand_grid_supertrend_oscillator(sweep)
            .map_err(|err| CudaSupertrendOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaSupertrendOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let max_run = Self::max_valid_run(high, low, source);
        let rows = combos.len();
        let cols = source.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut mults = Vec::with_capacity(rows);
        let mut smooths = Vec::with_capacity(rows);
        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let mult = combo.mult.unwrap_or(DEFAULT_MULT);
            let smooth = combo.smooth.unwrap_or(DEFAULT_SMOOTH);
            if length == 0 || length > cols {
                return Err(CudaSupertrendOscillatorError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            if !mult.is_finite() || mult <= 0.0 {
                return Err(CudaSupertrendOscillatorError::InvalidInput(format!(
                    "invalid multiplier: {mult}"
                )));
            }
            if smooth == 0 {
                return Err(CudaSupertrendOscillatorError::InvalidInput(format!(
                    "invalid smooth: {smooth}"
                )));
            }
            if max_run < length {
                return Err(CudaSupertrendOscillatorError::InvalidInput(format!(
                    "not enough valid data: needed={length}, valid={max_run}"
                )));
            }
            lengths.push(i32::try_from(length).map_err(|_| {
                CudaSupertrendOscillatorError::InvalidInput(format!(
                    "length out of range: {length}"
                ))
            })?);
            mults.push(mult);
            smooths.push(i32::try_from(smooth).map_err(|_| {
                CudaSupertrendOscillatorError::InvalidInput(format!(
                    "smooth out of range: {smooth}"
                ))
            })?);
        }

        let rows_i32 = i32::try_from(rows)
            .map_err(|_| CudaSupertrendOscillatorError::InvalidInput("rows out of range".into()))?;
        let cols_i32 = i32::try_from(cols)
            .map_err(|_| CudaSupertrendOscillatorError::InvalidInput("cols out of range".into()))?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaSupertrendOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaSupertrendOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaSupertrendOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaSupertrendOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaSupertrendOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_source = DeviceBuffer::from_slice(source)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_mults = DeviceBuffer::from_slice(&mults)?;
        let d_smooths = DeviceBuffer::from_slice(&smooths)?;
        let d_out_oscillator = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_histogram = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("supertrend_oscillator_batch_f64")
            .map_err(|_| CudaSupertrendOscillatorError::MissingKernelSymbol {
                name: "supertrend_oscillator_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + SUPERTREND_OSCILLATOR_BLOCK_X - 1) / SUPERTREND_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(SUPERTREND_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_source.as_device_ptr(),
                cols_i32,
                d_lengths.as_device_ptr(),
                d_mults.as_device_ptr(),
                d_smooths.as_device_ptr(),
                rows_i32,
                d_out_oscillator.as_device_ptr(),
                d_out_signal.as_device_ptr(),
                d_out_histogram.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaSupertrendOscillatorBatchResult {
            outputs: SupertrendOscillatorDeviceArrayF64Triple {
                oscillator: SupertrendOscillatorDeviceArrayF64 {
                    buf: d_out_oscillator,
                    rows,
                    cols,
                },
                signal: SupertrendOscillatorDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
                histogram: SupertrendOscillatorDeviceArrayF64 {
                    buf: d_out_histogram,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
