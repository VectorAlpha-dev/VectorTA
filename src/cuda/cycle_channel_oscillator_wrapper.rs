#![cfg(feature = "cuda")]

use crate::indicators::cycle_channel_oscillator::{
    expand_grid, CycleChannelOscillatorBatchRange, CycleChannelOscillatorParams,
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

const CYCLE_CHANNEL_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_MEDIUM_CYCLE_LENGTH: usize = 30;

#[derive(Debug, Error)]
pub enum CudaCycleChannelOscillatorError {
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

pub struct CycleChannelOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl CycleChannelOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CycleChannelOscillatorDeviceArrayF64Pair {
    pub fast: CycleChannelOscillatorDeviceArrayF64,
    pub slow: CycleChannelOscillatorDeviceArrayF64,
}

impl CycleChannelOscillatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.fast.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.fast.cols
    }
}

pub struct CudaCycleChannelOscillatorBatchResult {
    pub outputs: CycleChannelOscillatorDeviceArrayF64Pair,
    pub combos: Vec<CycleChannelOscillatorParams>,
}

pub struct CudaCycleChannelOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaCycleChannelOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaCycleChannelOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("cycle_channel_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaCycleChannelOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_quad(source: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..source.len()).find(|&i| {
            source[i].is_finite()
                && high[i].is_finite()
                && low[i].is_finite()
                && close[i].is_finite()
        })
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaCycleChannelOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaCycleChannelOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaCycleChannelOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaCycleChannelOscillatorError::LaunchConfigTooLarge {
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
        source: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &CycleChannelOscillatorBatchRange,
    ) -> Result<CudaCycleChannelOscillatorBatchResult, CudaCycleChannelOscillatorError> {
        if source.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaCycleChannelOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if source.len() != high.len() || source.len() != low.len() || source.len() != close.len() {
            return Err(CudaCycleChannelOscillatorError::InvalidInput(format!(
                "input length mismatch: source={}, high={}, low={}, close={}",
                source.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let first = Self::first_valid_quad(source, high, low, close).ok_or_else(|| {
            CudaCycleChannelOscillatorError::InvalidInput("all values are NaN".into())
        })?;

        let combos = expand_grid(sweep)
            .map_err(|err| CudaCycleChannelOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaCycleChannelOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = source.len();
        let valid = cols.saturating_sub(first);
        let mut short_cycle_lengths = Vec::with_capacity(rows);
        let mut medium_cycle_lengths = Vec::with_capacity(rows);
        let mut short_multipliers = Vec::with_capacity(rows);
        let mut medium_multipliers = Vec::with_capacity(rows);
        let mut max_needed = DEFAULT_MEDIUM_CYCLE_LENGTH / 2;

        for combo in &combos {
            let short_cycle_length = combo.short_cycle_length.ok_or_else(|| {
                CudaCycleChannelOscillatorError::InvalidInput(
                    "missing short_cycle_length".to_string(),
                )
            })?;
            let medium_cycle_length = combo.medium_cycle_length.ok_or_else(|| {
                CudaCycleChannelOscillatorError::InvalidInput(
                    "missing medium_cycle_length".to_string(),
                )
            })?;
            let short_multiplier = combo.short_multiplier.ok_or_else(|| {
                CudaCycleChannelOscillatorError::InvalidInput(
                    "missing short_multiplier".to_string(),
                )
            })?;
            let medium_multiplier = combo.medium_multiplier.ok_or_else(|| {
                CudaCycleChannelOscillatorError::InvalidInput(
                    "missing medium_multiplier".to_string(),
                )
            })?;

            if short_cycle_length < 2 {
                return Err(CudaCycleChannelOscillatorError::InvalidInput(format!(
                    "invalid short_cycle_length: {short_cycle_length}"
                )));
            }
            if medium_cycle_length < 2 {
                return Err(CudaCycleChannelOscillatorError::InvalidInput(format!(
                    "invalid medium_cycle_length: {medium_cycle_length}"
                )));
            }
            if !short_multiplier.is_finite() || short_multiplier < 0.0 {
                return Err(CudaCycleChannelOscillatorError::InvalidInput(format!(
                    "invalid short_multiplier: {short_multiplier}"
                )));
            }
            if !medium_multiplier.is_finite() || medium_multiplier < 0.0 {
                return Err(CudaCycleChannelOscillatorError::InvalidInput(format!(
                    "invalid medium_multiplier: {medium_multiplier}"
                )));
            }

            max_needed = max_needed.max(medium_cycle_length / 2);
            short_cycle_lengths.push(i32::try_from(short_cycle_length).map_err(|_| {
                CudaCycleChannelOscillatorError::InvalidInput(format!(
                    "short_cycle_length out of range: {short_cycle_length}"
                ))
            })?);
            medium_cycle_lengths.push(i32::try_from(medium_cycle_length).map_err(|_| {
                CudaCycleChannelOscillatorError::InvalidInput(format!(
                    "medium_cycle_length out of range: {medium_cycle_length}"
                ))
            })?);
            short_multipliers.push(short_multiplier);
            medium_multipliers.push(medium_multiplier);
        }

        if valid < max_needed {
            return Err(CudaCycleChannelOscillatorError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={valid}"
            )));
        }

        let rows_i32 = i32::try_from(rows).map_err(|_| {
            CudaCycleChannelOscillatorError::InvalidInput("rows out of range".into())
        })?;
        let cols_i32 = i32::try_from(cols).map_err(|_| {
            CudaCycleChannelOscillatorError::InvalidInput("cols out of range".into())
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaCycleChannelOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f64>() * 2)
            .ok_or_else(|| {
                CudaCycleChannelOscillatorError::InvalidInput("param bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaCycleChannelOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaCycleChannelOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaCycleChannelOscillatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaCycleChannelOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source = DeviceBuffer::from_slice(source)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_short_cycle_lengths = DeviceBuffer::from_slice(&short_cycle_lengths)?;
        let d_medium_cycle_lengths = DeviceBuffer::from_slice(&medium_cycle_lengths)?;
        let d_short_multipliers = DeviceBuffer::from_slice(&short_multipliers)?;
        let d_medium_multipliers = DeviceBuffer::from_slice(&medium_multipliers)?;
        let d_out_fast = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_slow = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_short_history = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_medium_history = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("cycle_channel_oscillator_batch_f64")
            .map_err(|_| CudaCycleChannelOscillatorError::MissingKernelSymbol {
                name: "cycle_channel_oscillator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + CYCLE_CHANNEL_OSCILLATOR_BLOCK_X - 1)
            / CYCLE_CHANNEL_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(CYCLE_CHANNEL_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols_i32,
                d_short_cycle_lengths.as_device_ptr(),
                d_medium_cycle_lengths.as_device_ptr(),
                d_short_multipliers.as_device_ptr(),
                d_medium_multipliers.as_device_ptr(),
                rows_i32,
                d_out_fast.as_device_ptr(),
                d_out_slow.as_device_ptr(),
                d_short_history.as_device_ptr(),
                d_medium_history.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaCycleChannelOscillatorBatchResult {
            outputs: CycleChannelOscillatorDeviceArrayF64Pair {
                fast: CycleChannelOscillatorDeviceArrayF64 {
                    buf: d_out_fast,
                    rows,
                    cols,
                },
                slow: CycleChannelOscillatorDeviceArrayF64 {
                    buf: d_out_slow,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
