#![cfg(feature = "cuda")]

use crate::indicators::premier_rsi_oscillator::{
    PremierRsiOscillatorBatchRange, PremierRsiOscillatorParams,
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

const PREMIER_RSI_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_STOCH_LENGTH: usize = 8;
const DEFAULT_SMOOTH_LENGTH: usize = 25;

#[derive(Debug, Error)]
pub enum CudaPremierRsiOscillatorError {
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

pub struct PremierRsiOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl PremierRsiOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
}

pub struct CudaPremierRsiOscillatorBatchResult {
    pub outputs: PremierRsiOscillatorDeviceArrayF64,
    pub combos: Vec<PremierRsiOscillatorParams>,
}

pub struct CudaPremierRsiOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaPremierRsiOscillatorError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(CudaPremierRsiOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }

    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end {
            out.push(x);
            let next = x.saturating_add(step);
            if next == x {
                break;
            }
            x = next;
        }
    } else {
        let mut x = start;
        while x >= end {
            out.push(x);
            let next = x.saturating_sub(step);
            if next == x {
                break;
            }
            x = next;
        }
    }

    if out.is_empty() {
        return Err(CudaPremierRsiOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_grid_premier_rsi_oscillator(
    range: &PremierRsiOscillatorBatchRange,
) -> Result<Vec<PremierRsiOscillatorParams>, CudaPremierRsiOscillatorError> {
    let rsi_lengths =
        expand_axis_usize(range.rsi_length.0, range.rsi_length.1, range.rsi_length.2)?;
    let stoch_lengths = expand_axis_usize(
        range.stoch_length.0,
        range.stoch_length.1,
        range.stoch_length.2,
    )?;
    let smooth_lengths = expand_axis_usize(
        range.smooth_length.0,
        range.smooth_length.1,
        range.smooth_length.2,
    )?;

    let mut combos = Vec::with_capacity(
        rsi_lengths
            .len()
            .saturating_mul(stoch_lengths.len())
            .saturating_mul(smooth_lengths.len()),
    );
    for &rsi_length in &rsi_lengths {
        for &stoch_length in &stoch_lengths {
            for &smooth_length in &smooth_lengths {
                combos.push(PremierRsiOscillatorParams {
                    rsi_length: Some(rsi_length),
                    stoch_length: Some(stoch_length),
                    smooth_length: Some(smooth_length),
                });
            }
        }
    }
    Ok(combos)
}

impl CudaPremierRsiOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaPremierRsiOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("premier_rsi_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaPremierRsiOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|v| v.is_finite())
    }

    fn count_valid_values(data: &[f64]) -> usize {
        data.iter().filter(|v| v.is_finite()).count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaPremierRsiOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaPremierRsiOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaPremierRsiOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaPremierRsiOscillatorError::LaunchConfigTooLarge {
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
        sweep: &PremierRsiOscillatorBatchRange,
    ) -> Result<CudaPremierRsiOscillatorBatchResult, CudaPremierRsiOscillatorError> {
        if data.is_empty() {
            return Err(CudaPremierRsiOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        Self::first_valid_value(data).ok_or_else(|| {
            CudaPremierRsiOscillatorError::InvalidInput("all values are NaN".into())
        })?;
        let valid = Self::count_valid_values(data);

        let combos = expand_grid_premier_rsi_oscillator(sweep)?;
        if combos.is_empty() {
            return Err(CudaPremierRsiOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut rsi_lengths = Vec::with_capacity(rows);
        let mut stoch_lengths = Vec::with_capacity(rows);
        let mut smooth_lengths = Vec::with_capacity(rows);

        for combo in &combos {
            let rsi_length = combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
            let stoch_length = combo.stoch_length.unwrap_or(DEFAULT_STOCH_LENGTH);
            let smooth_length = combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH);
            if rsi_length == 0 || stoch_length == 0 || smooth_length == 0 {
                return Err(CudaPremierRsiOscillatorError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            let needed = rsi_length
                .checked_add(stoch_length)
                .and_then(|v| v.checked_sub(1))
                .ok_or_else(|| {
                    CudaPremierRsiOscillatorError::InvalidInput("needed bars overflow".into())
                })?;
            if valid < needed {
                return Err(CudaPremierRsiOscillatorError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            rsi_lengths.push(rsi_length as i32);
            stoch_lengths.push(stoch_length as i32);
            smooth_lengths.push(smooth_length as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaPremierRsiOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| {
                CudaPremierRsiOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaPremierRsiOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaPremierRsiOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaPremierRsiOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_rsi_lengths = DeviceBuffer::from_slice(&rsi_lengths)?;
        let d_stoch_lengths = DeviceBuffer::from_slice(&stoch_lengths)?;
        let d_smooth_lengths = DeviceBuffer::from_slice(&smooth_lengths)?;
        let d_out_values = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("premier_rsi_oscillator_batch_f64")
            .map_err(|_| CudaPremierRsiOscillatorError::MissingKernelSymbol {
                name: "premier_rsi_oscillator_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + PREMIER_RSI_OSCILLATOR_BLOCK_X - 1) / PREMIER_RSI_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(PREMIER_RSI_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_rsi_lengths.as_device_ptr(),
                d_stoch_lengths.as_device_ptr(),
                d_smooth_lengths.as_device_ptr(),
                rows as i32,
                d_out_values.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaPremierRsiOscillatorBatchResult {
            outputs: PremierRsiOscillatorDeviceArrayF64 {
                buf: d_out_values,
                rows,
                cols,
            },
            combos,
        })
    }
}
