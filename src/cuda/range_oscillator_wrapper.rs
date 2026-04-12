#![cfg(feature = "cuda")]

use crate::indicators::range_oscillator::{RangeOscillatorBatchRange, RangeOscillatorParams};
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

const RANGE_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 50;
const DEFAULT_MULT: f64 = 2.0;
const ATR_FALLBACK_PERIOD: usize = 200;

#[derive(Debug, Error)]
pub enum CudaRangeOscillatorError {
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

pub struct RangeOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl RangeOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct RangeOscillatorDeviceArrayF64Nine {
    pub oscillator: RangeOscillatorDeviceArrayF64,
    pub ma: RangeOscillatorDeviceArrayF64,
    pub upper_band: RangeOscillatorDeviceArrayF64,
    pub lower_band: RangeOscillatorDeviceArrayF64,
    pub range_width: RangeOscillatorDeviceArrayF64,
    pub in_range: RangeOscillatorDeviceArrayF64,
    pub trend: RangeOscillatorDeviceArrayF64,
    pub break_up: RangeOscillatorDeviceArrayF64,
    pub break_down: RangeOscillatorDeviceArrayF64,
}

impl RangeOscillatorDeviceArrayF64Nine {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaRangeOscillatorBatchResult {
    pub outputs: RangeOscillatorDeviceArrayF64Nine,
    pub combos: Vec<RangeOscillatorParams>,
}

pub struct CudaRangeOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaRangeOscillatorError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start <= end {
        let mut current = start;
        while current <= end {
            out.push(current);
            match current.checked_add(step) {
                Some(next) => current = next,
                None => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            out.push(current);
            match current.checked_sub(step) {
                Some(next) => current = next,
                None => break,
            }
            if current < end {
                break;
            }
        }
    }

    if out.is_empty() {
        return Err(CudaRangeOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_axis_f64(start: f64, end: f64, step: f64) -> Result<Vec<f64>, CudaRangeOscillatorError> {
    let eps = 1e-12;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaRangeOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step.abs() < eps || (start - end).abs() < eps {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    let dir = if end >= start { 1.0 } else { -1.0 };
    let step_eff = dir * step.abs();
    let mut current = start;
    if dir > 0.0 {
        while current <= end + eps {
            out.push(current);
            current += step_eff;
        }
    } else {
        while current >= end - eps {
            out.push(current);
            current += step_eff;
        }
    }

    if out.is_empty() {
        return Err(CudaRangeOscillatorError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_grid_range_oscillator(
    range: &RangeOscillatorBatchRange,
) -> Result<Vec<RangeOscillatorParams>, CudaRangeOscillatorError> {
    let lengths = expand_axis_usize(range.length.0, range.length.1, range.length.2)?;
    let mults = expand_axis_f64(range.mult.0, range.mult.1, range.mult.2)?;
    let total = lengths
        .len()
        .checked_mul(mults.len())
        .ok_or_else(|| CudaRangeOscillatorError::InvalidInput("parameter grid overflow".into()))?;

    let mut combos = Vec::with_capacity(total);
    for &length in &lengths {
        for &mult in &mults {
            combos.push(RangeOscillatorParams {
                length: Some(length),
                mult: Some(mult),
            });
        }
    }
    Ok(combos)
}

impl CudaRangeOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaRangeOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("range_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaRangeOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_triplet(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len())
            .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
    }

    fn count_valid_from(high: &[f64], low: &[f64], close: &[f64], start: usize) -> usize {
        (start..close.len())
            .filter(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
            .count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaRangeOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaRangeOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaRangeOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaRangeOscillatorError::LaunchConfigTooLarge {
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
        sweep: &RangeOscillatorBatchRange,
    ) -> Result<CudaRangeOscillatorBatchResult, CudaRangeOscillatorError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaRangeOscillatorError::InvalidInput("empty input".into()));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaRangeOscillatorError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let first = Self::first_valid_triplet(high, low, close)
            .ok_or_else(|| CudaRangeOscillatorError::InvalidInput("all values are NaN".into()))?;
        let valid = Self::count_valid_from(high, low, close, first);

        let combos = expand_grid_range_oscillator(sweep)?;
        if combos.is_empty() {
            return Err(CudaRangeOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut mults = Vec::with_capacity(rows);
        let mut max_length = 0usize;
        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let mult = combo.mult.unwrap_or(DEFAULT_MULT);
            if length == 0 || length >= cols {
                return Err(CudaRangeOscillatorError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            if !mult.is_finite() || mult < 0.1 {
                return Err(CudaRangeOscillatorError::InvalidInput(format!(
                    "invalid mult: {mult}"
                )));
            }
            let needed = (length + 1).max(ATR_FALLBACK_PERIOD);
            if valid < needed {
                return Err(CudaRangeOscillatorError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            max_length = max_length.max(length);
            lengths.push(i32::try_from(length).map_err(|_| {
                CudaRangeOscillatorError::InvalidInput(format!("length out of range: {length}"))
            })?);
            mults.push(mult);
        }

        let rows_i32 = i32::try_from(rows)
            .map_err(|_| CudaRangeOscillatorError::InvalidInput("rows out of range".into()))?;
        let cols_i32 = i32::try_from(cols)
            .map_err(|_| CudaRangeOscillatorError::InvalidInput("cols out of range".into()))?;
        let storage_cols = max_length.checked_add(1).ok_or_else(|| {
            CudaRangeOscillatorError::InvalidInput("storage cols overflow".into())
        })?;
        let storage_cols_i32 = i32::try_from(storage_cols).map_err(|_| {
            CudaRangeOscillatorError::InvalidInput("storage cols out of range".into())
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| CudaRangeOscillatorError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaRangeOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaRangeOscillatorError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(9))
            .ok_or_else(|| {
                CudaRangeOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_elems = rows.checked_mul(storage_cols).ok_or_else(|| {
            CudaRangeOscillatorError::InvalidInput("scratch elements overflow".into())
        })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaRangeOscillatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaRangeOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_mults = DeviceBuffer::from_slice(&mults)?;
        let d_out_oscillator = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_upper_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_lower_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_range_width = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_in_range = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_trend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_break_up = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_break_down = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_close_storage = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };

        let func = self
            .module
            .get_function("range_oscillator_batch_f64")
            .map_err(|_| CudaRangeOscillatorError::MissingKernelSymbol {
                name: "range_oscillator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + RANGE_OSCILLATOR_BLOCK_X - 1) / RANGE_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(RANGE_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols_i32,
                d_lengths.as_device_ptr(),
                d_mults.as_device_ptr(),
                rows_i32,
                storage_cols_i32,
                d_out_oscillator.as_device_ptr(),
                d_out_ma.as_device_ptr(),
                d_out_upper_band.as_device_ptr(),
                d_out_lower_band.as_device_ptr(),
                d_out_range_width.as_device_ptr(),
                d_out_in_range.as_device_ptr(),
                d_out_trend.as_device_ptr(),
                d_out_break_up.as_device_ptr(),
                d_out_break_down.as_device_ptr(),
                d_close_storage.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaRangeOscillatorBatchResult {
            outputs: RangeOscillatorDeviceArrayF64Nine {
                oscillator: RangeOscillatorDeviceArrayF64 {
                    buf: d_out_oscillator,
                    rows,
                    cols,
                },
                ma: RangeOscillatorDeviceArrayF64 {
                    buf: d_out_ma,
                    rows,
                    cols,
                },
                upper_band: RangeOscillatorDeviceArrayF64 {
                    buf: d_out_upper_band,
                    rows,
                    cols,
                },
                lower_band: RangeOscillatorDeviceArrayF64 {
                    buf: d_out_lower_band,
                    rows,
                    cols,
                },
                range_width: RangeOscillatorDeviceArrayF64 {
                    buf: d_out_range_width,
                    rows,
                    cols,
                },
                in_range: RangeOscillatorDeviceArrayF64 {
                    buf: d_out_in_range,
                    rows,
                    cols,
                },
                trend: RangeOscillatorDeviceArrayF64 {
                    buf: d_out_trend,
                    rows,
                    cols,
                },
                break_up: RangeOscillatorDeviceArrayF64 {
                    buf: d_out_break_up,
                    rows,
                    cols,
                },
                break_down: RangeOscillatorDeviceArrayF64 {
                    buf: d_out_break_down,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
