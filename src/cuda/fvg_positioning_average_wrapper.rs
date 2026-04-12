#![cfg(feature = "cuda")]

use crate::indicators::fvg_positioning_average::{
    FvgPositioningAverageBatchRange, FvgPositioningAverageParams,
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

const FVG_POSITIONING_AVERAGE_BLOCK_X: u32 = 64;
const DEFAULT_LOOKBACK: usize = 30;
const DEFAULT_ATR_MULTIPLIER: f64 = 0.25;
const LOOKBACK_TYPE_BAR_COUNT: &str = "Bar Count";
const LOOKBACK_TYPE_FVG_COUNT: &str = "FVG Count";
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaFvgPositioningAverageError {
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

pub struct FvgPositioningAverageDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl FvgPositioningAverageDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct FvgPositioningAverageDeviceOutputs {
    pub bull_average: FvgPositioningAverageDeviceArrayF64,
    pub bear_average: FvgPositioningAverageDeviceArrayF64,
    pub bull_mid: FvgPositioningAverageDeviceArrayF64,
    pub bear_mid: FvgPositioningAverageDeviceArrayF64,
}

impl FvgPositioningAverageDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.bull_average.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.bull_average.cols
    }
}

pub struct CudaFvgPositioningAverageBatchResult {
    pub outputs: FvgPositioningAverageDeviceOutputs,
    pub combos: Vec<FvgPositioningAverageParams>,
}

pub struct CudaFvgPositioningAverage {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn canonical_lookback_type(value: &str) -> Result<&'static str, CudaFvgPositioningAverageError> {
    if value.eq_ignore_ascii_case(LOOKBACK_TYPE_BAR_COUNT) {
        return Ok(LOOKBACK_TYPE_BAR_COUNT);
    }
    if value.eq_ignore_ascii_case(LOOKBACK_TYPE_FVG_COUNT) {
        return Ok(LOOKBACK_TYPE_FVG_COUNT);
    }
    Err(CudaFvgPositioningAverageError::InvalidInput(format!(
        "invalid lookback_type: {value}"
    )))
}

#[inline]
fn lookback_mode_id(value: &str) -> Result<i32, CudaFvgPositioningAverageError> {
    match canonical_lookback_type(value)? {
        LOOKBACK_TYPE_BAR_COUNT => Ok(0),
        LOOKBACK_TYPE_FVG_COUNT => Ok(1),
        _ => unreachable!(),
    }
}

#[inline]
fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaFvgPositioningAverageError> {
    if start == 0 || end == 0 {
        return Err(CudaFvgPositioningAverageError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(CudaFvgPositioningAverageError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }

    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end {
            break;
        }
        let next = current.saturating_add(step);
        if next <= current {
            return Err(CudaFvgPositioningAverageError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        current = next.min(end);
        if current == *out.last().unwrap_or(&0) {
            break;
        }
    }
    Ok(out)
}

#[inline]
fn expand_f64_range(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaFvgPositioningAverageError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaFvgPositioningAverageError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(CudaFvgPositioningAverageError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }

    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end || (end - current).abs() <= 1.0e-12 {
            break;
        }
        let next = current + step;
        if next <= current {
            return Err(CudaFvgPositioningAverageError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        current = if next > end { end } else { next };
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    range: &FvgPositioningAverageBatchRange,
    lookback_type: &str,
) -> Result<Vec<FvgPositioningAverageParams>, CudaFvgPositioningAverageError> {
    let lookbacks = expand_usize_range(range.lookback.0, range.lookback.1, range.lookback.2)?;
    let atr_multipliers = expand_f64_range(
        range.atr_multiplier.0,
        range.atr_multiplier.1,
        range.atr_multiplier.2,
    )?;
    let lookback_type = canonical_lookback_type(lookback_type)?;
    let cap = lookbacks
        .len()
        .checked_mul(atr_multipliers.len())
        .ok_or_else(|| {
            CudaFvgPositioningAverageError::InvalidInput("parameter grid overflow".into())
        })?;
    let mut out = Vec::with_capacity(cap);
    for &lookback in &lookbacks {
        for &atr_multiplier in &atr_multipliers {
            out.push(FvgPositioningAverageParams {
                lookback: Some(lookback),
                lookback_type: Some(lookback_type.to_string()),
                atr_multiplier: Some(atr_multiplier),
            });
        }
    }
    Ok(out)
}

#[inline]
fn is_valid_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline]
fn longest_valid_run(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for (((open, high), low), close) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
    {
        if is_valid_ohlc(*open, *high, *low, *close) {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

impl CudaFvgPositioningAverage {
    pub fn new(device_id: usize) -> Result<Self, CudaFvgPositioningAverageError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("fvg_positioning_average_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaFvgPositioningAverageError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaFvgPositioningAverageError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaFvgPositioningAverageError::OutOfMemory {
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
    ) -> Result<(), CudaFvgPositioningAverageError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaFvgPositioningAverageError::LaunchConfigTooLarge {
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
        sweep: &FvgPositioningAverageBatchRange,
        lookback_type: &str,
    ) -> Result<CudaFvgPositioningAverageBatchResult, CudaFvgPositioningAverageError> {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaFvgPositioningAverageError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaFvgPositioningAverageError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let lookback_mode = lookback_mode_id(lookback_type)?;
        let combos = expand_grid_checked(sweep, lookback_type)?;
        let longest = longest_valid_run(open, high, low, close);
        if longest == 0 {
            return Err(CudaFvgPositioningAverageError::InvalidInput(
                "all values are NaN".into(),
            ));
        }
        if longest < 3 {
            return Err(CudaFvgPositioningAverageError::InvalidInput(format!(
                "not enough valid data: needed=3, valid={longest}"
            )));
        }

        let rows = combos.len();
        let cols = close.len();
        let max_lookback = combos
            .iter()
            .map(|params| params.lookback.unwrap_or(DEFAULT_LOOKBACK))
            .max()
            .unwrap_or(0);
        let max_atr_multiplier = combos
            .iter()
            .map(|params| params.atr_multiplier.unwrap_or(DEFAULT_ATR_MULTIPLIER))
            .fold(0.0_f64, f64::max);
        if max_lookback == 0 || !max_atr_multiplier.is_finite() || max_atr_multiplier < 0.0 {
            return Err(CudaFvgPositioningAverageError::InvalidInput(
                "invalid parameters".into(),
            ));
        }

        let level_cap = max_lookback.saturating_add(2);
        let lookbacks: Vec<i32> = combos
            .iter()
            .map(|params| params.lookback.unwrap_or(DEFAULT_LOOKBACK) as i32)
            .collect();
        let atr_multipliers: Vec<f64> = combos
            .iter()
            .map(|params| params.atr_multiplier.unwrap_or(DEFAULT_ATR_MULTIPLIER))
            .collect();

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaFvgPositioningAverageError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaFvgPositioningAverageError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|extra| value.checked_add(extra))
            })
            .ok_or_else(|| {
                CudaFvgPositioningAverageError::InvalidInput("param bytes overflow".into())
            })?;
        let scratch_left_bytes = rows
            .checked_mul(level_cap)
            .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaFvgPositioningAverageError::InvalidInput("scratch left overflow".into())
            })?;
        let scratch_value_bytes = rows
            .checked_mul(level_cap)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaFvgPositioningAverageError::InvalidInput("scratch value overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaFvgPositioningAverageError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(scratch_left_bytes))
            .and_then(|value| value.checked_add(scratch_value_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaFvgPositioningAverageError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lookbacks = DeviceBuffer::from_slice(&lookbacks)?;
        let d_atr_multipliers = DeviceBuffer::from_slice(&atr_multipliers)?;
        let scratch_elems = rows.checked_mul(level_cap).ok_or_else(|| {
            CudaFvgPositioningAverageError::InvalidInput("scratch elements overflow".into())
        })?;
        let mut d_bull_left = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_elems)? };
        let mut d_bull_value = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let mut d_bear_left = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_elems)? };
        let mut d_bear_value = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let mut d_bull_average = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bear_average = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bull_mid = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bear_mid = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("fvg_positioning_average_batch_f64")
            .map_err(|_| CudaFvgPositioningAverageError::MissingKernelSymbol {
                name: "fvg_positioning_average_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + FVG_POSITIONING_AVERAGE_BLOCK_X - 1) / FVG_POSITIONING_AVERAGE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(FVG_POSITIONING_AVERAGE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_lookbacks.as_device_ptr(),
                d_atr_multipliers.as_device_ptr(),
                lookback_mode,
                rows as i32,
                level_cap as i32,
                d_bull_left.as_device_ptr(),
                d_bull_value.as_device_ptr(),
                d_bear_left.as_device_ptr(),
                d_bear_value.as_device_ptr(),
                d_bull_average.as_device_ptr(),
                d_bear_average.as_device_ptr(),
                d_bull_mid.as_device_ptr(),
                d_bear_mid.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaFvgPositioningAverageBatchResult {
            outputs: FvgPositioningAverageDeviceOutputs {
                bull_average: FvgPositioningAverageDeviceArrayF64 {
                    buf: d_bull_average,
                    rows,
                    cols,
                },
                bear_average: FvgPositioningAverageDeviceArrayF64 {
                    buf: d_bear_average,
                    rows,
                    cols,
                },
                bull_mid: FvgPositioningAverageDeviceArrayF64 {
                    buf: d_bull_mid,
                    rows,
                    cols,
                },
                bear_mid: FvgPositioningAverageDeviceArrayF64 {
                    buf: d_bear_mid,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
