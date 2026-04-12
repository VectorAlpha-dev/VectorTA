#![cfg(feature = "cuda")]

use crate::indicators::candle_strength_oscillator::{
    CandleStrengthOscillatorBatchRange, CandleStrengthOscillatorParams,
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

const CANDLE_STRENGTH_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_PERIOD: usize = 50;
const DEFAULT_ATR_LENGTH: usize = 50;
const DEFAULT_MODE: &str = "bollinger";
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaCandleStrengthOscillatorError {
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

pub struct CandleStrengthOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl CandleStrengthOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CandleStrengthOscillatorDeviceOutputs {
    pub strength: CandleStrengthOscillatorDeviceArrayF64,
    pub highs: CandleStrengthOscillatorDeviceArrayF64,
    pub lows: CandleStrengthOscillatorDeviceArrayF64,
    pub mid: CandleStrengthOscillatorDeviceArrayF64,
    pub long_signal: CandleStrengthOscillatorDeviceArrayF64,
    pub short_signal: CandleStrengthOscillatorDeviceArrayF64,
}

impl CandleStrengthOscillatorDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.strength.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.strength.cols
    }
}

pub struct CudaCandleStrengthOscillatorBatchResult {
    pub outputs: CandleStrengthOscillatorDeviceOutputs,
    pub combos: Vec<CandleStrengthOscillatorParams>,
}

pub struct CudaCandleStrengthOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[derive(Clone, Copy)]
struct ResolvedParams {
    period: usize,
    atr_enabled: bool,
    atr_length: usize,
}

#[inline]
fn canonical_mode(value: &str) -> Result<&'static str, CudaCandleStrengthOscillatorError> {
    if value.eq_ignore_ascii_case("bollinger") || value.eq_ignore_ascii_case("bb") {
        return Ok("bollinger");
    }
    if value.eq_ignore_ascii_case("donchian") || value.eq_ignore_ascii_case("dc") {
        return Ok("donchian");
    }
    Err(CudaCandleStrengthOscillatorError::InvalidInput(format!(
        "invalid mode: {value}"
    )))
}

#[inline]
fn mode_id(value: &str) -> Result<i32, CudaCandleStrengthOscillatorError> {
    match canonical_mode(value)? {
        "bollinger" => Ok(0),
        "donchian" => Ok(1),
        _ => unreachable!(),
    }
}

#[inline]
fn expand_axis(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaCandleStrengthOscillatorError> {
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(CudaCandleStrengthOscillatorError::InvalidInput(format!(
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
        let next = current.checked_add(step).ok_or_else(|| {
            CudaCandleStrengthOscillatorError::InvalidInput("range step overflow".into())
        })?;
        if next <= current {
            return Err(CudaCandleStrengthOscillatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        current = next.min(end);
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    sweep: &CandleStrengthOscillatorBatchRange,
    fixed: &CandleStrengthOscillatorParams,
) -> Result<Vec<CandleStrengthOscillatorParams>, CudaCandleStrengthOscillatorError> {
    let periods = expand_axis(sweep.period)?;
    let atr_lengths = expand_axis(sweep.atr_length)?;
    let mut out = Vec::with_capacity(periods.len().saturating_mul(atr_lengths.len()));
    for &period in &periods {
        if period == 0 {
            return Err(CudaCandleStrengthOscillatorError::InvalidInput(format!(
                "invalid period: {period}"
            )));
        }
        for &atr_length in &atr_lengths {
            if atr_length == 0 {
                return Err(CudaCandleStrengthOscillatorError::InvalidInput(format!(
                    "invalid atr_length: {atr_length}"
                )));
            }
            out.push(CandleStrengthOscillatorParams {
                period: Some(period),
                atr_enabled: Some(fixed.atr_enabled.unwrap_or(false)),
                atr_length: Some(atr_length),
                mode: Some(
                    fixed
                        .mode
                        .clone()
                        .unwrap_or_else(|| DEFAULT_MODE.to_string()),
                ),
            });
        }
    }
    Ok(out)
}

#[inline]
fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let len = close.len();
    let mut i = 0usize;
    while i < len {
        if open[i].is_finite() && high[i].is_finite() && low[i].is_finite() && close[i].is_finite()
        {
            return i;
        }
        i += 1;
    }
    len
}

#[inline]
fn longest_valid_run(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for (((&o, &h), &l), &c) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
    {
        if o.is_finite() && h.is_finite() && l.is_finite() && c.is_finite() {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

#[inline]
fn sqrt_period(period: usize) -> usize {
    (period as f64).sqrt().floor() as usize
}

#[inline]
fn hma_warmup_prefix(period: usize) -> usize {
    period.saturating_add(sqrt_period(period)).saturating_sub(2)
}

#[inline]
fn atr_warmup_prefix(atr_enabled: bool, atr_length: usize) -> usize {
    if atr_enabled {
        atr_length.saturating_sub(1)
    } else {
        0
    }
}

#[inline]
fn levels_needed_bars(period: usize, atr_enabled: bool, atr_length: usize) -> usize {
    atr_warmup_prefix(atr_enabled, atr_length)
        .saturating_add(hma_warmup_prefix(period))
        .saturating_add(period.saturating_sub(1))
        .saturating_add(1)
}

#[inline]
fn resolve_params(
    params: &CandleStrengthOscillatorParams,
) -> Result<ResolvedParams, CudaCandleStrengthOscillatorError> {
    let period = params.period.unwrap_or(DEFAULT_PERIOD);
    if period == 0 {
        return Err(CudaCandleStrengthOscillatorError::InvalidInput(format!(
            "invalid period: {period}"
        )));
    }
    let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
    if atr_length == 0 {
        return Err(CudaCandleStrengthOscillatorError::InvalidInput(format!(
            "invalid atr_length: {atr_length}"
        )));
    }
    let atr_enabled = params.atr_enabled.unwrap_or(false);
    let _ = canonical_mode(params.mode.as_deref().unwrap_or(DEFAULT_MODE))?;
    Ok(ResolvedParams {
        period,
        atr_enabled,
        atr_length,
    })
}

impl CudaCandleStrengthOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaCandleStrengthOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("candle_strength_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaCandleStrengthOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaCandleStrengthOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaCandleStrengthOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaCandleStrengthOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaCandleStrengthOscillatorError::LaunchConfigTooLarge {
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
        sweep: &CandleStrengthOscillatorBatchRange,
        fixed: &CandleStrengthOscillatorParams,
    ) -> Result<CudaCandleStrengthOscillatorBatchResult, CudaCandleStrengthOscillatorError> {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaCandleStrengthOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaCandleStrengthOscillatorError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }
        if first_valid_ohlc(open, high, low, close) >= close.len() {
            return Err(CudaCandleStrengthOscillatorError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_checked(sweep, fixed)?;
        let rows = combos.len();
        let cols = close.len();
        let mode = fixed.mode.as_deref().unwrap_or(DEFAULT_MODE);
        let mode = mode_id(mode)?;

        let mut max_needed = 0usize;
        let mut max_period = 1usize;
        let mut max_half = 1usize;
        let mut max_sqrt = 1usize;
        let mut max_level = 1usize;
        let mut periods = Vec::with_capacity(rows);
        let mut atr_lengths = Vec::with_capacity(rows);

        for combo in &combos {
            let resolved = resolve_params(combo)?;
            periods.push(resolved.period as i32);
            atr_lengths.push(resolved.atr_length as i32);
            max_needed = max_needed.max(levels_needed_bars(
                resolved.period,
                resolved.atr_enabled,
                resolved.atr_length,
            ));
            max_period = max_period.max(resolved.period);
            max_half = max_half.max((resolved.period / 2).max(1));
            max_sqrt = max_sqrt.max(sqrt_period(resolved.period).max(1));
            max_level = max_level.max(resolved.period.max(1));
        }

        let valid = longest_valid_run(open, high, low, close);
        if valid < max_needed {
            return Err(CudaCandleStrengthOscillatorError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={valid}"
            )));
        }

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaCandleStrengthOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let full_elems = rows.checked_mul(max_period).ok_or_else(|| {
            CudaCandleStrengthOscillatorError::InvalidInput("full scratch overflow".into())
        })?;
        let half_elems = rows.checked_mul(max_half).ok_or_else(|| {
            CudaCandleStrengthOscillatorError::InvalidInput("half scratch overflow".into())
        })?;
        let sqrt_elems = rows.checked_mul(max_sqrt).ok_or_else(|| {
            CudaCandleStrengthOscillatorError::InvalidInput("sqrt scratch overflow".into())
        })?;
        let level_elems = rows.checked_mul(max_level).ok_or_else(|| {
            CudaCandleStrengthOscillatorError::InvalidInput("level scratch overflow".into())
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| {
                CudaCandleStrengthOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaCandleStrengthOscillatorError::InvalidInput("param bytes overflow".into())
            })?;
        let scratch_bytes = full_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| {
                half_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|extra| v.checked_add(extra))
            })
            .and_then(|v| {
                sqrt_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|extra| v.checked_add(extra))
            })
            .and_then(|v| {
                level_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|extra| v.checked_add(extra))
            })
            .ok_or_else(|| {
                CudaCandleStrengthOscillatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(6))
            .ok_or_else(|| {
                CudaCandleStrengthOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|v| v.checked_add(scratch_bytes))
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaCandleStrengthOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_atr_lengths = DeviceBuffer::from_slice(&atr_lengths)?;
        let mut d_full = unsafe { DeviceBuffer::<f64>::uninitialized(full_elems)? };
        let mut d_half = unsafe { DeviceBuffer::<f64>::uninitialized(half_elems)? };
        let mut d_sqrt = unsafe { DeviceBuffer::<f64>::uninitialized(sqrt_elems)? };
        let mut d_level = unsafe { DeviceBuffer::<f64>::uninitialized(level_elems)? };
        let mut d_strength = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_highs = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_lows = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_mid = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_long_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_short_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("candle_strength_oscillator_batch_f64")
            .map_err(|_| CudaCandleStrengthOscillatorError::MissingKernelSymbol {
                name: "candle_strength_oscillator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + CANDLE_STRENGTH_OSCILLATOR_BLOCK_X - 1)
            / CANDLE_STRENGTH_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(CANDLE_STRENGTH_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                d_atr_lengths.as_device_ptr(),
                i32::from(fixed.atr_enabled.unwrap_or(false)),
                mode,
                rows as i32,
                max_period as i32,
                max_half as i32,
                max_sqrt as i32,
                max_level as i32,
                d_full.as_device_ptr(),
                d_half.as_device_ptr(),
                d_sqrt.as_device_ptr(),
                d_level.as_device_ptr(),
                d_strength.as_device_ptr(),
                d_highs.as_device_ptr(),
                d_lows.as_device_ptr(),
                d_mid.as_device_ptr(),
                d_long_signal.as_device_ptr(),
                d_short_signal.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaCandleStrengthOscillatorBatchResult {
            outputs: CandleStrengthOscillatorDeviceOutputs {
                strength: CandleStrengthOscillatorDeviceArrayF64 {
                    buf: d_strength,
                    rows,
                    cols,
                },
                highs: CandleStrengthOscillatorDeviceArrayF64 {
                    buf: d_highs,
                    rows,
                    cols,
                },
                lows: CandleStrengthOscillatorDeviceArrayF64 {
                    buf: d_lows,
                    rows,
                    cols,
                },
                mid: CandleStrengthOscillatorDeviceArrayF64 {
                    buf: d_mid,
                    rows,
                    cols,
                },
                long_signal: CandleStrengthOscillatorDeviceArrayF64 {
                    buf: d_long_signal,
                    rows,
                    cols,
                },
                short_signal: CandleStrengthOscillatorDeviceArrayF64 {
                    buf: d_short_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
