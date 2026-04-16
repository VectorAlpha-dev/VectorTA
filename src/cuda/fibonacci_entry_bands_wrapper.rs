#![cfg(feature = "cuda")]

use crate::indicators::fibonacci_entry_bands::{
    FibonacciEntryBandsBatchRange, FibonacciEntryBandsParams,
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

const FIBONACCI_ENTRY_BANDS_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_SOURCE: &str = "hlc3";
const DEFAULT_LENGTH: usize = 21;
const DEFAULT_ATR_LENGTH: usize = 14;
const DEFAULT_USE_ATR: bool = true;
const DEFAULT_TP_AGGRESSIVENESS: &str = "low";

const SOURCE_OPEN: i32 = 0;
const SOURCE_HIGH: i32 = 1;
const SOURCE_LOW: i32 = 2;
const SOURCE_CLOSE: i32 = 3;
const SOURCE_HL2: i32 = 4;
const SOURCE_HLC3: i32 = 5;
const SOURCE_OHLC4: i32 = 6;
const SOURCE_HLCC4: i32 = 7;

const TP_LOW: i32 = 0;
const TP_MEDIUM: i32 = 1;
const TP_HIGH: i32 = 2;

#[derive(Debug, Error)]
pub enum CudaFibonacciEntryBandsError {
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

pub struct FibonacciEntryBandsDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl FibonacciEntryBandsDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct FibonacciEntryBandsDeviceOutputs {
    pub basis: FibonacciEntryBandsDeviceArrayF64,
    pub trend: FibonacciEntryBandsDeviceArrayF64,
    pub upper_0618: FibonacciEntryBandsDeviceArrayF64,
    pub upper_1000: FibonacciEntryBandsDeviceArrayF64,
    pub upper_1618: FibonacciEntryBandsDeviceArrayF64,
    pub upper_2618: FibonacciEntryBandsDeviceArrayF64,
    pub lower_0618: FibonacciEntryBandsDeviceArrayF64,
    pub lower_1000: FibonacciEntryBandsDeviceArrayF64,
    pub lower_1618: FibonacciEntryBandsDeviceArrayF64,
    pub lower_2618: FibonacciEntryBandsDeviceArrayF64,
    pub tp_long_band: FibonacciEntryBandsDeviceArrayF64,
    pub tp_short_band: FibonacciEntryBandsDeviceArrayF64,
    pub long_entry: FibonacciEntryBandsDeviceArrayF64,
    pub short_entry: FibonacciEntryBandsDeviceArrayF64,
    pub rejection_long: FibonacciEntryBandsDeviceArrayF64,
    pub rejection_short: FibonacciEntryBandsDeviceArrayF64,
    pub long_bounce: FibonacciEntryBandsDeviceArrayF64,
    pub short_bounce: FibonacciEntryBandsDeviceArrayF64,
}

impl FibonacciEntryBandsDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.basis.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.basis.cols
    }
}

pub struct CudaFibonacciEntryBandsBatchResult {
    pub outputs: FibonacciEntryBandsDeviceOutputs,
    pub combos: Vec<FibonacciEntryBandsParams>,
}

pub struct CudaFibonacciEntryBands {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn canonical_source(value: &str) -> Result<(&'static str, i32), CudaFibonacciEntryBandsError> {
    if value.eq_ignore_ascii_case("open") {
        Ok(("open", SOURCE_OPEN))
    } else if value.eq_ignore_ascii_case("high") {
        Ok(("high", SOURCE_HIGH))
    } else if value.eq_ignore_ascii_case("low") {
        Ok(("low", SOURCE_LOW))
    } else if value.eq_ignore_ascii_case("close") {
        Ok(("close", SOURCE_CLOSE))
    } else if value.eq_ignore_ascii_case("hl2") {
        Ok(("hl2", SOURCE_HL2))
    } else if value.eq_ignore_ascii_case("hlc3") {
        Ok(("hlc3", SOURCE_HLC3))
    } else if value.eq_ignore_ascii_case("ohlc4") {
        Ok(("ohlc4", SOURCE_OHLC4))
    } else if value.eq_ignore_ascii_case("hlcc4") || value.eq_ignore_ascii_case("hlcc") {
        Ok(("hlcc4", SOURCE_HLCC4))
    } else {
        Err(CudaFibonacciEntryBandsError::InvalidInput(format!(
            "invalid source: {value}"
        )))
    }
}

#[inline]
fn canonical_tp_mode(value: &str) -> Result<(&'static str, i32), CudaFibonacciEntryBandsError> {
    if value.eq_ignore_ascii_case("low") {
        Ok(("low", TP_LOW))
    } else if value.eq_ignore_ascii_case("medium") {
        Ok(("medium", TP_MEDIUM))
    } else if value.eq_ignore_ascii_case("high") {
        Ok(("high", TP_HIGH))
    } else {
        Err(CudaFibonacciEntryBandsError::InvalidInput(format!(
            "invalid tp_aggressiveness: {value}"
        )))
    }
}

#[inline]
fn source_needs_open(source_name: &str) -> bool {
    source_name.eq_ignore_ascii_case("open") || source_name.eq_ignore_ascii_case("ohlc4")
}

#[inline]
fn valid_bar(source_name: &str, open: f64, high: f64, low: f64, close: f64) -> bool {
    high.is_finite()
        && low.is_finite()
        && close.is_finite()
        && (!source_needs_open(source_name) || open.is_finite())
}

#[inline]
fn expand_axis(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaFibonacciEntryBandsError> {
    if start > end {
        return Err(CudaFibonacciEntryBandsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(CudaFibonacciEntryBandsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    let mut out = Vec::new();
    let mut current = start;
    while current <= end {
        out.push(current);
        current = current.checked_add(step).ok_or_else(|| {
            CudaFibonacciEntryBandsError::InvalidInput("range step overflow".into())
        })?;
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    sweep: &FibonacciEntryBandsBatchRange,
) -> Result<Vec<FibonacciEntryBandsParams>, CudaFibonacciEntryBandsError> {
    let (source_name, _) = canonical_source(&sweep.source)?;
    let (tp_name, _) = canonical_tp_mode(&sweep.tp_aggressiveness)?;
    let lengths = expand_axis(sweep.length)?;
    let atr_lengths = expand_axis(sweep.atr_length)?;
    let mut out = Vec::with_capacity(lengths.len().saturating_mul(atr_lengths.len()));
    for &length in &lengths {
        for &atr_length in &atr_lengths {
            out.push(FibonacciEntryBandsParams {
                source: Some(source_name.to_string()),
                length: Some(length),
                atr_length: Some(atr_length),
                use_atr: Some(sweep.use_atr),
                tp_aggressiveness: Some(tp_name.to_string()),
            });
        }
    }
    Ok(out)
}

#[inline]
fn longest_valid_run(
    source_name: &str,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for (((&o, &h), &l), &c) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
    {
        if valid_bar(source_name, o, h, l, c) {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

impl CudaFibonacciEntryBands {
    pub fn new(device_id: usize) -> Result<Self, CudaFibonacciEntryBandsError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("fibonacci_entry_bands_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaFibonacciEntryBandsError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaFibonacciEntryBandsError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaFibonacciEntryBandsError::OutOfMemory {
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
    ) -> Result<(), CudaFibonacciEntryBandsError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaFibonacciEntryBandsError::LaunchConfigTooLarge {
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
        sweep: &FibonacciEntryBandsBatchRange,
    ) -> Result<CudaFibonacciEntryBandsBatchResult, CudaFibonacciEntryBandsError> {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaFibonacciEntryBandsError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaFibonacciEntryBandsError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let (source_name, source_mode) =
            canonical_source(&sweep.source).or_else(|_| canonical_source(DEFAULT_SOURCE))?;
        let (_, tp_mode) = canonical_tp_mode(&sweep.tp_aggressiveness)
            .or_else(|_| canonical_tp_mode(DEFAULT_TP_AGGRESSIVENESS))?;
        let combos = expand_grid_checked(sweep)?;
        if combos.is_empty() {
            return Err(CudaFibonacciEntryBandsError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let valid = longest_valid_run(source_name, open, high, low, close);
        if valid == 0 {
            return Err(CudaFibonacciEntryBandsError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let use_atr = sweep.use_atr;
        let mut lengths = Vec::with_capacity(rows);
        let mut atr_lengths = Vec::with_capacity(rows);
        let mut max_length = 1usize;

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let atr_length = combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
            if length == 0 || length > cols {
                return Err(CudaFibonacciEntryBandsError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            if atr_length == 0 || atr_length > cols {
                return Err(CudaFibonacciEntryBandsError::InvalidInput(format!(
                    "invalid atr_length: atr_length={atr_length}, data_len={cols}"
                )));
            }
            let needed = if use_atr { atr_length } else { length };
            if valid < needed {
                return Err(CudaFibonacciEntryBandsError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            max_length = max_length.max(length);
            lengths.push(i32::try_from(length).map_err(|_| {
                CudaFibonacciEntryBandsError::InvalidInput("length out of range".into())
            })?);
            atr_lengths.push(i32::try_from(atr_length).map_err(|_| {
                CudaFibonacciEntryBandsError::InvalidInput("atr_length out of range".into())
            })?);
        }

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaFibonacciEntryBandsError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaFibonacciEntryBandsError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() * 2)
            .ok_or_else(|| {
                CudaFibonacciEntryBandsError::InvalidInput("param bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(18))
            .ok_or_else(|| {
                CudaFibonacciEntryBandsError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_bytes = rows
            .checked_mul(max_length)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                CudaFibonacciEntryBandsError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaFibonacciEntryBandsError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_atr_lengths = DeviceBuffer::from_slice(&atr_lengths)?;
        let d_stdev_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_length)? };

        let d_basis = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_trend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_upper_0618 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_upper_1000 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_upper_1618 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_upper_2618 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_lower_0618 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_lower_1000 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_lower_1618 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_lower_2618 = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_tp_long_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_tp_short_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_long_entry = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_short_entry = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_rejection_long = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_rejection_short = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_long_bounce = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_short_bounce = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("fibonacci_entry_bands_batch_f64")
            .map_err(|_| CudaFibonacciEntryBandsError::MissingKernelSymbol {
                name: "fibonacci_entry_bands_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + FIBONACCI_ENTRY_BANDS_BLOCK_X - 1) / FIBONACCI_ENTRY_BANDS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(FIBONACCI_ENTRY_BANDS_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_atr_lengths.as_device_ptr(),
                source_mode,
                if use_atr { 1 } else { 0 },
                tp_mode,
                rows as i32,
                max_length as i32,
                d_stdev_scratch.as_device_ptr(),
                d_basis.as_device_ptr(),
                d_trend.as_device_ptr(),
                d_upper_0618.as_device_ptr(),
                d_upper_1000.as_device_ptr(),
                d_upper_1618.as_device_ptr(),
                d_upper_2618.as_device_ptr(),
                d_lower_0618.as_device_ptr(),
                d_lower_1000.as_device_ptr(),
                d_lower_1618.as_device_ptr(),
                d_lower_2618.as_device_ptr(),
                d_tp_long_band.as_device_ptr(),
                d_tp_short_band.as_device_ptr(),
                d_long_entry.as_device_ptr(),
                d_short_entry.as_device_ptr(),
                d_rejection_long.as_device_ptr(),
                d_rejection_short.as_device_ptr(),
                d_long_bounce.as_device_ptr(),
                d_short_bounce.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaFibonacciEntryBandsBatchResult {
            outputs: FibonacciEntryBandsDeviceOutputs {
                basis: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_basis,
                    rows,
                    cols,
                },
                trend: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_trend,
                    rows,
                    cols,
                },
                upper_0618: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_upper_0618,
                    rows,
                    cols,
                },
                upper_1000: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_upper_1000,
                    rows,
                    cols,
                },
                upper_1618: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_upper_1618,
                    rows,
                    cols,
                },
                upper_2618: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_upper_2618,
                    rows,
                    cols,
                },
                lower_0618: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_lower_0618,
                    rows,
                    cols,
                },
                lower_1000: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_lower_1000,
                    rows,
                    cols,
                },
                lower_1618: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_lower_1618,
                    rows,
                    cols,
                },
                lower_2618: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_lower_2618,
                    rows,
                    cols,
                },
                tp_long_band: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_tp_long_band,
                    rows,
                    cols,
                },
                tp_short_band: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_tp_short_band,
                    rows,
                    cols,
                },
                long_entry: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_long_entry,
                    rows,
                    cols,
                },
                short_entry: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_short_entry,
                    rows,
                    cols,
                },
                rejection_long: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_rejection_long,
                    rows,
                    cols,
                },
                rejection_short: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_rejection_short,
                    rows,
                    cols,
                },
                long_bounce: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_long_bounce,
                    rows,
                    cols,
                },
                short_bounce: FibonacciEntryBandsDeviceArrayF64 {
                    buf: d_short_bounce,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
