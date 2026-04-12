#![cfg(feature = "cuda")]

use crate::indicators::grover_llorens_cycle_oscillator::{
    GroverLlorensCycleOscillatorBatchRange, GroverLlorensCycleOscillatorParams,
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

const GROVER_LLORENS_CYCLE_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 100;
const DEFAULT_MULT: f64 = 10.0;
const DEFAULT_SOURCE: &str = "close";
const DEFAULT_RSI_PERIOD: usize = 20;
const FLOAT_TOL: f64 = 1e-12;

const SOURCE_OPEN: i32 = 0;
const SOURCE_HIGH: i32 = 1;
const SOURCE_LOW: i32 = 2;
const SOURCE_CLOSE: i32 = 3;
const SOURCE_HL2: i32 = 4;
const SOURCE_HLC3: i32 = 5;
const SOURCE_OHLC4: i32 = 6;
const SOURCE_HLCC4: i32 = 7;

#[derive(Debug, Error)]
pub enum CudaGroverLlorensCycleOscillatorError {
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

pub struct GroverLlorensCycleOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl GroverLlorensCycleOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaGroverLlorensCycleOscillatorBatchResult {
    pub outputs: GroverLlorensCycleOscillatorDeviceArrayF64,
    pub combos: Vec<GroverLlorensCycleOscillatorParams>,
}

pub struct CudaGroverLlorensCycleOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[derive(Clone, Copy)]
enum SourceKind {
    Open,
    High,
    Low,
    Close,
    Hl2,
    Hlc3,
    Ohlc4,
    Hlcc4,
}

impl SourceKind {
    #[inline]
    fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("open") {
            Some(Self::Open)
        } else if value.eq_ignore_ascii_case("high") {
            Some(Self::High)
        } else if value.eq_ignore_ascii_case("low") {
            Some(Self::Low)
        } else if value.eq_ignore_ascii_case("close") {
            Some(Self::Close)
        } else if value.eq_ignore_ascii_case("hl2") {
            Some(Self::Hl2)
        } else if value.eq_ignore_ascii_case("hlc3") {
            Some(Self::Hlc3)
        } else if value.eq_ignore_ascii_case("ohlc4") {
            Some(Self::Ohlc4)
        } else if value.eq_ignore_ascii_case("hlcc4") || value.eq_ignore_ascii_case("hlcc") {
            Some(Self::Hlcc4)
        } else {
            None
        }
    }

    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::High => "high",
            Self::Low => "low",
            Self::Close => "close",
            Self::Hl2 => "hl2",
            Self::Hlc3 => "hlc3",
            Self::Ohlc4 => "ohlc4",
            Self::Hlcc4 => "hlcc4",
        }
    }

    #[inline]
    fn needs_open(self) -> bool {
        matches!(self, Self::Open | Self::Ohlc4)
    }

    #[inline]
    fn value(self, open: f64, high: f64, low: f64, close: f64) -> f64 {
        match self {
            Self::Open => open,
            Self::High => high,
            Self::Low => low,
            Self::Close => close,
            Self::Hl2 => 0.5 * (high + low),
            Self::Hlc3 => (high + low + close) / 3.0,
            Self::Ohlc4 => (open + high + low + close) * 0.25,
            Self::Hlcc4 => (high + low + close + close) * 0.25,
        }
    }

    #[inline]
    fn id(self) -> i32 {
        match self {
            Self::Open => SOURCE_OPEN,
            Self::High => SOURCE_HIGH,
            Self::Low => SOURCE_LOW,
            Self::Close => SOURCE_CLOSE,
            Self::Hl2 => SOURCE_HL2,
            Self::Hlc3 => SOURCE_HLC3,
            Self::Ohlc4 => SOURCE_OHLC4,
            Self::Hlcc4 => SOURCE_HLCC4,
        }
    }
}

fn valid_bar(source: SourceKind, open: f64, high: f64, low: f64, close: f64) -> bool {
    if !(high.is_finite() && low.is_finite() && close.is_finite()) {
        return false;
    }
    if source.needs_open() && !open.is_finite() {
        return false;
    }
    source.value(open, high, low, close).is_finite()
}

fn count_valid_bars(
    source: SourceKind,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> usize {
    let mut count = 0usize;
    for i in 0..close.len() {
        if valid_bar(source, open[i], high[i], low[i], close[i]) {
            count += 1;
        }
    }
    count
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaGroverLlorensCycleOscillatorError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            match value.checked_add(step) {
                Some(next) if next > value => value = next,
                _ => break,
            }
        }
    } else {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_sub(step);
            if next >= value || next < end {
                break;
            }
            value = next;
        }
    }

    if out.is_empty() {
        return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }
    Ok(out)
}

fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaGroverLlorensCycleOscillatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }
    if step.abs() <= FLOAT_TOL || (start - end).abs() <= FLOAT_TOL {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end + FLOAT_TOL {
            out.push(value);
            value += step.abs();
            if out.len() > 1_000_000 {
                break;
            }
        }
    } else {
        let mut value = start;
        while value >= end - FLOAT_TOL {
            out.push(value);
            value -= step.abs();
            if out.len() > 1_000_000 {
                break;
            }
        }
    }

    if out.is_empty() {
        return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
            format!("invalid range: start={start}, end={end}, step={step}"),
        ));
    }
    Ok(out)
}

fn expand_grid(
    sweep: &GroverLlorensCycleOscillatorBatchRange,
) -> Result<Vec<GroverLlorensCycleOscillatorParams>, CudaGroverLlorensCycleOscillatorError> {
    let lengths = expand_axis_usize(sweep.length.0, sweep.length.1, sweep.length.2)?;
    let mults = expand_axis_f64(sweep.mult.0, sweep.mult.1, sweep.mult.2)?;
    let rsi_periods =
        expand_axis_usize(sweep.rsi_period.0, sweep.rsi_period.1, sweep.rsi_period.2)?;
    let source = SourceKind::parse(&sweep.source).ok_or_else(|| {
        CudaGroverLlorensCycleOscillatorError::InvalidInput(format!(
            "invalid source: {}",
            sweep.source
        ))
    })?;

    let mut combos = Vec::with_capacity(lengths.len() * mults.len() * rsi_periods.len());
    for &length in &lengths {
        for &mult in &mults {
            for &rsi_period in &rsi_periods {
                combos.push(GroverLlorensCycleOscillatorParams {
                    length: Some(length),
                    mult: Some(mult),
                    source: Some(source.as_str().to_string()),
                    smooth: Some(sweep.smooth),
                    rsi_period: Some(rsi_period),
                });
            }
        }
    }
    Ok(combos)
}

impl CudaGroverLlorensCycleOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaGroverLlorensCycleOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("grover_llorens_cycle_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaGroverLlorensCycleOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaGroverLlorensCycleOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaGroverLlorensCycleOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaGroverLlorensCycleOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(
                CudaGroverLlorensCycleOscillatorError::LaunchConfigTooLarge {
                    gx: grid.x,
                    gy: grid.y,
                    gz: grid.z,
                    bx: block.x,
                    by: block.y,
                    bz: block.z,
                },
            );
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &GroverLlorensCycleOscillatorBatchRange,
    ) -> Result<CudaGroverLlorensCycleOscillatorBatchResult, CudaGroverLlorensCycleOscillatorError>
    {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
                format!(
                    "input length mismatch: open={}, high={}, low={}, close={}",
                    open.len(),
                    high.len(),
                    low.len(),
                    close.len()
                ),
            ));
        }

        let source = SourceKind::parse(&sweep.source).ok_or_else(|| {
            CudaGroverLlorensCycleOscillatorError::InvalidInput(format!(
                "invalid source: {}",
                sweep.source
            ))
        })?;
        let valid_total = count_valid_bars(source, open, high, low, close);
        if valid_total == 0 {
            return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut mults = Vec::with_capacity(rows);
        let mut rsi_periods = Vec::with_capacity(rows);
        let mut max_needed = 0usize;

        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let mult = combo.mult.unwrap_or(DEFAULT_MULT);
            let rsi_period = combo.rsi_period.unwrap_or(DEFAULT_RSI_PERIOD);

            if length == 0 || length > cols {
                return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
                    format!("invalid length: length={length}, data_len={cols}"),
                ));
            }
            if !mult.is_finite() {
                return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
                    format!("invalid mult: {mult}"),
                ));
            }
            if rsi_period == 0 {
                return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
                    format!("invalid rsi_period: {rsi_period}"),
                ));
            }

            max_needed = max_needed.max(length.max(rsi_period));
            lengths.push(i32::try_from(length).map_err(|_| {
                CudaGroverLlorensCycleOscillatorError::InvalidInput(format!(
                    "length out of range: {length}"
                ))
            })?);
            mults.push(mult);
            rsi_periods.push(i32::try_from(rsi_period).map_err(|_| {
                CudaGroverLlorensCycleOscillatorError::InvalidInput(format!(
                    "rsi_period out of range: {rsi_period}"
                ))
            })?);
        }

        if valid_total < max_needed {
            return Err(CudaGroverLlorensCycleOscillatorError::InvalidInput(
                format!("not enough valid data: needed={max_needed}, valid={valid_total}"),
            ));
        }

        let rows_i32 = i32::try_from(rows).map_err(|_| {
            CudaGroverLlorensCycleOscillatorError::InvalidInput("rows out of range".into())
        })?;
        let cols_i32 = i32::try_from(cols).map_err(|_| {
            CudaGroverLlorensCycleOscillatorError::InvalidInput("cols out of range".into())
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaGroverLlorensCycleOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaGroverLlorensCycleOscillatorError::InvalidInput("param bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaGroverLlorensCycleOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaGroverLlorensCycleOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaGroverLlorensCycleOscillatorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_mults = DeviceBuffer::from_slice(&mults)?;
        let d_rsi_periods = DeviceBuffer::from_slice(&rsi_periods)?;
        let d_out_values = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("grover_llorens_cycle_oscillator_batch_f64")
            .map_err(
                |_| CudaGroverLlorensCycleOscillatorError::MissingKernelSymbol {
                    name: "grover_llorens_cycle_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + GROVER_LLORENS_CYCLE_OSCILLATOR_BLOCK_X - 1)
            / GROVER_LLORENS_CYCLE_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(GROVER_LLORENS_CYCLE_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols_i32,
                d_lengths.as_device_ptr(),
                d_mults.as_device_ptr(),
                source.id(),
                if sweep.smooth { 1i32 } else { 0i32 },
                d_rsi_periods.as_device_ptr(),
                rows_i32,
                d_out_values.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaGroverLlorensCycleOscillatorBatchResult {
            outputs: GroverLlorensCycleOscillatorDeviceArrayF64 {
                buf: d_out_values,
                rows,
                cols,
            },
            combos,
        })
    }
}
