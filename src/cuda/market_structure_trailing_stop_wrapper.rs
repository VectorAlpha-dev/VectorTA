#![cfg(feature = "cuda")]

use crate::indicators::market_structure_trailing_stop::{
    MarketStructureTrailingStopBatchRange, MarketStructureTrailingStopParams,
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

const MARKET_STRUCTURE_TRAILING_STOP_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const RESET_ON_CHOCH: i32 = 0;
const RESET_ON_ALL: i32 = 1;

#[derive(Debug, Error)]
pub enum CudaMarketStructureTrailingStopError {
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

pub struct MarketStructureTrailingStopDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MarketStructureTrailingStopDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct MarketStructureTrailingStopDeviceArrayF64Triple {
    pub trailing_stop: MarketStructureTrailingStopDeviceArrayF64,
    pub state: MarketStructureTrailingStopDeviceArrayF64,
    pub structure: MarketStructureTrailingStopDeviceArrayF64,
}

impl MarketStructureTrailingStopDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.trailing_stop.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.trailing_stop.cols
    }
}

pub struct CudaMarketStructureTrailingStopBatchResult {
    pub outputs: MarketStructureTrailingStopDeviceArrayF64Triple,
    pub combos: Vec<MarketStructureTrailingStopParams>,
}

pub struct CudaMarketStructureTrailingStop {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn parse_reset_on(value: &str) -> Result<i32, CudaMarketStructureTrailingStopError> {
    if value.eq_ignore_ascii_case("CHoCH") {
        Ok(RESET_ON_CHOCH)
    } else if value.eq_ignore_ascii_case("All") {
        Ok(RESET_ON_ALL)
    } else {
        Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
            "invalid reset_on: {value}"
        )))
    }
}

fn expand_axis_usize(
    field: &'static str,
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaMarketStructureTrailingStopError> {
    if start == 0 || end == 0 {
        return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
            "invalid {field} range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
            "invalid {field} range: start={start}, end={end}, step={step}"
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
            return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
                "invalid {field} range: start={start}, end={end}, step={step}"
            )));
        }
        current = next.min(end);
        if current == *out.last().unwrap() {
            break;
        }
    }
    Ok(out)
}

fn expand_axis_f64(
    field: &'static str,
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaMarketStructureTrailingStopError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
            "invalid {field} range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
            "invalid {field} range: start={start}, end={end}, step={step}"
        )));
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end || (end - current).abs() <= 1e-12 {
            break;
        }
        let next = current + step;
        if next <= current {
            return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
                "invalid {field} range: start={start}, end={end}, step={step}"
            )));
        }
        current = if next > end { end } else { next };
    }
    Ok(out)
}

fn expand_grid_market_structure_trailing_stop_checked(
    range: &MarketStructureTrailingStopBatchRange,
    reset_on: &str,
) -> Result<Vec<MarketStructureTrailingStopParams>, CudaMarketStructureTrailingStopError> {
    let lengths = expand_axis_usize("length", range.length.0, range.length.1, range.length.2)?;
    let increment_factors = expand_axis_f64(
        "increment_factor",
        range.increment_factor.0,
        range.increment_factor.1,
        range.increment_factor.2,
    )?;
    let _ = parse_reset_on(reset_on)?;

    let mut combos = Vec::with_capacity(lengths.len().saturating_mul(increment_factors.len()));
    for &length in &lengths {
        for &increment_factor in &increment_factors {
            combos.push(MarketStructureTrailingStopParams {
                length: Some(length),
                increment_factor: Some(increment_factor),
                reset_on: Some(reset_on.to_string()),
            });
        }
    }
    if combos.is_empty() {
        return Err(CudaMarketStructureTrailingStopError::InvalidInput(
            "empty parameter grid".into(),
        ));
    }
    Ok(combos)
}

impl CudaMarketStructureTrailingStop {
    pub fn new(device_id: usize) -> Result<Self, CudaMarketStructureTrailingStopError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("market_structure_trailing_stop_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMarketStructureTrailingStopError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn is_valid_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
        open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
    }

    fn longest_valid_run(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
        let mut best = 0usize;
        let mut run = 0usize;
        for i in 0..close.len() {
            if Self::is_valid_ohlc(open[i], high[i], low[i], close[i]) {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        best
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaMarketStructureTrailingStopError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaMarketStructureTrailingStopError::OutOfMemory {
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
    ) -> Result<(), CudaMarketStructureTrailingStopError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaMarketStructureTrailingStopError::LaunchConfigTooLarge {
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
        sweep: &MarketStructureTrailingStopBatchRange,
        reset_on: &str,
    ) -> Result<CudaMarketStructureTrailingStopBatchResult, CudaMarketStructureTrailingStopError>
    {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaMarketStructureTrailingStopError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let reset_mode = parse_reset_on(reset_on)?;
        let longest = Self::longest_valid_run(open, high, low, close);
        if longest == 0 {
            return Err(CudaMarketStructureTrailingStopError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_market_structure_trailing_stop_checked(sweep, reset_on)?;
        let rows = combos.len();
        let cols = close.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut increment_factors = Vec::with_capacity(rows);

        for combo in &combos {
            let length = combo.length.unwrap_or(14);
            let increment_factor = combo.increment_factor.unwrap_or(100.0);
            if length == 0 {
                return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
                    "invalid length: {length}"
                )));
            }
            if !increment_factor.is_finite() || increment_factor < 0.0 {
                return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
                    "invalid increment_factor: {increment_factor}"
                )));
            }
            let needed = length
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| {
                    CudaMarketStructureTrailingStopError::InvalidInput("length overflow".into())
                })?;
            if longest < needed {
                return Err(CudaMarketStructureTrailingStopError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={longest}"
                )));
            }
            lengths.push(i32::try_from(length).map_err(|_| {
                CudaMarketStructureTrailingStopError::InvalidInput(format!(
                    "length out of range: {length}"
                ))
            })?);
            increment_factors.push(increment_factor);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaMarketStructureTrailingStopError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaMarketStructureTrailingStopError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaMarketStructureTrailingStopError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaMarketStructureTrailingStopError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaMarketStructureTrailingStopError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_increment_factors = DeviceBuffer::from_slice(&increment_factors)?;
        let d_out_trailing_stop = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_state = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_structure = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("market_structure_trailing_stop_batch_f64")
            .map_err(
                |_| CudaMarketStructureTrailingStopError::MissingKernelSymbol {
                    name: "market_structure_trailing_stop_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + MARKET_STRUCTURE_TRAILING_STOP_BLOCK_X - 1)
            / MARKET_STRUCTURE_TRAILING_STOP_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(MARKET_STRUCTURE_TRAILING_STOP_BLOCK_X);
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
                d_increment_factors.as_device_ptr(),
                rows as i32,
                reset_mode,
                d_out_trailing_stop.as_device_ptr(),
                d_out_state.as_device_ptr(),
                d_out_structure.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaMarketStructureTrailingStopBatchResult {
            outputs: MarketStructureTrailingStopDeviceArrayF64Triple {
                trailing_stop: MarketStructureTrailingStopDeviceArrayF64 {
                    buf: d_out_trailing_stop,
                    rows,
                    cols,
                },
                state: MarketStructureTrailingStopDeviceArrayF64 {
                    buf: d_out_state,
                    rows,
                    cols,
                },
                structure: MarketStructureTrailingStopDeviceArrayF64 {
                    buf: d_out_structure,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
