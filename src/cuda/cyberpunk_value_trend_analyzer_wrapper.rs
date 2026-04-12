#![cfg(feature = "cuda")]

use crate::indicators::cyberpunk_value_trend_analyzer::{
    CyberpunkValueTrendAnalyzerBatchRange, CyberpunkValueTrendAnalyzerParams,
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

const CYBERPUNK_VALUE_TREND_ANALYZER_BLOCK_X: u32 = 64;
const DEFAULT_ENTRY_LEVEL: usize = 30;
const DEFAULT_EXIT_LEVEL: usize = 75;
const RANGE75_WINDOW: usize = 75;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaCyberpunkValueTrendAnalyzerError {
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

pub struct CyberpunkValueTrendAnalyzerDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl CyberpunkValueTrendAnalyzerDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CyberpunkValueTrendAnalyzerDeviceOutputs {
    pub value_trend: CyberpunkValueTrendAnalyzerDeviceArrayF64,
    pub value_trend_lag: CyberpunkValueTrendAnalyzerDeviceArrayF64,
    pub deviation_index: CyberpunkValueTrendAnalyzerDeviceArrayF64,
    pub overbought_signal: CyberpunkValueTrendAnalyzerDeviceArrayF64,
    pub buy_signal: CyberpunkValueTrendAnalyzerDeviceArrayF64,
    pub sell_signal: CyberpunkValueTrendAnalyzerDeviceArrayF64,
}

impl CyberpunkValueTrendAnalyzerDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.value_trend.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.value_trend.cols
    }
}

pub struct CudaCyberpunkValueTrendAnalyzerBatchResult {
    pub outputs: CyberpunkValueTrendAnalyzerDeviceOutputs,
    pub combos: Vec<CyberpunkValueTrendAnalyzerParams>,
}

pub struct CudaCyberpunkValueTrendAnalyzer {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaCyberpunkValueTrendAnalyzerError> {
    if !(1..=100).contains(&start) || !(1..=100).contains(&end) {
        return Err(CudaCyberpunkValueTrendAnalyzerError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(CudaCyberpunkValueTrendAnalyzerError::InvalidInput(format!(
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
            CudaCyberpunkValueTrendAnalyzerError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            ))
        })?;
        if next <= current {
            return Err(CudaCyberpunkValueTrendAnalyzerError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        current = next.min(end);
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    range: &CyberpunkValueTrendAnalyzerBatchRange,
) -> Result<Vec<CyberpunkValueTrendAnalyzerParams>, CudaCyberpunkValueTrendAnalyzerError> {
    let entry_levels = axis_usize(
        range.entry_level.0,
        range.entry_level.1,
        range.entry_level.2,
    )?;
    let exit_levels = axis_usize(range.exit_level.0, range.exit_level.1, range.exit_level.2)?;
    let cap = entry_levels
        .len()
        .checked_mul(exit_levels.len())
        .ok_or_else(|| {
            CudaCyberpunkValueTrendAnalyzerError::InvalidInput("parameter grid overflow".into())
        })?;
    let mut out = Vec::with_capacity(cap);
    for &entry_level in &entry_levels {
        for &exit_level in &exit_levels {
            out.push(CyberpunkValueTrendAnalyzerParams {
                entry_level: Some(entry_level),
                exit_level: Some(exit_level),
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

impl CudaCyberpunkValueTrendAnalyzer {
    pub fn new(device_id: usize) -> Result<Self, CudaCyberpunkValueTrendAnalyzerError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("cyberpunk_value_trend_analyzer_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaCyberpunkValueTrendAnalyzerError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaCyberpunkValueTrendAnalyzerError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaCyberpunkValueTrendAnalyzerError::OutOfMemory {
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
    ) -> Result<(), CudaCyberpunkValueTrendAnalyzerError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaCyberpunkValueTrendAnalyzerError::LaunchConfigTooLarge {
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
        sweep: &CyberpunkValueTrendAnalyzerBatchRange,
    ) -> Result<CudaCyberpunkValueTrendAnalyzerBatchResult, CudaCyberpunkValueTrendAnalyzerError>
    {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaCyberpunkValueTrendAnalyzerError::InvalidInput(
                "empty input".into(),
            ));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaCyberpunkValueTrendAnalyzerError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = expand_grid_checked(sweep)?;
        let longest = longest_valid_run(open, high, low, close);
        if longest == 0 {
            return Err(CudaCyberpunkValueTrendAnalyzerError::InvalidInput(
                "all values are NaN".into(),
            ));
        }
        if longest < RANGE75_WINDOW {
            return Err(CudaCyberpunkValueTrendAnalyzerError::InvalidInput(format!(
                "not enough valid data: needed={RANGE75_WINDOW}, valid={longest}"
            )));
        }

        let rows = combos.len();
        let cols = close.len();
        let entry_levels: Vec<i32> = combos
            .iter()
            .map(|params| params.entry_level.unwrap_or(DEFAULT_ENTRY_LEVEL) as i32)
            .collect();
        let exit_levels: Vec<i32> = combos
            .iter()
            .map(|params| params.exit_level.unwrap_or(DEFAULT_EXIT_LEVEL) as i32)
            .collect();

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaCyberpunkValueTrendAnalyzerError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaCyberpunkValueTrendAnalyzerError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaCyberpunkValueTrendAnalyzerError::InvalidInput("param bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(6))
            .ok_or_else(|| {
                CudaCyberpunkValueTrendAnalyzerError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaCyberpunkValueTrendAnalyzerError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_entry_levels = DeviceBuffer::from_slice(&entry_levels)?;
        let d_exit_levels = DeviceBuffer::from_slice(&exit_levels)?;
        let mut d_value_trend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_value_trend_lag = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_deviation_index = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_overbought_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_buy_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_sell_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("cyberpunk_value_trend_analyzer_batch_f64")
            .map_err(
                |_| CudaCyberpunkValueTrendAnalyzerError::MissingKernelSymbol {
                    name: "cyberpunk_value_trend_analyzer_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + CYBERPUNK_VALUE_TREND_ANALYZER_BLOCK_X - 1)
            / CYBERPUNK_VALUE_TREND_ANALYZER_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(CYBERPUNK_VALUE_TREND_ANALYZER_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_entry_levels.as_device_ptr(),
                d_exit_levels.as_device_ptr(),
                rows as i32,
                d_value_trend.as_device_ptr(),
                d_value_trend_lag.as_device_ptr(),
                d_deviation_index.as_device_ptr(),
                d_overbought_signal.as_device_ptr(),
                d_buy_signal.as_device_ptr(),
                d_sell_signal.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaCyberpunkValueTrendAnalyzerBatchResult {
            outputs: CyberpunkValueTrendAnalyzerDeviceOutputs {
                value_trend: CyberpunkValueTrendAnalyzerDeviceArrayF64 {
                    buf: d_value_trend,
                    rows,
                    cols,
                },
                value_trend_lag: CyberpunkValueTrendAnalyzerDeviceArrayF64 {
                    buf: d_value_trend_lag,
                    rows,
                    cols,
                },
                deviation_index: CyberpunkValueTrendAnalyzerDeviceArrayF64 {
                    buf: d_deviation_index,
                    rows,
                    cols,
                },
                overbought_signal: CyberpunkValueTrendAnalyzerDeviceArrayF64 {
                    buf: d_overbought_signal,
                    rows,
                    cols,
                },
                buy_signal: CyberpunkValueTrendAnalyzerDeviceArrayF64 {
                    buf: d_buy_signal,
                    rows,
                    cols,
                },
                sell_signal: CyberpunkValueTrendAnalyzerDeviceArrayF64 {
                    buf: d_sell_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
