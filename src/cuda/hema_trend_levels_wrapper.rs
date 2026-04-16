#![cfg(feature = "cuda")]

use crate::indicators::hema_trend_levels::{HemaTrendLevelsBatchRange, HemaTrendLevelsParams};
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

const HEMA_TREND_LEVELS_BLOCK_X: u32 = 64;
const DEFAULT_FAST_LENGTH: usize = 20;
const DEFAULT_SLOW_LENGTH: usize = 40;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaHemaTrendLevelsError {
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

pub struct HemaTrendLevelsDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl HemaTrendLevelsDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct HemaTrendLevelsDeviceOutputs {
    pub fast_hema: HemaTrendLevelsDeviceArrayF64,
    pub slow_hema: HemaTrendLevelsDeviceArrayF64,
    pub trend_direction: HemaTrendLevelsDeviceArrayF64,
    pub bar_state: HemaTrendLevelsDeviceArrayF64,
    pub bullish_crossover: HemaTrendLevelsDeviceArrayF64,
    pub bearish_crossunder: HemaTrendLevelsDeviceArrayF64,
    pub box_offset: HemaTrendLevelsDeviceArrayF64,
    pub bull_box_top: HemaTrendLevelsDeviceArrayF64,
    pub bull_box_bottom: HemaTrendLevelsDeviceArrayF64,
    pub bear_box_top: HemaTrendLevelsDeviceArrayF64,
    pub bear_box_bottom: HemaTrendLevelsDeviceArrayF64,
    pub bullish_test: HemaTrendLevelsDeviceArrayF64,
    pub bearish_test: HemaTrendLevelsDeviceArrayF64,
    pub bullish_test_level: HemaTrendLevelsDeviceArrayF64,
    pub bearish_test_level: HemaTrendLevelsDeviceArrayF64,
}

impl HemaTrendLevelsDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.fast_hema.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.fast_hema.cols
    }
}

pub struct CudaHemaTrendLevelsBatchResult {
    pub outputs: HemaTrendLevelsDeviceOutputs,
    pub combos: Vec<HemaTrendLevelsParams>,
}

pub struct CudaHemaTrendLevels {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
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
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaHemaTrendLevelsError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            let next = value.saturating_add(step);
            if next == value {
                break;
            }
            value = next;
        }
    } else {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_sub(step);
            if next == value || next < end {
                break;
            }
            value = next;
        }
    }
    if out.is_empty() {
        return Err(CudaHemaTrendLevelsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    sweep: &HemaTrendLevelsBatchRange,
) -> Result<Vec<HemaTrendLevelsParams>, CudaHemaTrendLevelsError> {
    let fast_lengths = expand_axis_usize(sweep.fast_length)?;
    let slow_lengths = expand_axis_usize(sweep.slow_length)?;
    let mut combos = Vec::with_capacity(fast_lengths.len().saturating_mul(slow_lengths.len()));
    for fast_length in fast_lengths {
        if fast_length == 0 {
            return Err(CudaHemaTrendLevelsError::InvalidInput(format!(
                "invalid fast_length: {fast_length}"
            )));
        }
        for &slow_length in &slow_lengths {
            if slow_length == 0 {
                return Err(CudaHemaTrendLevelsError::InvalidInput(format!(
                    "invalid slow_length: {slow_length}"
                )));
            }
            combos.push(HemaTrendLevelsParams {
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
            });
        }
    }
    Ok(combos)
}

impl CudaHemaTrendLevels {
    pub fn new(device_id: usize) -> Result<Self, CudaHemaTrendLevelsError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("hema_trend_levels_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaHemaTrendLevelsError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaHemaTrendLevelsError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaHemaTrendLevelsError::OutOfMemory {
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
    ) -> Result<(), CudaHemaTrendLevelsError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaHemaTrendLevelsError::LaunchConfigTooLarge {
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
        sweep: &HemaTrendLevelsBatchRange,
    ) -> Result<CudaHemaTrendLevelsBatchResult, CudaHemaTrendLevelsError> {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaHemaTrendLevelsError::InvalidInput("empty input".into()));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaHemaTrendLevelsError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }
        if first_valid_ohlc(open, high, low, close) >= close.len() {
            return Err(CudaHemaTrendLevelsError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_checked(sweep)?;
        let rows = combos.len();
        let cols = close.len();
        let fast_lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH) as i32)
            .collect();
        let slow_lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH) as i32)
            .collect();

        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaHemaTrendLevelsError::InvalidInput("rows*cols overflow".into()))?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| CudaHemaTrendLevelsError::InvalidInput("input bytes overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| CudaHemaTrendLevelsError::InvalidInput("param bytes overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(15))
            .ok_or_else(|| {
                CudaHemaTrendLevelsError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaHemaTrendLevelsError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_fast_lengths = DeviceBuffer::from_slice(&fast_lengths)?;
        let d_slow_lengths = DeviceBuffer::from_slice(&slow_lengths)?;
        let mut d_fast_hema = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_slow_hema = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_trend_direction = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bar_state = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bullish_crossover = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bearish_crossunder = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_box_offset = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bull_box_top = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bull_box_bottom = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bear_box_top = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bear_box_bottom = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bullish_test = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bearish_test = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bullish_test_level = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_bearish_test_level = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("hema_trend_levels_batch_f64")
            .map_err(|_| CudaHemaTrendLevelsError::MissingKernelSymbol {
                name: "hema_trend_levels_batch_f64",
            })?;
        let grid_x = ((rows as u32) + HEMA_TREND_LEVELS_BLOCK_X - 1) / HEMA_TREND_LEVELS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(HEMA_TREND_LEVELS_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_fast_lengths.as_device_ptr(),
                d_slow_lengths.as_device_ptr(),
                rows as i32,
                d_fast_hema.as_device_ptr(),
                d_slow_hema.as_device_ptr(),
                d_trend_direction.as_device_ptr(),
                d_bar_state.as_device_ptr(),
                d_bullish_crossover.as_device_ptr(),
                d_bearish_crossunder.as_device_ptr(),
                d_box_offset.as_device_ptr(),
                d_bull_box_top.as_device_ptr(),
                d_bull_box_bottom.as_device_ptr(),
                d_bear_box_top.as_device_ptr(),
                d_bear_box_bottom.as_device_ptr(),
                d_bullish_test.as_device_ptr(),
                d_bearish_test.as_device_ptr(),
                d_bullish_test_level.as_device_ptr(),
                d_bearish_test_level.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaHemaTrendLevelsBatchResult {
            outputs: HemaTrendLevelsDeviceOutputs {
                fast_hema: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_fast_hema,
                    rows,
                    cols,
                },
                slow_hema: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_slow_hema,
                    rows,
                    cols,
                },
                trend_direction: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_trend_direction,
                    rows,
                    cols,
                },
                bar_state: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bar_state,
                    rows,
                    cols,
                },
                bullish_crossover: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bullish_crossover,
                    rows,
                    cols,
                },
                bearish_crossunder: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bearish_crossunder,
                    rows,
                    cols,
                },
                box_offset: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_box_offset,
                    rows,
                    cols,
                },
                bull_box_top: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bull_box_top,
                    rows,
                    cols,
                },
                bull_box_bottom: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bull_box_bottom,
                    rows,
                    cols,
                },
                bear_box_top: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bear_box_top,
                    rows,
                    cols,
                },
                bear_box_bottom: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bear_box_bottom,
                    rows,
                    cols,
                },
                bullish_test: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bullish_test,
                    rows,
                    cols,
                },
                bearish_test: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bearish_test,
                    rows,
                    cols,
                },
                bullish_test_level: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bullish_test_level,
                    rows,
                    cols,
                },
                bearish_test_level: HemaTrendLevelsDeviceArrayF64 {
                    buf: d_bearish_test_level,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
