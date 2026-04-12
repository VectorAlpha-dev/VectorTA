#![cfg(feature = "cuda")]

use crate::indicators::fibonacci_trailing_stop::{
    FibonacciTrailingStopBatchRange, FibonacciTrailingStopParams,
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

const FIBONACCI_TRAILING_STOP_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LEFT_BARS: usize = 20;
const DEFAULT_RIGHT_BARS: usize = 1;
const DEFAULT_LEVEL: f64 = -0.382;
const DEFAULT_TRIGGER: &str = "close";
const FLOAT_TOL: f64 = 1e-12;

const TRIGGER_CLOSE: i32 = 0;
const TRIGGER_WICK: i32 = 1;

#[derive(Debug, Error)]
pub enum CudaFibonacciTrailingStopError {
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

pub struct FibonacciTrailingStopDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl FibonacciTrailingStopDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct FibonacciTrailingStopDeviceArrayF64Quad {
    pub trailing_stop: FibonacciTrailingStopDeviceArrayF64,
    pub long_stop: FibonacciTrailingStopDeviceArrayF64,
    pub short_stop: FibonacciTrailingStopDeviceArrayF64,
    pub direction: FibonacciTrailingStopDeviceArrayF64,
}

impl FibonacciTrailingStopDeviceArrayF64Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.trailing_stop.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.trailing_stop.cols
    }
}

pub struct CudaFibonacciTrailingStopBatchResult {
    pub outputs: FibonacciTrailingStopDeviceArrayF64Quad,
    pub combos: Vec<FibonacciTrailingStopParams>,
}

pub struct CudaFibonacciTrailingStop {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn canonical_trigger_name(trigger: Option<&str>) -> String {
    trigger.unwrap_or(DEFAULT_TRIGGER).to_ascii_lowercase()
}

fn parse_trigger_mode(trigger: &str) -> Result<i32, CudaFibonacciTrailingStopError> {
    match trigger {
        "close" => Ok(TRIGGER_CLOSE),
        "wick" => Ok(TRIGGER_WICK),
        other => Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
            "invalid trigger: {other}"
        ))),
    }
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaFibonacciTrailingStopError> {
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
        return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaFibonacciTrailingStopError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }

    let mut out = Vec::new();
    let mut value = start;
    while value <= end + FLOAT_TOL {
        out.push(value.min(end));
        value += step;
    }
    if (out.last().copied().unwrap_or(start) - end).abs() > 1e-9 {
        return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_grid_fibonacci_trailing_stop(
    sweep: &FibonacciTrailingStopBatchRange,
) -> Result<Vec<FibonacciTrailingStopParams>, CudaFibonacciTrailingStopError> {
    let left_values = expand_axis_usize(sweep.left_bars.0, sweep.left_bars.1, sweep.left_bars.2)?;
    let right_values =
        expand_axis_usize(sweep.right_bars.0, sweep.right_bars.1, sweep.right_bars.2)?;
    let level_values = expand_axis_f64(sweep.level.0, sweep.level.1, sweep.level.2)?;
    let trigger = canonical_trigger_name(sweep.trigger.as_deref());
    parse_trigger_mode(&trigger)?;

    let mut combos = Vec::with_capacity(
        left_values
            .len()
            .saturating_mul(right_values.len())
            .saturating_mul(level_values.len()),
    );
    for left_bars in left_values {
        for &right_bars in &right_values {
            for &level in &level_values {
                combos.push(FibonacciTrailingStopParams {
                    left_bars: Some(left_bars),
                    right_bars: Some(right_bars),
                    level: Some(level),
                    trigger: Some(trigger.clone()),
                });
            }
        }
    }

    Ok(combos)
}

impl CudaFibonacciTrailingStop {
    pub fn new(device_id: usize) -> Result<Self, CudaFibonacciTrailingStopError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("fibonacci_trailing_stop_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaFibonacciTrailingStopError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn max_valid_run(high: &[f64], low: &[f64], close: &[f64]) -> usize {
        let mut best = 0usize;
        let mut run = 0usize;
        for i in 0..close.len() {
            if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        best
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaFibonacciTrailingStopError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaFibonacciTrailingStopError::OutOfMemory {
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
    ) -> Result<(), CudaFibonacciTrailingStopError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaFibonacciTrailingStopError::LaunchConfigTooLarge {
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
        sweep: &FibonacciTrailingStopBatchRange,
    ) -> Result<CudaFibonacciTrailingStopBatchResult, CudaFibonacciTrailingStopError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaFibonacciTrailingStopError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let max_run = Self::max_valid_run(high, low, close);
        if max_run == 0 {
            return Err(CudaFibonacciTrailingStopError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_fibonacci_trailing_stop(sweep)?;
        if combos.is_empty() {
            return Err(CudaFibonacciTrailingStopError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut left_bars = Vec::with_capacity(rows);
        let mut right_bars = Vec::with_capacity(rows);
        let mut levels = Vec::with_capacity(rows);
        let mut trigger_modes = Vec::with_capacity(rows);

        for combo in &combos {
            let left = combo.left_bars.unwrap_or(DEFAULT_LEFT_BARS);
            let right = combo.right_bars.unwrap_or(DEFAULT_RIGHT_BARS);
            let level = combo.level.unwrap_or(DEFAULT_LEVEL);
            let trigger_name = canonical_trigger_name(combo.trigger.as_deref());
            let trigger_mode = parse_trigger_mode(&trigger_name)?;

            if left == 0 {
                return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
                    "invalid left_bars: left_bars={left}, data_len={cols}"
                )));
            }
            if right == 0 {
                return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
                    "invalid right_bars: right_bars={right}, data_len={cols}"
                )));
            }
            if !level.is_finite() {
                return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
                    "invalid level: {level}"
                )));
            }

            let needed = left + right + 1;
            if needed > cols {
                return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={cols}"
                )));
            }
            if max_run < needed {
                return Err(CudaFibonacciTrailingStopError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={max_run}"
                )));
            }

            left_bars.push(i32::try_from(left).map_err(|_| {
                CudaFibonacciTrailingStopError::InvalidInput(format!(
                    "left_bars out of range: {left}"
                ))
            })?);
            right_bars.push(i32::try_from(right).map_err(|_| {
                CudaFibonacciTrailingStopError::InvalidInput(format!(
                    "right_bars out of range: {right}"
                ))
            })?);
            levels.push(level);
            trigger_modes.push(trigger_mode);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaFibonacciTrailingStopError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(3))
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaFibonacciTrailingStopError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaFibonacciTrailingStopError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaFibonacciTrailingStopError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaFibonacciTrailingStopError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_left_bars = DeviceBuffer::from_slice(&left_bars)?;
        let d_right_bars = DeviceBuffer::from_slice(&right_bars)?;
        let d_levels = DeviceBuffer::from_slice(&levels)?;
        let d_trigger_modes = DeviceBuffer::from_slice(&trigger_modes)?;
        let d_out_trailing_stop = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_long_stop = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_short_stop = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_direction = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("fibonacci_trailing_stop_batch_f64")
            .map_err(|_| CudaFibonacciTrailingStopError::MissingKernelSymbol {
                name: "fibonacci_trailing_stop_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + FIBONACCI_TRAILING_STOP_BLOCK_X - 1) / FIBONACCI_TRAILING_STOP_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(FIBONACCI_TRAILING_STOP_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                i32::try_from(cols).map_err(|_| {
                    CudaFibonacciTrailingStopError::InvalidInput("cols out of range".into())
                })?,
                d_left_bars.as_device_ptr(),
                d_right_bars.as_device_ptr(),
                d_levels.as_device_ptr(),
                d_trigger_modes.as_device_ptr(),
                i32::try_from(rows).map_err(|_| {
                    CudaFibonacciTrailingStopError::InvalidInput("rows out of range".into())
                })?,
                d_out_trailing_stop.as_device_ptr(),
                d_out_long_stop.as_device_ptr(),
                d_out_short_stop.as_device_ptr(),
                d_out_direction.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaFibonacciTrailingStopBatchResult {
            outputs: FibonacciTrailingStopDeviceArrayF64Quad {
                trailing_stop: FibonacciTrailingStopDeviceArrayF64 {
                    buf: d_out_trailing_stop,
                    rows,
                    cols,
                },
                long_stop: FibonacciTrailingStopDeviceArrayF64 {
                    buf: d_out_long_stop,
                    rows,
                    cols,
                },
                short_stop: FibonacciTrailingStopDeviceArrayF64 {
                    buf: d_out_short_stop,
                    rows,
                    cols,
                },
                direction: FibonacciTrailingStopDeviceArrayF64 {
                    buf: d_out_direction,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
