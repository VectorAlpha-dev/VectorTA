#![cfg(feature = "cuda")]

use crate::indicators::neighboring_trailing_stop::{
    NeighboringTrailingStopBatchRange, NeighboringTrailingStopParams,
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

const NEIGHBORING_TRAILING_STOP_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_BUFFER_SIZE: usize = 200;
const DEFAULT_K: usize = 50;
const DEFAULT_PERCENTILE: f64 = 90.0;
const DEFAULT_SMOOTH: usize = 5;
const MIN_BUFFER_SIZE: usize = 100;
const MIN_K: usize = 5;
const FLOAT_TOL: f64 = 1e-12;

#[derive(Debug, Error)]
pub enum CudaNeighboringTrailingStopError {
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

pub struct NeighboringTrailingStopDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl NeighboringTrailingStopDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct NeighboringTrailingStopDeviceArrayF64Hex {
    pub trailing_stop: NeighboringTrailingStopDeviceArrayF64,
    pub bullish_band: NeighboringTrailingStopDeviceArrayF64,
    pub bearish_band: NeighboringTrailingStopDeviceArrayF64,
    pub direction: NeighboringTrailingStopDeviceArrayF64,
    pub discovery_bull: NeighboringTrailingStopDeviceArrayF64,
    pub discovery_bear: NeighboringTrailingStopDeviceArrayF64,
}

impl NeighboringTrailingStopDeviceArrayF64Hex {
    #[inline]
    pub fn rows(&self) -> usize {
        self.trailing_stop.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.trailing_stop.cols
    }
}

pub struct CudaNeighboringTrailingStopBatchResult {
    pub outputs: NeighboringTrailingStopDeviceArrayF64Hex,
    pub combos: Vec<NeighboringTrailingStopParams>,
}

pub struct CudaNeighboringTrailingStop {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

fn first_valid_ohlc(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..high.len()).find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaNeighboringTrailingStopError> {
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
        return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, CudaNeighboringTrailingStopError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
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
        return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

fn expand_grid_neighboring_trailing_stop(
    sweep: &NeighboringTrailingStopBatchRange,
) -> Result<Vec<NeighboringTrailingStopParams>, CudaNeighboringTrailingStopError> {
    let buffer_sizes = expand_axis_usize(
        sweep.buffer_size.0,
        sweep.buffer_size.1,
        sweep.buffer_size.2,
    )?;
    let ks = expand_axis_usize(sweep.k.0, sweep.k.1, sweep.k.2)?;
    let percentiles = expand_axis_f64(sweep.percentile.0, sweep.percentile.1, sweep.percentile.2)?;
    let smooths = expand_axis_usize(sweep.smooth.0, sweep.smooth.1, sweep.smooth.2)?;

    let mut combos = Vec::with_capacity(
        buffer_sizes
            .len()
            .saturating_mul(ks.len())
            .saturating_mul(percentiles.len())
            .saturating_mul(smooths.len()),
    );
    for buffer_size in buffer_sizes {
        for &k in &ks {
            for &percentile in &percentiles {
                for &smooth in &smooths {
                    combos.push(NeighboringTrailingStopParams {
                        buffer_size: Some(buffer_size),
                        k: Some(k),
                        percentile: Some(percentile),
                        smooth: Some(smooth),
                    });
                }
            }
        }
    }
    Ok(combos)
}

impl CudaNeighboringTrailingStop {
    pub fn new(device_id: usize) -> Result<Self, CudaNeighboringTrailingStopError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("neighboring_trailing_stop_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaNeighboringTrailingStopError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaNeighboringTrailingStopError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaNeighboringTrailingStopError::OutOfMemory {
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
    ) -> Result<(), CudaNeighboringTrailingStopError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaNeighboringTrailingStopError::LaunchConfigTooLarge {
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
        sweep: &NeighboringTrailingStopBatchRange,
    ) -> Result<CudaNeighboringTrailingStopBatchResult, CudaNeighboringTrailingStopError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaNeighboringTrailingStopError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }
        if first_valid_ohlc(high, low, close).is_none() {
            return Err(CudaNeighboringTrailingStopError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_neighboring_trailing_stop(sweep)?;
        if combos.is_empty() {
            return Err(CudaNeighboringTrailingStopError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut buffer_sizes = Vec::with_capacity(rows);
        let mut ks = Vec::with_capacity(rows);
        let mut percentiles = Vec::with_capacity(rows);
        let mut smooths = Vec::with_capacity(rows);
        let mut max_buffer_size = 0usize;
        let mut max_smooth = 0usize;

        for combo in &combos {
            let buffer_size = combo.buffer_size.unwrap_or(DEFAULT_BUFFER_SIZE);
            let k = combo.k.unwrap_or(DEFAULT_K);
            let percentile = combo.percentile.unwrap_or(DEFAULT_PERCENTILE);
            let smooth = combo.smooth.unwrap_or(DEFAULT_SMOOTH);

            if buffer_size < MIN_BUFFER_SIZE {
                return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
                    "invalid buffer_size: buffer_size={buffer_size}, min={MIN_BUFFER_SIZE}"
                )));
            }
            if k < MIN_K {
                return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
                    "invalid k: k={k}, min={MIN_K}"
                )));
            }
            if !percentile.is_finite() || !(1.0..=99.0).contains(&percentile) {
                return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
                    "invalid percentile: {percentile}"
                )));
            }
            if smooth == 0 {
                return Err(CudaNeighboringTrailingStopError::InvalidInput(format!(
                    "invalid smooth: {smooth}"
                )));
            }

            buffer_sizes.push(i32::try_from(buffer_size).map_err(|_| {
                CudaNeighboringTrailingStopError::InvalidInput(format!(
                    "buffer_size out of range: {buffer_size}"
                ))
            })?);
            ks.push(i32::try_from(k).map_err(|_| {
                CudaNeighboringTrailingStopError::InvalidInput(format!("k out of range: {k}"))
            })?);
            percentiles.push(percentile);
            smooths.push(i32::try_from(smooth).map_err(|_| {
                CudaNeighboringTrailingStopError::InvalidInput(format!(
                    "smooth out of range: {smooth}"
                ))
            })?);
            max_buffer_size = max_buffer_size.max(buffer_size);
            max_smooth = max_smooth.max(smooth);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaNeighboringTrailingStopError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(3))
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaNeighboringTrailingStopError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaNeighboringTrailingStopError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(6))
            .ok_or_else(|| {
                CudaNeighboringTrailingStopError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_elems = rows
            .checked_mul(max_buffer_size)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| {
                rows.checked_mul(max_smooth)
                    .and_then(|other| other.checked_mul(2))
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaNeighboringTrailingStopError::InvalidInput("scratch elements overflow".into())
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaNeighboringTrailingStopError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaNeighboringTrailingStopError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_buffer_sizes = DeviceBuffer::from_slice(&buffer_sizes)?;
        let d_ks = DeviceBuffer::from_slice(&ks)?;
        let d_percentiles = DeviceBuffer::from_slice(&percentiles)?;
        let d_smooths = DeviceBuffer::from_slice(&smooths)?;
        let d_out_trailing_stop = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bullish_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bearish_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_direction = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_discovery_bull = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_discovery_bear = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_price_buffers =
            unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_buffer_size)? };
        let d_sorted_buffers =
            unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_buffer_size)? };
        let d_bull_sma_buffers = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_smooth)? };
        let d_bear_sma_buffers = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_smooth)? };

        let func = self
            .module
            .get_function("neighboring_trailing_stop_batch_f64")
            .map_err(|_| CudaNeighboringTrailingStopError::MissingKernelSymbol {
                name: "neighboring_trailing_stop_batch_f64",
            })?;
        let grid_x = ((rows as u32) + NEIGHBORING_TRAILING_STOP_BLOCK_X - 1)
            / NEIGHBORING_TRAILING_STOP_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(NEIGHBORING_TRAILING_STOP_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                i32::try_from(cols).map_err(|_| {
                    CudaNeighboringTrailingStopError::InvalidInput("cols out of range".into())
                })?,
                d_buffer_sizes.as_device_ptr(),
                d_ks.as_device_ptr(),
                d_percentiles.as_device_ptr(),
                d_smooths.as_device_ptr(),
                i32::try_from(rows).map_err(|_| {
                    CudaNeighboringTrailingStopError::InvalidInput("rows out of range".into())
                })?,
                i32::try_from(max_buffer_size).map_err(|_| {
                    CudaNeighboringTrailingStopError::InvalidInput(
                        "max_buffer_size out of range".into(),
                    )
                })?,
                i32::try_from(max_smooth).map_err(|_| {
                    CudaNeighboringTrailingStopError::InvalidInput(
                        "max_smooth out of range".into(),
                    )
                })?,
                d_out_trailing_stop.as_device_ptr(),
                d_out_bullish_band.as_device_ptr(),
                d_out_bearish_band.as_device_ptr(),
                d_out_direction.as_device_ptr(),
                d_out_discovery_bull.as_device_ptr(),
                d_out_discovery_bear.as_device_ptr(),
                d_price_buffers.as_device_ptr(),
                d_sorted_buffers.as_device_ptr(),
                d_bull_sma_buffers.as_device_ptr(),
                d_bear_sma_buffers.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaNeighboringTrailingStopBatchResult {
            outputs: NeighboringTrailingStopDeviceArrayF64Hex {
                trailing_stop: NeighboringTrailingStopDeviceArrayF64 {
                    buf: d_out_trailing_stop,
                    rows,
                    cols,
                },
                bullish_band: NeighboringTrailingStopDeviceArrayF64 {
                    buf: d_out_bullish_band,
                    rows,
                    cols,
                },
                bearish_band: NeighboringTrailingStopDeviceArrayF64 {
                    buf: d_out_bearish_band,
                    rows,
                    cols,
                },
                direction: NeighboringTrailingStopDeviceArrayF64 {
                    buf: d_out_direction,
                    rows,
                    cols,
                },
                discovery_bull: NeighboringTrailingStopDeviceArrayF64 {
                    buf: d_out_discovery_bull,
                    rows,
                    cols,
                },
                discovery_bear: NeighboringTrailingStopDeviceArrayF64 {
                    buf: d_out_discovery_bear,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
