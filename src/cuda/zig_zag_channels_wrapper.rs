#![cfg(feature = "cuda")]

use crate::indicators::zig_zag_channels::{ZigZagChannelsBatchRange, ZigZagChannelsParams};
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

const ZIG_ZAG_CHANNELS_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaZigZagChannelsError {
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

pub struct ZigZagChannelsDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl ZigZagChannelsDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct ZigZagChannelsDeviceArrayF64Triple {
    pub middle: ZigZagChannelsDeviceArrayF64,
    pub upper: ZigZagChannelsDeviceArrayF64,
    pub lower: ZigZagChannelsDeviceArrayF64,
}

impl ZigZagChannelsDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.middle.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.middle.cols
    }
}

pub struct CudaZigZagChannelsBatchResult {
    pub outputs: ZigZagChannelsDeviceArrayF64Triple,
    pub combos: Vec<ZigZagChannelsParams>,
}

pub struct CudaZigZagChannels {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn expand_grid_checked(
    range: &ZigZagChannelsBatchRange,
) -> Result<Vec<ZigZagChannelsParams>, CudaZigZagChannelsError> {
    let (start, end, step) = range.length;
    if start == 0 || end == 0 {
        return Err(CudaZigZagChannelsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step == 0 {
        return Ok(vec![ZigZagChannelsParams {
            length: Some(start),
            extend: Some(range.extend),
        }]);
    }
    if start > end {
        return Err(CudaZigZagChannelsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }

    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(ZigZagChannelsParams {
            length: Some(current),
            extend: Some(range.extend),
        });
        if current >= end {
            break;
        }
        let next = current.saturating_add(step);
        if next <= current {
            return Err(CudaZigZagChannelsError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        current = next.min(end);
        if current == out.last().and_then(|item| item.length).unwrap_or(0) {
            break;
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
    let mut cur = 0usize;
    for i in 0..close.len() {
        if is_valid_ohlc(open[i], high[i], low[i], close[i]) {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

impl CudaZigZagChannels {
    pub fn new(device_id: usize) -> Result<Self, CudaZigZagChannelsError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("zig_zag_channels_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaZigZagChannelsError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaZigZagChannelsError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaZigZagChannelsError::OutOfMemory {
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
    ) -> Result<(), CudaZigZagChannelsError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaZigZagChannelsError::LaunchConfigTooLarge {
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
        sweep: &ZigZagChannelsBatchRange,
    ) -> Result<CudaZigZagChannelsBatchResult, CudaZigZagChannelsError> {
        if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaZigZagChannelsError::InvalidInput("empty input".into()));
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(CudaZigZagChannelsError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}",
                open.len(),
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = expand_grid_checked(sweep)?;
        let max_length = combos
            .iter()
            .map(|params| params.length.unwrap_or(100))
            .max()
            .unwrap_or(0);
        if max_length == 0 {
            return Err(CudaZigZagChannelsError::InvalidInput(
                "invalid length".into(),
            ));
        }
        let longest = longest_valid_run(open, high, low, close);
        if longest == 0 {
            return Err(CudaZigZagChannelsError::InvalidInput(
                "all values are NaN".into(),
            ));
        }
        let needed = max_length
            .checked_add(1)
            .ok_or_else(|| CudaZigZagChannelsError::InvalidInput("length overflow".into()))?;
        if longest < needed {
            return Err(CudaZigZagChannelsError::InvalidInput(format!(
                "not enough valid data: needed={needed}, valid={longest}"
            )));
        }

        let rows = combos.len();
        let cols = close.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.length.unwrap_or(100) as i32)
            .collect();
        let extends: Vec<i32> = combos
            .iter()
            .map(|params| i32::from(params.extend.unwrap_or(true)))
            .collect();

        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaZigZagChannelsError::InvalidInput("rows*cols overflow".into()))?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| CudaZigZagChannelsError::InvalidInput("input bytes overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| CudaZigZagChannelsError::InvalidInput("param bytes overflow".into()))?;
        let scratch_elems = rows
            .checked_mul(max_length)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaZigZagChannelsError::InvalidInput("scratch elems overflow".into())
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaZigZagChannelsError::InvalidInput("scratch bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| CudaZigZagChannelsError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaZigZagChannelsError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_extends = DeviceBuffer::from_slice(&extends)?;
        let mut d_scratch = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_elems)? };
        let mut d_middle = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("zig_zag_channels_batch_f64")
            .map_err(|_| CudaZigZagChannelsError::MissingKernelSymbol {
                name: "zig_zag_channels_batch_f64",
            })?;
        let grid_x = ((rows as u32) + ZIG_ZAG_CHANNELS_BLOCK_X - 1) / ZIG_ZAG_CHANNELS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ZIG_ZAG_CHANNELS_BLOCK_X);
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
                d_extends.as_device_ptr(),
                rows as i32,
                max_length as i32,
                d_scratch.as_device_ptr(),
                d_middle.as_device_ptr(),
                d_upper.as_device_ptr(),
                d_lower.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaZigZagChannelsBatchResult {
            outputs: ZigZagChannelsDeviceArrayF64Triple {
                middle: ZigZagChannelsDeviceArrayF64 {
                    buf: d_middle,
                    rows,
                    cols,
                },
                upper: ZigZagChannelsDeviceArrayF64 {
                    buf: d_upper,
                    rows,
                    cols,
                },
                lower: ZigZagChannelsDeviceArrayF64 {
                    buf: d_lower,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
