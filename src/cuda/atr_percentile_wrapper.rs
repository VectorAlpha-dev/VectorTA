#![cfg(feature = "cuda")]

use crate::indicators::atr_percentile::{AtrPercentileBatchRange, AtrPercentileParams};
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

const ATR_PERCENTILE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaAtrPercentileError {
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

pub struct AtrPercentileDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AtrPercentileDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaAtrPercentileBatchResult {
    pub outputs: AtrPercentileDeviceArrayF64,
    pub combos: Vec<AtrPercentileParams>,
}

pub struct CudaAtrPercentile {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAtrPercentile {
    pub fn new(device_id: usize) -> Result<Self, CudaAtrPercentileError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("atr_percentile_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAtrPercentileError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaAtrPercentileError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            while value <= end {
                out.push(value);
                match value.checked_add(step.max(1)) {
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
                let next = value.saturating_sub(step.max(1));
                if next == value || next < end {
                    break;
                }
                value = next;
            }
        }

        if out.is_empty() {
            return Err(CudaAtrPercentileError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &AtrPercentileBatchRange,
    ) -> Result<Vec<AtrPercentileParams>, CudaAtrPercentileError> {
        let atr_lengths = Self::axis(sweep.atr_length)?;
        let percentile_lengths = Self::axis(sweep.percentile_length)?;
        let mut combos =
            Vec::with_capacity(atr_lengths.len().saturating_mul(percentile_lengths.len()));
        for atr_length in atr_lengths {
            if atr_length == 0 {
                return Err(CudaAtrPercentileError::InvalidInput(
                    "atr_length must be > 0".into(),
                ));
            }
            for percentile_length in &percentile_lengths {
                if *percentile_length == 0 {
                    return Err(CudaAtrPercentileError::InvalidInput(
                        "percentile_length must be > 0".into(),
                    ));
                }
                combos.push(AtrPercentileParams {
                    atr_length: Some(atr_length),
                    percentile_length: Some(*percentile_length),
                });
            }
        }
        Ok(combos)
    }

    fn valid_hlc_bar(high: f64, low: f64, close: f64) -> bool {
        high.is_finite() && low.is_finite() && close.is_finite()
    }

    fn first_valid_hlc(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len()).find(|&i| Self::valid_hlc_bar(high[i], low[i], close[i]))
    }

    fn count_valid_hlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
        let mut count = 0usize;
        for i in 0..close.len() {
            if Self::valid_hlc_bar(high[i], low[i], close[i]) {
                count += 1;
            }
        }
        count
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaAtrPercentileError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAtrPercentileError::OutOfMemory {
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
    ) -> Result<(), CudaAtrPercentileError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaAtrPercentileError::LaunchConfigTooLarge {
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
        sweep: &AtrPercentileBatchRange,
    ) -> Result<CudaAtrPercentileBatchResult, CudaAtrPercentileError> {
        let len = close.len();
        if len == 0 {
            return Err(CudaAtrPercentileError::InvalidInput("empty input".into()));
        }
        if high.len() != len || low.len() != len {
            return Err(CudaAtrPercentileError::InvalidInput(format!(
                "inconsistent slice lengths: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let first = Self::first_valid_hlc(high, low, close)
            .ok_or_else(|| CudaAtrPercentileError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep)?;
        let max_needed = combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(10) + combo.percentile_length.unwrap_or(50))
            .max()
            .unwrap_or(0);
        let valid = Self::count_valid_hlc(high, low, close);
        if valid < max_needed {
            return Err(CudaAtrPercentileError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={valid}"
            )));
        }
        if first >= len {
            return Err(CudaAtrPercentileError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let rows = combos.len();
        let cols = len;
        let atr_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(10) as i32)
            .collect();
        let percentile_lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.percentile_length.unwrap_or(50) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| CudaAtrPercentileError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = atr_lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| {
                percentile_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|w| v.checked_add(w))
            })
            .ok_or_else(|| CudaAtrPercentileError::InvalidInput("params bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAtrPercentileError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaAtrPercentileError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaAtrPercentileError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_atr_lengths = DeviceBuffer::from_slice(&atr_lengths)?;
        let d_percentile_lengths = DeviceBuffer::from_slice(&percentile_lengths)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("atr_percentile_batch_f64")
            .map_err(|_| CudaAtrPercentileError::MissingKernelSymbol {
                name: "atr_percentile_batch_f64",
            })?;
        let grid_x = ((rows as u32) + ATR_PERCENTILE_BLOCK_X - 1) / ATR_PERCENTILE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ATR_PERCENTILE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_atr_lengths.as_device_ptr(),
                d_percentile_lengths.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaAtrPercentileBatchResult {
            outputs: AtrPercentileDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
