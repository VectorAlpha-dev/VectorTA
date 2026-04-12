#![cfg(feature = "cuda")]

use crate::indicators::donchian_channel_width::{
    DonchianChannelWidthBatchRange, DonchianChannelWidthParams,
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

const DONCHIAN_CHANNEL_WIDTH_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaDonchianChannelWidthError {
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

pub struct DeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
}

pub struct CudaDonchianChannelWidthBatchResult {
    pub outputs: DeviceArrayF64,
    pub combos: Vec<DonchianChannelWidthParams>,
}

pub struct CudaDonchianChannelWidth {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaDonchianChannelWidth {
    pub fn new(device_id: usize) -> Result<Self, CudaDonchianChannelWidthError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("donchian_channel_width_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaDonchianChannelWidthError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaDonchianChannelWidthError> {
        if start == 0 || end == 0 {
            return Err(CudaDonchianChannelWidthError::InvalidInput(format!(
                "invalid period range: start={start}, end={end}, step={step}"
            )));
        }
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
            while value >= end {
                out.push(value);
                if value < end + step {
                    break;
                }
                value = value.saturating_sub(step);
                if value == 0 {
                    break;
                }
            }
        }

        if out.is_empty() {
            return Err(CudaDonchianChannelWidthError::InvalidInput(format!(
                "invalid period range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &DonchianChannelWidthBatchRange,
    ) -> Result<Vec<DonchianChannelWidthParams>, CudaDonchianChannelWidthError> {
        Ok(Self::axis_usize(sweep.period)?
            .into_iter()
            .map(|period| DonchianChannelWidthParams {
                period: Some(period),
            })
            .collect())
    }

    fn longest_valid_pair_run(high: &[f64], low: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for (&h, &l) in high.iter().zip(low.iter()) {
            if h.is_finite() && l.is_finite() {
                cur += 1;
                if cur > best {
                    best = cur;
                }
            } else {
                cur = 0;
            }
        }
        best
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaDonchianChannelWidthError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaDonchianChannelWidthError::OutOfMemory {
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
    ) -> Result<(), CudaDonchianChannelWidthError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaDonchianChannelWidthError::LaunchConfigTooLarge {
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
        sweep: &DonchianChannelWidthBatchRange,
    ) -> Result<CudaDonchianChannelWidthBatchResult, CudaDonchianChannelWidthError> {
        if high.is_empty() || low.is_empty() {
            return Err(CudaDonchianChannelWidthError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() {
            return Err(CudaDonchianChannelWidthError::InvalidInput(format!(
                "input length mismatch: high_len={} low_len={}",
                high.len(),
                low.len()
            )));
        }

        let combos = Self::expand_grid(sweep)?;
        let max_period = combos
            .iter()
            .map(|params| params.period.unwrap_or(20))
            .max()
            .unwrap_or(0);
        if max_period == 0 || max_period > high.len() {
            return Err(CudaDonchianChannelWidthError::InvalidInput(format!(
                "invalid period: period={max_period}, data_len={}",
                high.len()
            )));
        }
        let longest = Self::longest_valid_pair_run(high, low);
        if longest == 0 {
            return Err(CudaDonchianChannelWidthError::InvalidInput(
                "all pairs are NaN".into(),
            ));
        }
        if longest < max_period {
            return Err(CudaDonchianChannelWidthError::InvalidInput(format!(
                "not enough valid data: needed={max_period}, valid={longest}"
            )));
        }

        let rows = combos.len();
        let cols = high.len();
        let periods: Vec<i32> = combos
            .iter()
            .map(|params| params.period.unwrap_or(20) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaDonchianChannelWidthError::InvalidInput("input bytes overflow".into())
            })?;
        let period_bytes = periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaDonchianChannelWidthError::InvalidInput("period bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaDonchianChannelWidthError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaDonchianChannelWidthError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(period_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaDonchianChannelWidthError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("donchian_channel_width_batch_f64")
            .map_err(|_| CudaDonchianChannelWidthError::MissingKernelSymbol {
                name: "donchian_channel_width_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + DONCHIAN_CHANNEL_WIDTH_BLOCK_X - 1) / DONCHIAN_CHANNEL_WIDTH_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(DONCHIAN_CHANNEL_WIDTH_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                cols as i32,
                d_periods.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaDonchianChannelWidthBatchResult {
            outputs: DeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
