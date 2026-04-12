#![cfg(feature = "cuda")]

use crate::indicators::rolling_z_score_trend::{
    RollingZScoreTrendBatchRange, RollingZScoreTrendParams,
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

const ROLLING_Z_SCORE_TREND_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaRollingZScoreTrendError {
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
}

pub struct RollingZScoreTrendDeviceArrayF64Pair {
    pub zscore: DeviceArrayF64,
    pub momentum: DeviceArrayF64,
}

impl RollingZScoreTrendDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.zscore.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.zscore.cols
    }
}

pub struct CudaRollingZScoreTrendBatchResult {
    pub outputs: RollingZScoreTrendDeviceArrayF64Pair,
    pub combos: Vec<RollingZScoreTrendParams>,
}

pub struct CudaRollingZScoreTrend {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaRollingZScoreTrend {
    pub fn new(device_id: usize) -> Result<Self, CudaRollingZScoreTrendError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("rolling_z_score_trend_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaRollingZScoreTrendError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaRollingZScoreTrendError> {
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
            return Err(CudaRollingZScoreTrendError::InvalidInput(format!(
                "invalid lookback range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &RollingZScoreTrendBatchRange,
    ) -> Result<Vec<RollingZScoreTrendParams>, CudaRollingZScoreTrendError> {
        Ok(Self::axis_usize(sweep.lookback_period)?
            .into_iter()
            .map(|lookback_period| RollingZScoreTrendParams {
                lookback_period: Some(lookback_period),
            })
            .collect())
    }

    fn longest_valid_run(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for &value in data {
            if value.is_finite() {
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

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaRollingZScoreTrendError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaRollingZScoreTrendError::OutOfMemory {
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
    ) -> Result<(), CudaRollingZScoreTrendError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaRollingZScoreTrendError::LaunchConfigTooLarge {
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
        data: &[f64],
        sweep: &RollingZScoreTrendBatchRange,
    ) -> Result<CudaRollingZScoreTrendBatchResult, CudaRollingZScoreTrendError> {
        if data.is_empty() {
            return Err(CudaRollingZScoreTrendError::InvalidInput(
                "empty data".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let max_lookback = combos
            .iter()
            .map(|params| params.lookback_period.unwrap_or(20))
            .max()
            .unwrap_or(0);
        if max_lookback == 0 || max_lookback > data.len() {
            return Err(CudaRollingZScoreTrendError::InvalidInput(format!(
                "invalid lookback_period: lookback_period={max_lookback}, data_len={}",
                data.len()
            )));
        }

        let longest = Self::longest_valid_run(data);
        if longest == 0 {
            return Err(CudaRollingZScoreTrendError::InvalidInput(
                "all values are NaN".into(),
            ));
        }
        if longest < max_lookback {
            return Err(CudaRollingZScoreTrendError::InvalidInput(format!(
                "not enough valid data: needed={max_lookback}, valid={longest}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let lookbacks: Vec<i32> = combos
            .iter()
            .map(|params| params.lookback_period.unwrap_or(20) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaRollingZScoreTrendError::InvalidInput("input bytes overflow".into())
            })?;
        let lookback_bytes = lookbacks
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaRollingZScoreTrendError::InvalidInput("lookback bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaRollingZScoreTrendError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| {
                CudaRollingZScoreTrendError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(lookback_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaRollingZScoreTrendError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lookbacks = DeviceBuffer::from_slice(&lookbacks)?;
        let mut d_zscore = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let mut d_momentum = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("rolling_z_score_trend_batch_f64")
            .map_err(|_| CudaRollingZScoreTrendError::MissingKernelSymbol {
                name: "rolling_z_score_trend_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + ROLLING_Z_SCORE_TREND_BLOCK_X - 1) / ROLLING_Z_SCORE_TREND_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ROLLING_Z_SCORE_TREND_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lookbacks.as_device_ptr(),
                rows as i32,
                d_zscore.as_device_ptr(),
                d_momentum.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaRollingZScoreTrendBatchResult {
            outputs: RollingZScoreTrendDeviceArrayF64Pair {
                zscore: DeviceArrayF64 {
                    buf: d_zscore,
                    rows,
                    cols,
                },
                momentum: DeviceArrayF64 {
                    buf: d_momentum,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
