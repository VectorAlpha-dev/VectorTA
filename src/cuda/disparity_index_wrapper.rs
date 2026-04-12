#![cfg(feature = "cuda")]

use crate::indicators::disparity_index::{DisparityIndexBatchRange, DisparityIndexParams};
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

const DISPARITY_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaDisparityIndexError {
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

pub struct DisparityIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DisparityIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaDisparityIndexBatchResult {
    pub outputs: DisparityIndexDeviceArrayF64,
    pub combos: Vec<DisparityIndexParams>,
}

pub struct CudaDisparityIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaDisparityIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaDisparityIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("disparity_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaDisparityIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn normalize_smoothing_type(value: &str) -> Option<i32> {
        let normalized = value.trim();
        if normalized.eq_ignore_ascii_case("ema") {
            Some(0)
        } else if normalized.eq_ignore_ascii_case("sma") {
            Some(1)
        } else {
            None
        }
    }

    fn longest_valid_run(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for &value in data {
            if value.is_finite() {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        best
    }

    fn warmup_prefix(ema_period: usize, lookback_period: usize, smoothing_period: usize) -> usize {
        ema_period
            .saturating_add(lookback_period)
            .saturating_add(smoothing_period)
            .saturating_sub(3)
    }

    fn expand_axis(
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Vec<usize>, CudaDisparityIndexError> {
        if start == 0 || end == 0 {
            return Err(CudaDisparityIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        if step == 0 {
            return Ok(vec![start]);
        }
        if start > end {
            return Err(CudaDisparityIndexError::InvalidInput(format!(
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
            let next = current.saturating_add(step);
            if next <= current {
                return Err(CudaDisparityIndexError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            current = next.min(end);
        }
        Ok(out)
    }

    fn expand_grid(
        range: &DisparityIndexBatchRange,
    ) -> Result<Vec<DisparityIndexParams>, CudaDisparityIndexError> {
        let ema_periods =
            Self::expand_axis(range.ema_period.0, range.ema_period.1, range.ema_period.2)?;
        let lookbacks = Self::expand_axis(
            range.lookback_period.0,
            range.lookback_period.1,
            range.lookback_period.2,
        )?;
        let smoothing_periods = Self::expand_axis(
            range.smoothing_period.0,
            range.smoothing_period.1,
            range.smoothing_period.2,
        )?;
        let smoothing_types = if range.smoothing_types.is_empty() {
            vec!["ema".to_string()]
        } else {
            range.smoothing_types.clone()
        };

        let total = ema_periods
            .len()
            .checked_mul(lookbacks.len())
            .and_then(|value| value.checked_mul(smoothing_periods.len()))
            .and_then(|value| value.checked_mul(smoothing_types.len()))
            .ok_or_else(|| {
                CudaDisparityIndexError::InvalidInput("parameter grid size overflow".into())
            })?;

        let mut out = Vec::with_capacity(total);
        for &ema_period in &ema_periods {
            for &lookback_period in &lookbacks {
                for &smoothing_period in &smoothing_periods {
                    for smoothing_type in &smoothing_types {
                        if Self::normalize_smoothing_type(smoothing_type).is_none() {
                            return Err(CudaDisparityIndexError::InvalidInput(format!(
                                "invalid smoothing_type: {smoothing_type}"
                            )));
                        }
                        out.push(DisparityIndexParams {
                            ema_period: Some(ema_period),
                            lookback_period: Some(lookback_period),
                            smoothing_period: Some(smoothing_period),
                            smoothing_type: Some(smoothing_type.clone()),
                        });
                    }
                }
            }
        }
        Ok(out)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaDisparityIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaDisparityIndexError::OutOfMemory {
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
    ) -> Result<(), CudaDisparityIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaDisparityIndexError::LaunchConfigTooLarge {
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
        range: &DisparityIndexBatchRange,
    ) -> Result<CudaDisparityIndexBatchResult, CudaDisparityIndexError> {
        if data.is_empty() {
            return Err(CudaDisparityIndexError::InvalidInput("empty input".into()));
        }
        let longest = Self::longest_valid_run(data);
        if longest == 0 {
            return Err(CudaDisparityIndexError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = Self::expand_grid(range)?;
        let rows = combos.len();
        let cols = data.len();
        let mut max_lookback = 0usize;
        let mut max_smoothing = 0usize;
        let mut max_needed = 0usize;
        let ema_periods: Vec<i32> = combos
            .iter()
            .map(|combo| combo.ema_period.unwrap_or(14))
            .map(|value| {
                max_needed = max_needed.max(Self::warmup_prefix(value, 1, 1));
                value as i32
            })
            .collect();
        let lookback_periods: Vec<i32> = combos
            .iter()
            .map(|combo| combo.lookback_period.unwrap_or(14))
            .map(|value| {
                max_lookback = max_lookback.max(value);
                value as i32
            })
            .collect();
        let smoothing_periods: Vec<i32> = combos
            .iter()
            .map(|combo| combo.smoothing_period.unwrap_or(9))
            .map(|value| {
                max_smoothing = max_smoothing.max(value);
                value as i32
            })
            .collect();
        let smoothing_flags: Vec<i32> = combos
            .iter()
            .map(|combo| {
                Self::normalize_smoothing_type(combo.smoothing_type.as_deref().unwrap_or("ema"))
                    .ok_or_else(|| {
                        CudaDisparityIndexError::InvalidInput(format!(
                            "invalid smoothing_type: {}",
                            combo.smoothing_type.as_deref().unwrap_or("ema")
                        ))
                    })
            })
            .collect::<Result<_, _>>()?;

        for combo in &combos {
            let ema_period = combo.ema_period.unwrap_or(14);
            let lookback_period = combo.lookback_period.unwrap_or(14);
            let smoothing_period = combo.smoothing_period.unwrap_or(9);
            if ema_period == 0 || lookback_period == 0 || smoothing_period == 0 {
                return Err(CudaDisparityIndexError::InvalidInput(
                    "periods must be > 0".into(),
                ));
            }
            let needed = Self::warmup_prefix(ema_period, lookback_period, smoothing_period)
                .saturating_add(1);
            max_needed = max_needed.max(needed);
        }
        if longest < max_needed {
            return Err(CudaDisparityIndexError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={longest}"
            )));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaDisparityIndexError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = ema_periods
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                lookback_periods
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                smoothing_periods
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                smoothing_flags
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| CudaDisparityIndexError::InvalidInput("params bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaDisparityIndexError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaDisparityIndexError::InvalidInput("output bytes overflow".into()))?;
        let disparity_scratch = rows.checked_mul(max_lookback).ok_or_else(|| {
            CudaDisparityIndexError::InvalidInput("rows*max_lookback overflow".into())
        })?;
        let sma_scratch = rows.checked_mul(max_smoothing).ok_or_else(|| {
            CudaDisparityIndexError::InvalidInput("rows*max_smoothing overflow".into())
        })?;
        let scratch_bytes = disparity_scratch
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                sma_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaDisparityIndexError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaDisparityIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_ema_periods = DeviceBuffer::from_slice(&ema_periods)?;
        let d_lookback_periods = DeviceBuffer::from_slice(&lookback_periods)?;
        let d_smoothing_periods = DeviceBuffer::from_slice(&smoothing_periods)?;
        let d_smoothing_flags = DeviceBuffer::from_slice(&smoothing_flags)?;
        let d_disparity_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(disparity_scratch)? };
        let d_sma_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(sma_scratch)? };
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("disparity_index_batch_f64")
            .map_err(|_| CudaDisparityIndexError::MissingKernelSymbol {
                name: "disparity_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + DISPARITY_INDEX_BLOCK_X - 1) / DISPARITY_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(DISPARITY_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_ema_periods.as_device_ptr(),
                d_lookback_periods.as_device_ptr(),
                d_smoothing_periods.as_device_ptr(),
                d_smoothing_flags.as_device_ptr(),
                rows as i32,
                max_lookback as i32,
                max_smoothing as i32,
                d_disparity_buffer.as_device_ptr(),
                d_sma_buffer.as_device_ptr(),
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaDisparityIndexBatchResult {
            outputs: DisparityIndexDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
