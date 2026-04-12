#![cfg(feature = "cuda")]

use crate::indicators::trend_direction_force_index::{
    TrendDirectionForceIndexBatchRange, TrendDirectionForceIndexParams,
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

const TREND_DIRECTION_FORCE_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaTrendDirectionForceIndexError {
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

pub struct TrendDirectionForceIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl TrendDirectionForceIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaTrendDirectionForceIndexBatchResult {
    pub outputs: TrendDirectionForceIndexDeviceArrayF64,
    pub combos: Vec<TrendDirectionForceIndexParams>,
}

pub struct CudaTrendDirectionForceIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaTrendDirectionForceIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaTrendDirectionForceIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("trend_direction_force_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaTrendDirectionForceIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn half_length(length: usize) -> usize {
        (length / 2).max(1)
    }

    fn required_samples(length: usize) -> usize {
        Self::half_length(length).saturating_mul(2)
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

    fn expand_grid(
        range: &TrendDirectionForceIndexBatchRange,
    ) -> Result<Vec<TrendDirectionForceIndexParams>, CudaTrendDirectionForceIndexError> {
        let (start, end, step) = range.length;
        if start == 0 || end == 0 {
            return Err(CudaTrendDirectionForceIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        if step == 0 {
            return Ok(vec![TrendDirectionForceIndexParams {
                length: Some(start),
            }]);
        }
        if start > end {
            return Err(CudaTrendDirectionForceIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }

        let mut out = Vec::new();
        let mut cur = start;
        loop {
            out.push(TrendDirectionForceIndexParams { length: Some(cur) });
            if cur >= end {
                break;
            }
            let next = cur.saturating_add(step);
            if next <= cur {
                return Err(CudaTrendDirectionForceIndexError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            cur = next.min(end);
            if cur == out.last().and_then(|p| p.length).unwrap_or(cur) {
                break;
            }
        }
        Ok(out)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaTrendDirectionForceIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaTrendDirectionForceIndexError::OutOfMemory {
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
    ) -> Result<(), CudaTrendDirectionForceIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaTrendDirectionForceIndexError::LaunchConfigTooLarge {
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
        sweep: &TrendDirectionForceIndexBatchRange,
    ) -> Result<CudaTrendDirectionForceIndexBatchResult, CudaTrendDirectionForceIndexError> {
        if data.is_empty() {
            return Err(CudaTrendDirectionForceIndexError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let max_length = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(10))
            .max()
            .unwrap_or(0);
        let max_run = Self::longest_valid_run(data);
        if max_run == 0 {
            return Err(CudaTrendDirectionForceIndexError::InvalidInput(
                "all values are NaN".into(),
            ));
        }
        let needed = Self::required_samples(max_length);
        if max_run < needed {
            return Err(CudaTrendDirectionForceIndexError::InvalidInput(format!(
                "not enough valid data: needed={needed}, valid={max_run}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(10))
            .map(|length| {
                if length == 0 {
                    Err(CudaTrendDirectionForceIndexError::InvalidInput(
                        "length must be > 0".into(),
                    ))
                } else {
                    Ok(length as i32)
                }
            })
            .collect::<Result<_, _>>()?;
        let max_norm_window = max_length.saturating_mul(3).max(1);

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaTrendDirectionForceIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaTrendDirectionForceIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaTrendDirectionForceIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaTrendDirectionForceIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_len = rows.checked_mul(max_norm_window).ok_or_else(|| {
            CudaTrendDirectionForceIndexError::InvalidInput("rows*norm_window overflow".into())
        })?;
        let scratch_bytes = scratch_len
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                scratch_len
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaTrendDirectionForceIndexError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaTrendDirectionForceIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_deque_indices = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_len)? };
        let d_deque_values = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_len)? };
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("trend_direction_force_index_batch_f64")
            .map_err(|_| CudaTrendDirectionForceIndexError::MissingKernelSymbol {
                name: "trend_direction_force_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + TREND_DIRECTION_FORCE_INDEX_BLOCK_X - 1)
            / TREND_DIRECTION_FORCE_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(TREND_DIRECTION_FORCE_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                max_norm_window as i32,
                d_deque_indices.as_device_ptr(),
                d_deque_values.as_device_ptr(),
                d_out.as_device_ptr()
            ))?;
        }

        Ok(CudaTrendDirectionForceIndexBatchResult {
            outputs: TrendDirectionForceIndexDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
