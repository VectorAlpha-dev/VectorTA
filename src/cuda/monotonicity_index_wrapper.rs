#![cfg(feature = "cuda")]

use crate::indicators::monotonicity_index::{
    MonotonicityIndexBatchRange, MonotonicityIndexMode, MonotonicityIndexParams,
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

const MONOTONICITY_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaMonotonicityIndexError {
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

pub struct MonotonicityIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MonotonicityIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct MonotonicityIndexDeviceArrayF64Triplet {
    pub index: MonotonicityIndexDeviceArrayF64,
    pub cumulative_mean: MonotonicityIndexDeviceArrayF64,
    pub upper_bound: MonotonicityIndexDeviceArrayF64,
}

impl MonotonicityIndexDeviceArrayF64Triplet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.index.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.index.cols
    }
}

pub struct CudaMonotonicityIndexBatchResult {
    pub outputs: MonotonicityIndexDeviceArrayF64Triplet,
    pub combos: Vec<MonotonicityIndexParams>,
}

pub struct CudaMonotonicityIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaMonotonicityIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaMonotonicityIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("monotonicity_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaMonotonicityIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn max_consecutive_valid_values(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut run = 0usize;
        for &value in data {
            if value.is_finite() {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        best
    }

    fn expand_axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaMonotonicityIndexError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end {
                out.push(x);
                let next = x.saturating_add(step);
                if next == x {
                    break;
                }
                x = next;
            }
        } else {
            let mut x = start;
            loop {
                out.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(step);
                if next == x || next < end {
                    break;
                }
                x = next;
            }
        }

        if out.is_empty() {
            return Err(CudaMonotonicityIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &MonotonicityIndexBatchRange,
    ) -> Result<Vec<MonotonicityIndexParams>, CudaMonotonicityIndexError> {
        let lengths = Self::expand_axis_usize(sweep.length)?;
        let index_smooths = Self::expand_axis_usize(sweep.index_smooth)?;

        let mut combos = Vec::with_capacity(lengths.len().saturating_mul(index_smooths.len()));
        for length in lengths {
            if length < 2 {
                return Err(CudaMonotonicityIndexError::InvalidInput(format!(
                    "invalid length: {length}"
                )));
            }
            for index_smooth in index_smooths.iter().copied() {
                if index_smooth == 0 {
                    return Err(CudaMonotonicityIndexError::InvalidInput(
                        "index_smooth must be > 0".into(),
                    ));
                }
                combos.push(MonotonicityIndexParams {
                    length: Some(length),
                    mode: Some(sweep.mode),
                    index_smooth: Some(index_smooth),
                });
            }
        }
        Ok(combos)
    }

    fn mode_flag(mode: MonotonicityIndexMode) -> i32 {
        match mode {
            MonotonicityIndexMode::Efficiency => 0,
            MonotonicityIndexMode::Complexity => 1,
        }
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaMonotonicityIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaMonotonicityIndexError::OutOfMemory {
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
    ) -> Result<(), CudaMonotonicityIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaMonotonicityIndexError::LaunchConfigTooLarge {
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
        sweep: &MonotonicityIndexBatchRange,
    ) -> Result<CudaMonotonicityIndexBatchResult, CudaMonotonicityIndexError> {
        if data.is_empty() {
            return Err(CudaMonotonicityIndexError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let first = Self::first_valid_value(data)
            .ok_or_else(|| CudaMonotonicityIndexError::InvalidInput("all values are NaN".into()))?;
        let valid = Self::max_consecutive_valid_values(data);
        let mut max_needed = 0usize;
        let mut max_length = 0usize;
        let mut max_index_smooth = 0usize;
        for combo in &combos {
            let length = combo.length.unwrap_or(20);
            let index_smooth = combo.index_smooth.unwrap_or(5);
            let needed = length.saturating_add(index_smooth).saturating_sub(1);
            max_needed = max_needed.max(needed);
            max_length = max_length.max(length);
            max_index_smooth = max_index_smooth.max(index_smooth);
        }
        if valid < max_needed {
            return Err(CudaMonotonicityIndexError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(20) as i32)
            .collect();
        let index_smooths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.index_smooth.unwrap_or(5) as i32)
            .collect();
        let mode_flags: Vec<i32> = combos
            .iter()
            .map(|combo| Self::mode_flag(combo.mode.unwrap_or_default()))
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaMonotonicityIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                index_smooths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                mode_flags
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaMonotonicityIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaMonotonicityIndexError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaMonotonicityIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let length_scratch = rows.checked_mul(max_length).ok_or_else(|| {
            CudaMonotonicityIndexError::InvalidInput("rows*max_length overflow".into())
        })?;
        let smooth_scratch = rows.checked_mul(max_index_smooth).ok_or_else(|| {
            CudaMonotonicityIndexError::InvalidInput("rows*max_index_smooth overflow".into())
        })?;
        let scratch_bytes = length_scratch
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .and_then(|value| {
                length_scratch
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| other.checked_mul(2))
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                smooth_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaMonotonicityIndexError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaMonotonicityIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_index_smooths = DeviceBuffer::from_slice(&index_smooths)?;
        let d_mode_flags = DeviceBuffer::from_slice(&mode_flags)?;
        let d_window_ring = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_window_copy = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_inc_pool_vals = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_inc_pool_weights = unsafe { DeviceBuffer::<i32>::uninitialized(length_scratch)? };
        let d_dec_pool_vals = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_dec_pool_weights = unsafe { DeviceBuffer::<i32>::uninitialized(length_scratch)? };
        let d_sma_buf = unsafe { DeviceBuffer::<f64>::uninitialized(smooth_scratch)? };
        let d_out_index = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_cumulative = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("monotonicity_index_batch_f64")
            .map_err(|_| CudaMonotonicityIndexError::MissingKernelSymbol {
                name: "monotonicity_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + MONOTONICITY_INDEX_BLOCK_X - 1) / MONOTONICITY_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(MONOTONICITY_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_index_smooths.as_device_ptr(),
                d_mode_flags.as_device_ptr(),
                rows as i32,
                max_length as i32,
                max_index_smooth as i32,
                d_window_ring.as_device_ptr(),
                d_window_copy.as_device_ptr(),
                d_inc_pool_vals.as_device_ptr(),
                d_inc_pool_weights.as_device_ptr(),
                d_dec_pool_vals.as_device_ptr(),
                d_dec_pool_weights.as_device_ptr(),
                d_sma_buf.as_device_ptr(),
                d_out_index.as_device_ptr(),
                d_out_cumulative.as_device_ptr(),
                d_out_upper.as_device_ptr()
            ))?;
        }

        let _ = first;
        Ok(CudaMonotonicityIndexBatchResult {
            outputs: MonotonicityIndexDeviceArrayF64Triplet {
                index: MonotonicityIndexDeviceArrayF64 {
                    buf: d_out_index,
                    rows,
                    cols,
                },
                cumulative_mean: MonotonicityIndexDeviceArrayF64 {
                    buf: d_out_cumulative,
                    rows,
                    cols,
                },
                upper_bound: MonotonicityIndexDeviceArrayF64 {
                    buf: d_out_upper,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
