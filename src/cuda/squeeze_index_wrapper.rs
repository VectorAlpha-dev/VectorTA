#![cfg(feature = "cuda")]

use crate::indicators::squeeze_index::{SqueezeIndexBatchRange, SqueezeIndexParams};
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

const SQUEEZE_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const FLOAT_TOL: f64 = 1e-12;
const DEFAULT_CONV: f64 = 50.0;
const DEFAULT_LENGTH: usize = 20;

#[derive(Debug, Error)]
pub enum CudaSqueezeIndexError {
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

pub struct SqueezeIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl SqueezeIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaSqueezeIndexBatchResult {
    pub outputs: SqueezeIndexDeviceArrayF64,
    pub combos: Vec<SqueezeIndexParams>,
}

pub struct CudaSqueezeIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaSqueezeIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaSqueezeIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("squeeze_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaSqueezeIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn count_valid_values(data: &[f64]) -> usize {
        data.iter().filter(|value| value.is_finite()).count()
    }

    fn expand_axis_usize(
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Vec<usize>, CudaSqueezeIndexError> {
        if start == end {
            return Ok(vec![start]);
        }
        if step == 0 {
            return Err(CudaSqueezeIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }

        let mut out = Vec::new();
        if start < end {
            let mut current = start;
            while current <= end {
                out.push(current);
                let next = current.saturating_add(step);
                if next == current {
                    break;
                }
                current = next;
            }
        } else {
            let mut current = start;
            while current >= end {
                out.push(current);
                let next = current.saturating_sub(step);
                if next == current {
                    break;
                }
                current = next;
            }
        }

        if out.is_empty() {
            return Err(CudaSqueezeIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_axis_f64(start: f64, end: f64, step: f64) -> Result<Vec<f64>, CudaSqueezeIndexError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(CudaSqueezeIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        if (start - end).abs() <= FLOAT_TOL {
            return Ok(vec![start]);
        }
        if step == 0.0 {
            return Err(CudaSqueezeIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }

        let mut out = Vec::new();
        if start < end {
            let mut current = start;
            while current <= end + FLOAT_TOL {
                out.push(current);
                current += step.abs();
            }
        } else {
            let mut current = start;
            while current >= end - FLOAT_TOL {
                out.push(current);
                current -= step.abs();
            }
        }

        if out.is_empty() {
            return Err(CudaSqueezeIndexError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &SqueezeIndexBatchRange,
    ) -> Result<Vec<SqueezeIndexParams>, CudaSqueezeIndexError> {
        let convs = Self::expand_axis_f64(sweep.conv.0, sweep.conv.1, sweep.conv.2)?;
        let lengths = Self::expand_axis_usize(sweep.length.0, sweep.length.1, sweep.length.2)?;

        let mut combos = Vec::with_capacity(convs.len().saturating_mul(lengths.len()));
        for &conv in &convs {
            if !conv.is_finite() || conv <= 1.0 {
                return Err(CudaSqueezeIndexError::InvalidInput(format!(
                    "invalid conv: {conv}"
                )));
            }
            for &length in &lengths {
                if length == 0 {
                    return Err(CudaSqueezeIndexError::InvalidInput(
                        "invalid length: 0".into(),
                    ));
                }
                combos.push(SqueezeIndexParams {
                    conv: Some(conv),
                    length: Some(length),
                });
            }
        }
        Ok(combos)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaSqueezeIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaSqueezeIndexError::OutOfMemory {
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
    ) -> Result<(), CudaSqueezeIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaSqueezeIndexError::LaunchConfigTooLarge {
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
        sweep: &SqueezeIndexBatchRange,
    ) -> Result<CudaSqueezeIndexBatchResult, CudaSqueezeIndexError> {
        if data.is_empty() {
            return Err(CudaSqueezeIndexError::InvalidInput("empty input".into()));
        }
        Self::first_valid_value(data)
            .ok_or_else(|| CudaSqueezeIndexError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaSqueezeIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let valid = Self::count_valid_values(data);
        let mut convs = Vec::with_capacity(rows);
        let mut lengths = Vec::with_capacity(rows);
        let mut max_length = 0usize;

        for combo in &combos {
            let conv = combo.conv.unwrap_or(DEFAULT_CONV);
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            if !conv.is_finite() || conv <= 1.0 {
                return Err(CudaSqueezeIndexError::InvalidInput(format!(
                    "invalid conv: {conv}"
                )));
            }
            if length == 0 {
                return Err(CudaSqueezeIndexError::InvalidInput(
                    "invalid length: 0".into(),
                ));
            }
            max_length = max_length.max(length);
            convs.push(conv);
            lengths.push(length as i32);
        }

        if valid < max_length {
            return Err(CudaSqueezeIndexError::InvalidInput(format!(
                "not enough valid data: needed={max_length}, valid={valid}"
            )));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaSqueezeIndexError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| CudaSqueezeIndexError::InvalidInput("params bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaSqueezeIndexError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaSqueezeIndexError::InvalidInput("output bytes overflow".into()))?;
        let scratch_elems = rows
            .checked_mul(max_length)
            .ok_or_else(|| CudaSqueezeIndexError::InvalidInput("scratch overflow".into()))?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                scratch_elems
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| CudaSqueezeIndexError::InvalidInput("scratch bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| CudaSqueezeIndexError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_convs = DeviceBuffer::from_slice(&convs)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_ring_vals = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_ring_valid = unsafe { DeviceBuffer::<i32>::uninitialized(scratch_elems)? };
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("squeeze_index_batch_f64")
            .map_err(|_| CudaSqueezeIndexError::MissingKernelSymbol {
                name: "squeeze_index_batch_f64",
            })?;
        let grid_x = ((rows as u32) + SQUEEZE_INDEX_BLOCK_X - 1) / SQUEEZE_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(SQUEEZE_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_convs.as_device_ptr(),
                d_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                d_ring_vals.as_device_ptr(),
                d_ring_valid.as_device_ptr(),
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaSqueezeIndexBatchResult {
            outputs: SqueezeIndexDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
