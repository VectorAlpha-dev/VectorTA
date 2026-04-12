#![cfg(feature = "cuda")]

use crate::indicators::forward_backward_exponential_oscillator::{
    ForwardBackwardExponentialOscillatorBatchRange, ForwardBackwardExponentialOscillatorParams,
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

const FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 20;
const DEFAULT_SMOOTH: usize = 10;

#[derive(Debug, Error)]
pub enum CudaForwardBackwardExponentialOscillatorError {
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

pub struct ForwardBackwardExponentialOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl ForwardBackwardExponentialOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct ForwardBackwardExponentialOscillatorDeviceArrayF64Triple {
    pub forward_backward: ForwardBackwardExponentialOscillatorDeviceArrayF64,
    pub backward: ForwardBackwardExponentialOscillatorDeviceArrayF64,
    pub histogram: ForwardBackwardExponentialOscillatorDeviceArrayF64,
}

impl ForwardBackwardExponentialOscillatorDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.forward_backward.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.forward_backward.cols
    }
}

pub struct CudaForwardBackwardExponentialOscillatorBatchResult {
    pub outputs: ForwardBackwardExponentialOscillatorDeviceArrayF64Triple,
    pub combos: Vec<ForwardBackwardExponentialOscillatorParams>,
}

pub struct CudaForwardBackwardExponentialOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaForwardBackwardExponentialOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaForwardBackwardExponentialOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module =
            crate::load_cuda_embedded_module!("forward_backward_exponential_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaForwardBackwardExponentialOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_usize(
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Vec<usize>, CudaForwardBackwardExponentialOscillatorError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start <= end {
            let mut current = start;
            while current <= end {
                out.push(current);
                match current.checked_add(step) {
                    Some(next) => current = next,
                    None => break,
                }
            }
        } else {
            let mut current = start;
            while current >= end {
                out.push(current);
                match current.checked_sub(step) {
                    Some(next) => current = next,
                    None => break,
                }
                if current < end {
                    break;
                }
            }
        }

        if out.is_empty() {
            return Err(CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                format!("invalid range: start={start}, end={end}, step={step}"),
            ));
        }
        Ok(out)
    }

    fn expand_grid(
        range: &ForwardBackwardExponentialOscillatorBatchRange,
    ) -> Result<
        Vec<ForwardBackwardExponentialOscillatorParams>,
        CudaForwardBackwardExponentialOscillatorError,
    > {
        let lengths = Self::axis_usize(range.length.0, range.length.1, range.length.2)?;
        let smooths = Self::axis_usize(range.smooth.0, range.smooth.1, range.smooth.2)?;
        let total = lengths.len().checked_mul(smooths.len()).ok_or_else(|| {
            CudaForwardBackwardExponentialOscillatorError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}",
                range.length.0, range.length.1, range.length.2
            ))
        })?;

        let mut out = Vec::with_capacity(total);
        for length in lengths {
            for &smooth in &smooths {
                out.push(ForwardBackwardExponentialOscillatorParams {
                    length: Some(length),
                    smooth: Some(smooth),
                });
            }
        }
        Ok(out)
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaForwardBackwardExponentialOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaForwardBackwardExponentialOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaForwardBackwardExponentialOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(
                CudaForwardBackwardExponentialOscillatorError::LaunchConfigTooLarge {
                    gx: grid.x,
                    gy: grid.y,
                    gz: grid.z,
                    bx: block.x,
                    by: block.y,
                    bz: block.z,
                },
            );
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        data: &[f64],
        range: &ForwardBackwardExponentialOscillatorBatchRange,
    ) -> Result<
        CudaForwardBackwardExponentialOscillatorBatchResult,
        CudaForwardBackwardExponentialOscillatorError,
    > {
        if data.is_empty() {
            return Err(CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = Self::expand_grid(range)?;
        let first = data
            .iter()
            .position(|value| value.is_finite())
            .ok_or_else(|| {
                CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                    "all values are NaN".into(),
                )
            })?;
        let valid = data[first..]
            .iter()
            .filter(|value| value.is_finite())
            .count();
        let mut max_length = 0usize;
        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            let smooth = combo.smooth.unwrap_or(DEFAULT_SMOOTH);
            if length == 0 || length > data.len() {
                return Err(CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                    format!(
                        "invalid length: length = {length}, data length = {}",
                        data.len()
                    ),
                ));
            }
            if smooth == 0 {
                return Err(CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                    format!(
                        "invalid smooth: smooth = {smooth}, data length = {}",
                        data.len()
                    ),
                ));
            }
            let needed = length.max(2);
            if valid < needed {
                return Err(CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                    format!("not enough valid data: needed = {needed}, valid = {valid}"),
                ));
            }
            max_length = max_length.max(length);
        }

        let rows = combos.len();
        let cols = data.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as i32)
            .collect();
        let smooths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.smooth.unwrap_or(DEFAULT_SMOOTH) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                smooths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaForwardBackwardExponentialOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let scratch_elems = rows.checked_mul(max_length).ok_or_else(|| {
            CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                "rows*max_length overflow".into(),
            )
        })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                    "scratch bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaForwardBackwardExponentialOscillatorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_smooths = DeviceBuffer::from_slice(&smooths)?;
        let d_ema1_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_diff_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(scratch_elems)? };
        let d_out_forward_backward = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_backward = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_histogram = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("forward_backward_exponential_oscillator_batch_f64")
            .map_err(
                |_| CudaForwardBackwardExponentialOscillatorError::MissingKernelSymbol {
                    name: "forward_backward_exponential_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_BLOCK_X - 1)
            / FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_smooths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                d_ema1_buffer.as_device_ptr(),
                d_diff_buffer.as_device_ptr(),
                d_out_forward_backward.as_device_ptr(),
                d_out_backward.as_device_ptr(),
                d_out_histogram.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaForwardBackwardExponentialOscillatorBatchResult {
            outputs: ForwardBackwardExponentialOscillatorDeviceArrayF64Triple {
                forward_backward: ForwardBackwardExponentialOscillatorDeviceArrayF64 {
                    buf: d_out_forward_backward,
                    rows,
                    cols,
                },
                backward: ForwardBackwardExponentialOscillatorDeviceArrayF64 {
                    buf: d_out_backward,
                    rows,
                    cols,
                },
                histogram: ForwardBackwardExponentialOscillatorDeviceArrayF64 {
                    buf: d_out_histogram,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
