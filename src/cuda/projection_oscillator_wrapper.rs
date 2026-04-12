#![cfg(feature = "cuda")]

use crate::indicators::projection_oscillator::{
    expand_grid_projection_oscillator, ProjectionOscillatorBatchRange, ProjectionOscillatorParams,
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

const PROJECTION_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaProjectionOscillatorError {
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

pub struct ProjectionOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl ProjectionOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct ProjectionOscillatorDeviceArrayF64Pair {
    pub pbo: ProjectionOscillatorDeviceArrayF64,
    pub signal: ProjectionOscillatorDeviceArrayF64,
}

impl ProjectionOscillatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.pbo.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.pbo.cols
    }
}

pub struct CudaProjectionOscillatorBatchResult {
    pub outputs: ProjectionOscillatorDeviceArrayF64Pair,
    pub combos: Vec<ProjectionOscillatorParams>,
}

pub struct CudaProjectionOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaProjectionOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaProjectionOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("projection_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaProjectionOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn signal_needed_bars(length: usize, smooth_length: usize) -> usize {
        length
            .saturating_mul(2)
            .saturating_add(smooth_length.saturating_mul(2))
            .saturating_sub(3)
    }

    fn longest_valid_run(high: &[f64], low: &[f64], source: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for ((&h, &l), &s) in high.iter().zip(low.iter()).zip(source.iter()) {
            if h.is_finite() && l.is_finite() && s.is_finite() {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        best
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaProjectionOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaProjectionOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaProjectionOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaProjectionOscillatorError::LaunchConfigTooLarge {
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
        source: &[f64],
        sweep: &ProjectionOscillatorBatchRange,
    ) -> Result<CudaProjectionOscillatorBatchResult, CudaProjectionOscillatorError> {
        let len = high.len();
        if len == 0 || low.is_empty() || source.is_empty() {
            return Err(CudaProjectionOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if low.len() != len || source.len() != len {
            return Err(CudaProjectionOscillatorError::InvalidInput(format!(
                "input length mismatch: high={len}, low={}, source={}",
                low.len(),
                source.len()
            )));
        }

        let combos = expand_grid_projection_oscillator(sweep)
            .map_err(|err| CudaProjectionOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaProjectionOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let longest = Self::longest_valid_run(high, low, source);
        if longest == 0 {
            return Err(CudaProjectionOscillatorError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let rows = combos.len();
        let cols = len;
        let mut max_length = 0usize;
        let mut max_smooth_length = 0usize;
        let mut max_needed = 0usize;
        let mut lengths = Vec::with_capacity(rows);
        let mut smooth_lengths = Vec::with_capacity(rows);

        for combo in &combos {
            let length = combo.length.unwrap_or(14);
            let smooth_length = combo.smooth_length.unwrap_or(4);
            if length == 0 || smooth_length == 0 {
                return Err(CudaProjectionOscillatorError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            max_length = max_length.max(length);
            max_smooth_length = max_smooth_length.max(smooth_length);
            max_needed = max_needed.max(Self::signal_needed_bars(length, smooth_length));
            lengths.push(length as i32);
            smooth_lengths.push(smooth_length as i32);
        }

        if longest < max_needed {
            return Err(CudaProjectionOscillatorError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={longest}"
            )));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaProjectionOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                smooth_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaProjectionOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaProjectionOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaProjectionOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let length_scratch = rows.checked_mul(max_length).ok_or_else(|| {
            CudaProjectionOscillatorError::InvalidInput("length scratch overflow".into())
        })?;
        let smooth_scratch = rows.checked_mul(max_smooth_length).ok_or_else(|| {
            CudaProjectionOscillatorError::InvalidInput("smooth scratch overflow".into())
        })?;
        let scratch_bytes = length_scratch
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                length_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                length_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                length_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                smooth_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                smooth_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaProjectionOscillatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaProjectionOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_source = DeviceBuffer::from_slice(source)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_smooth_lengths = DeviceBuffer::from_slice(&smooth_lengths)?;
        let d_high_window = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_low_window = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_high_slopes = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_low_slopes = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_pbo_window = unsafe { DeviceBuffer::<f64>::uninitialized(smooth_scratch)? };
        let d_signal_window = unsafe { DeviceBuffer::<f64>::uninitialized(smooth_scratch)? };
        let d_out_pbo = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("projection_oscillator_batch_f64")
            .map_err(|_| CudaProjectionOscillatorError::MissingKernelSymbol {
                name: "projection_oscillator_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + PROJECTION_OSCILLATOR_BLOCK_X - 1) / PROJECTION_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(PROJECTION_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_source.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_smooth_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                max_smooth_length as i32,
                d_high_window.as_device_ptr(),
                d_low_window.as_device_ptr(),
                d_high_slopes.as_device_ptr(),
                d_low_slopes.as_device_ptr(),
                d_pbo_window.as_device_ptr(),
                d_signal_window.as_device_ptr(),
                d_out_pbo.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaProjectionOscillatorBatchResult {
            outputs: ProjectionOscillatorDeviceArrayF64Pair {
                pbo: ProjectionOscillatorDeviceArrayF64 {
                    buf: d_out_pbo,
                    rows,
                    cols,
                },
                signal: ProjectionOscillatorDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
