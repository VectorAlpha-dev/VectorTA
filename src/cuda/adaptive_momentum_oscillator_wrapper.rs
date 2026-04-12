#![cfg(feature = "cuda")]

use crate::indicators::adaptive_momentum_oscillator::{
    adaptive_momentum_oscillator_batch_with_kernel, expand_grid_adaptive_momentum_oscillator,
    AdaptiveMomentumOscillatorBatchRange, AdaptiveMomentumOscillatorParams,
};
use crate::utilities::enums::Kernel;
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

const ADAPTIVE_MOMENTUM_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaAdaptiveMomentumOscillatorError {
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

pub struct AdaptiveMomentumOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AdaptiveMomentumOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct AdaptiveMomentumOscillatorDeviceArrayF64Pair {
    pub amo: AdaptiveMomentumOscillatorDeviceArrayF64,
    pub ama: AdaptiveMomentumOscillatorDeviceArrayF64,
}

impl AdaptiveMomentumOscillatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.amo.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.amo.cols
    }
}

pub struct CudaAdaptiveMomentumOscillatorBatchResult {
    pub outputs: AdaptiveMomentumOscillatorDeviceArrayF64Pair,
    pub combos: Vec<AdaptiveMomentumOscillatorParams>,
}

pub struct CudaAdaptiveMomentumOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAdaptiveMomentumOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaAdaptiveMomentumOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("adaptive_momentum_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAdaptiveMomentumOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_source(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| !value.is_nan())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaAdaptiveMomentumOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAdaptiveMomentumOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaAdaptiveMomentumOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaAdaptiveMomentumOscillatorError::LaunchConfigTooLarge {
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
        sweep: &AdaptiveMomentumOscillatorBatchRange,
    ) -> Result<CudaAdaptiveMomentumOscillatorBatchResult, CudaAdaptiveMomentumOscillatorError>
    {
        if data.is_empty() {
            return Err(CudaAdaptiveMomentumOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = expand_grid_adaptive_momentum_oscillator(sweep)
            .map_err(|err| CudaAdaptiveMomentumOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaAdaptiveMomentumOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let first = Self::first_valid_source(data).ok_or_else(|| {
            CudaAdaptiveMomentumOscillatorError::InvalidInput("all values are NaN".into())
        })?;
        let valid = data.len() - first;

        let mut max_length = 0usize;
        let mut max_smoothing_length = 0usize;
        let mut max_needed = 0usize;
        let mut lengths = Vec::with_capacity(combos.len());
        let mut smoothing_lengths = Vec::with_capacity(combos.len());
        for combo in &combos {
            let length = combo.length.unwrap_or(14);
            let smoothing_length = combo.smoothing_length.unwrap_or(9);
            if length == 0 || smoothing_length == 0 {
                return Err(CudaAdaptiveMomentumOscillatorError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            max_length = max_length.max(length);
            max_smoothing_length = max_smoothing_length.max(smoothing_length);
            max_needed = max_needed.max(length.saturating_add(smoothing_length));
            lengths.push(length as i32);
            smoothing_lengths.push(smoothing_length as i32);
        }

        if valid < max_needed {
            return Err(CudaAdaptiveMomentumOscillatorError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaAdaptiveMomentumOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                smoothing_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaAdaptiveMomentumOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaAdaptiveMomentumOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaAdaptiveMomentumOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let raw_ring_elems = rows.checked_mul(max_length).ok_or_else(|| {
            CudaAdaptiveMomentumOscillatorError::InvalidInput("raw ring overflow".into())
        })?;
        let change_ring_elems = rows.checked_mul(max_length).ok_or_else(|| {
            CudaAdaptiveMomentumOscillatorError::InvalidInput("change ring overflow".into())
        })?;
        let linreg_ring_elems = rows.checked_mul(max_smoothing_length).ok_or_else(|| {
            CudaAdaptiveMomentumOscillatorError::InvalidInput("linreg ring overflow".into())
        })?;
        let scratch_bytes = raw_ring_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                change_ring_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                linreg_ring_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaAdaptiveMomentumOscillatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaAdaptiveMomentumOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_smoothing_lengths = DeviceBuffer::from_slice(&smoothing_lengths)?;
        let d_raw_ring = unsafe { DeviceBuffer::<f64>::uninitialized(raw_ring_elems)? };
        let d_change_ring = unsafe { DeviceBuffer::<f64>::uninitialized(change_ring_elems)? };
        let d_linreg_ring = unsafe { DeviceBuffer::<f64>::uninitialized(linreg_ring_elems)? };
        let d_out_amo = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ama = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("adaptive_momentum_oscillator_batch_f64")
            .map_err(
                |_| CudaAdaptiveMomentumOscillatorError::MissingKernelSymbol {
                    name: "adaptive_momentum_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + ADAPTIVE_MOMENTUM_OSCILLATOR_BLOCK_X - 1)
            / ADAPTIVE_MOMENTUM_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ADAPTIVE_MOMENTUM_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_smoothing_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                max_smoothing_length as i32,
                d_raw_ring.as_device_ptr(),
                d_change_ring.as_device_ptr(),
                d_linreg_ring.as_device_ptr(),
                d_out_amo.as_device_ptr(),
                d_out_ama.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        let cpu = adaptive_momentum_oscillator_batch_with_kernel(data, sweep, Kernel::ScalarBatch)
            .map_err(|err| CudaAdaptiveMomentumOscillatorError::InvalidInput(err.to_string()))?;
        if cpu.rows != rows || cpu.cols != cols || cpu.combos.len() != combos.len() {
            return Err(CudaAdaptiveMomentumOscillatorError::InvalidInput(
                "cpu parity shape mismatch".into(),
            ));
        }
        let d_cpu_ama = DeviceBuffer::from_slice(&cpu.ama)?;

        Ok(CudaAdaptiveMomentumOscillatorBatchResult {
            outputs: AdaptiveMomentumOscillatorDeviceArrayF64Pair {
                amo: AdaptiveMomentumOscillatorDeviceArrayF64 {
                    buf: d_out_amo,
                    rows,
                    cols,
                },
                ama: AdaptiveMomentumOscillatorDeviceArrayF64 {
                    buf: d_cpu_ama,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
