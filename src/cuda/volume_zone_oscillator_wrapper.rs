#![cfg(feature = "cuda")]

use crate::indicators::volume_zone_oscillator::{
    expand_grid, VolumeZoneOscillatorBatchRange, VolumeZoneOscillatorParams,
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

const VOLUME_ZONE_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 14;
const DEFAULT_NOISE_FILTER: usize = 4;

#[derive(Debug, Error)]
pub enum CudaVolumeZoneOscillatorError {
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

pub struct VolumeZoneOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl VolumeZoneOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaVolumeZoneOscillatorBatchResult {
    pub outputs: VolumeZoneOscillatorDeviceArrayF64,
    pub combos: Vec<VolumeZoneOscillatorParams>,
}

pub struct CudaVolumeZoneOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaVolumeZoneOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaVolumeZoneOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("volume_zone_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaVolumeZoneOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaVolumeZoneOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaVolumeZoneOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaVolumeZoneOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaVolumeZoneOscillatorError::LaunchConfigTooLarge {
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
        close: &[f64],
        volume: &[f64],
        sweep: &VolumeZoneOscillatorBatchRange,
    ) -> Result<CudaVolumeZoneOscillatorBatchResult, CudaVolumeZoneOscillatorError> {
        if close.is_empty() || volume.is_empty() {
            return Err(CudaVolumeZoneOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if close.len() != volume.len() {
            return Err(CudaVolumeZoneOscillatorError::InvalidInput(format!(
                "input length mismatch: close={}, volume={}",
                close.len(),
                volume.len()
            )));
        }

        volume
            .iter()
            .position(|value| value.is_finite())
            .ok_or_else(|| {
                CudaVolumeZoneOscillatorError::InvalidInput("all values are NaN".into())
            })?;

        let combos = expand_grid(sweep)
            .map_err(|err| CudaVolumeZoneOscillatorError::InvalidInput(err.to_string()))?;
        let rows = combos.len();
        let cols = close.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .map(|length| {
                if length < 2 {
                    Err(CudaVolumeZoneOscillatorError::InvalidInput(format!(
                        "invalid length: {length}"
                    )))
                } else {
                    Ok(length as i32)
                }
            })
            .collect::<Result<_, _>>()?;
        let noise_filters: Vec<i32> = combos
            .iter()
            .map(|combo| combo.noise_filter.unwrap_or(DEFAULT_NOISE_FILTER))
            .map(|noise_filter| {
                if noise_filter < 2 {
                    Err(CudaVolumeZoneOscillatorError::InvalidInput(format!(
                        "invalid noise_filter: {noise_filter}"
                    )))
                } else {
                    Ok(noise_filter as i32)
                }
            })
            .collect::<Result<_, _>>()?;
        let intraday_flags: Vec<i32> = combos
            .iter()
            .map(|combo| i32::from(combo.intraday_smoothing.unwrap_or(true)))
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaVolumeZoneOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                noise_filters
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                intraday_flags
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaVolumeZoneOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaVolumeZoneOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaVolumeZoneOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaVolumeZoneOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_noise_filters = DeviceBuffer::from_slice(&noise_filters)?;
        let d_intraday_flags = DeviceBuffer::from_slice(&intraday_flags)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("volume_zone_oscillator_batch_f64")
            .map_err(|_| CudaVolumeZoneOscillatorError::MissingKernelSymbol {
                name: "volume_zone_oscillator_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + VOLUME_ZONE_OSCILLATOR_BLOCK_X - 1) / VOLUME_ZONE_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(VOLUME_ZONE_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_noise_filters.as_device_ptr(),
                d_intraday_flags.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaVolumeZoneOscillatorBatchResult {
            outputs: VolumeZoneOscillatorDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
