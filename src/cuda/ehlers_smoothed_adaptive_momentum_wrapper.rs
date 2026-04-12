#![cfg(feature = "cuda")]

use crate::indicators::ehlers_smoothed_adaptive_momentum::{
    expand_grid, EhlersSmoothedAdaptiveMomentumBatchRange, EhlersSmoothedAdaptiveMomentumParams,
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

const EHLERS_SMOOTHED_ADAPTIVE_MOMENTUM_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_ALPHA: f64 = 0.07;
const DEFAULT_CUTOFF: f64 = 8.0;
const REQUIRED_VALID_SAMPLES: usize = 76;

#[derive(Debug, Error)]
pub enum CudaEhlersSmoothedAdaptiveMomentumError {
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

pub struct EhlersSmoothedAdaptiveMomentumDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersSmoothedAdaptiveMomentumDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaEhlersSmoothedAdaptiveMomentumBatchResult {
    pub outputs: EhlersSmoothedAdaptiveMomentumDeviceArrayF64,
    pub combos: Vec<EhlersSmoothedAdaptiveMomentumParams>,
}

pub struct CudaEhlersSmoothedAdaptiveMomentum {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaEhlersSmoothedAdaptiveMomentum {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersSmoothedAdaptiveMomentumError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ehlers_smoothed_adaptive_momentum_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEhlersSmoothedAdaptiveMomentumError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaEhlersSmoothedAdaptiveMomentumError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEhlersSmoothedAdaptiveMomentumError::OutOfMemory {
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
    ) -> Result<(), CudaEhlersSmoothedAdaptiveMomentumError> {
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
                CudaEhlersSmoothedAdaptiveMomentumError::LaunchConfigTooLarge {
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
        sweep: &EhlersSmoothedAdaptiveMomentumBatchRange,
    ) -> Result<
        CudaEhlersSmoothedAdaptiveMomentumBatchResult,
        CudaEhlersSmoothedAdaptiveMomentumError,
    > {
        if data.is_empty() {
            return Err(CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput(
                "empty input".into(),
            ));
        }
        let first = Self::first_valid_value(data).ok_or_else(|| {
            CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput("all values are NaN".into())
        })?;
        let valid = data.len().saturating_sub(first);
        if valid < REQUIRED_VALID_SAMPLES {
            return Err(CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput(
                format!("not enough valid data: needed={REQUIRED_VALID_SAMPLES}, valid={valid}"),
            ));
        }

        let combos = expand_grid(sweep).map_err(|err| {
            CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput(err.to_string())
        })?;
        if combos.is_empty() {
            return Err(CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut alphas = Vec::with_capacity(rows);
        let mut cutoffs = Vec::with_capacity(rows);
        for combo in &combos {
            let alpha = combo.alpha.unwrap_or(DEFAULT_ALPHA);
            let cutoff = combo.cutoff.unwrap_or(DEFAULT_CUTOFF);
            if !alpha.is_finite() || alpha < 0.0 {
                return Err(CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput(
                    format!("invalid alpha: {alpha}"),
                ));
            }
            if !cutoff.is_finite() || cutoff <= 0.0 {
                return Err(CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput(
                    format!("invalid cutoff: {cutoff}"),
                ));
            }
            alphas.push(alpha);
            cutoffs.push(cutoff);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEhlersSmoothedAdaptiveMomentumError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_alphas = DeviceBuffer::from_slice(&alphas)?;
        let d_cutoffs = DeviceBuffer::from_slice(&cutoffs)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("ehlers_smoothed_adaptive_momentum_batch_f64")
            .map_err(
                |_| CudaEhlersSmoothedAdaptiveMomentumError::MissingKernelSymbol {
                    name: "ehlers_smoothed_adaptive_momentum_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + EHLERS_SMOOTHED_ADAPTIVE_MOMENTUM_BLOCK_X - 1)
            / EHLERS_SMOOTHED_ADAPTIVE_MOMENTUM_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EHLERS_SMOOTHED_ADAPTIVE_MOMENTUM_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_alphas.as_device_ptr(),
                d_cutoffs.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaEhlersSmoothedAdaptiveMomentumBatchResult {
            outputs: EhlersSmoothedAdaptiveMomentumDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
