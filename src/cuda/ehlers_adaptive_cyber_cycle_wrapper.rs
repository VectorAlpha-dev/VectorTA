#![cfg(feature = "cuda")]

use crate::indicators::ehlers_adaptive_cyber_cycle::{
    expand_grid, EhlersAdaptiveCyberCycleBatchRange, EhlersAdaptiveCyberCycleParams,
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

const EHLERS_ADAPTIVE_CYBER_CYCLE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_ALPHA: f64 = 0.07;
const REQUIRED_VALID_SAMPLES: usize = 3;

#[derive(Debug, Error)]
pub enum CudaEhlersAdaptiveCyberCycleError {
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

pub struct EhlersAdaptiveCyberCycleDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersAdaptiveCyberCycleDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct EhlersAdaptiveCyberCycleDeviceArrayF64Pair {
    pub cycle: EhlersAdaptiveCyberCycleDeviceArrayF64,
    pub trigger: EhlersAdaptiveCyberCycleDeviceArrayF64,
}

impl EhlersAdaptiveCyberCycleDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.cycle.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cycle.cols
    }
}

pub struct CudaEhlersAdaptiveCyberCycleBatchResult {
    pub outputs: EhlersAdaptiveCyberCycleDeviceArrayF64Pair,
    pub combos: Vec<EhlersAdaptiveCyberCycleParams>,
}

pub struct CudaEhlersAdaptiveCyberCycle {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaEhlersAdaptiveCyberCycle {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersAdaptiveCyberCycleError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ehlers_adaptive_cyber_cycle_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEhlersAdaptiveCyberCycleError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaEhlersAdaptiveCyberCycleError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEhlersAdaptiveCyberCycleError::OutOfMemory {
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
    ) -> Result<(), CudaEhlersAdaptiveCyberCycleError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaEhlersAdaptiveCyberCycleError::LaunchConfigTooLarge {
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
        sweep: &EhlersAdaptiveCyberCycleBatchRange,
    ) -> Result<CudaEhlersAdaptiveCyberCycleBatchResult, CudaEhlersAdaptiveCyberCycleError> {
        if data.is_empty() {
            return Err(CudaEhlersAdaptiveCyberCycleError::InvalidInput(
                "empty input".into(),
            ));
        }
        let first = Self::first_valid_value(data).ok_or_else(|| {
            CudaEhlersAdaptiveCyberCycleError::InvalidInput("all values are NaN".into())
        })?;

        let combos = expand_grid(sweep)
            .map_err(|err| CudaEhlersAdaptiveCyberCycleError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaEhlersAdaptiveCyberCycleError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let valid = cols.saturating_sub(first);
        let mut alphas = Vec::with_capacity(rows);
        for combo in &combos {
            let alpha = combo.alpha.unwrap_or(DEFAULT_ALPHA);
            if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                return Err(CudaEhlersAdaptiveCyberCycleError::InvalidInput(format!(
                    "invalid alpha: {alpha}"
                )));
            }
            if valid < REQUIRED_VALID_SAMPLES {
                return Err(CudaEhlersAdaptiveCyberCycleError::InvalidInput(format!(
                    "not enough valid data: needed={}, valid={valid}",
                    REQUIRED_VALID_SAMPLES
                )));
            }
            alphas.push(alpha);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersAdaptiveCyberCycleError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersAdaptiveCyberCycleError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaEhlersAdaptiveCyberCycleError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaEhlersAdaptiveCyberCycleError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEhlersAdaptiveCyberCycleError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_alphas = DeviceBuffer::from_slice(&alphas)?;
        let d_out_cycle = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_trigger = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("ehlers_adaptive_cyber_cycle_batch_f64")
            .map_err(|_| CudaEhlersAdaptiveCyberCycleError::MissingKernelSymbol {
                name: "ehlers_adaptive_cyber_cycle_batch_f64",
            })?;
        let grid_x = ((rows as u32) + EHLERS_ADAPTIVE_CYBER_CYCLE_BLOCK_X - 1)
            / EHLERS_ADAPTIVE_CYBER_CYCLE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EHLERS_ADAPTIVE_CYBER_CYCLE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_alphas.as_device_ptr(),
                rows as i32,
                d_out_cycle.as_device_ptr(),
                d_out_trigger.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaEhlersAdaptiveCyberCycleBatchResult {
            outputs: EhlersAdaptiveCyberCycleDeviceArrayF64Pair {
                cycle: EhlersAdaptiveCyberCycleDeviceArrayF64 {
                    buf: d_out_cycle,
                    rows,
                    cols,
                },
                trigger: EhlersAdaptiveCyberCycleDeviceArrayF64 {
                    buf: d_out_trigger,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
