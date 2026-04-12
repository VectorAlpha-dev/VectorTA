#![cfg(feature = "cuda")]

use crate::indicators::ehlers_adaptive_cg::{EhlersAdaptiveCgBatchRange, EhlersAdaptiveCgParams};
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

const EHLERS_ADAPTIVE_CG_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaEhlersAdaptiveCgError {
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

pub struct EhlersAdaptiveCgDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersAdaptiveCgDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct EhlersAdaptiveCgDeviceArrayF64Pair {
    pub cg: EhlersAdaptiveCgDeviceArrayF64,
    pub trigger: EhlersAdaptiveCgDeviceArrayF64,
}

impl EhlersAdaptiveCgDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.cg.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cg.cols
    }
}

pub struct CudaEhlersAdaptiveCgBatchResult {
    pub outputs: EhlersAdaptiveCgDeviceArrayF64Pair,
    pub combos: Vec<EhlersAdaptiveCgParams>,
}

pub struct CudaEhlersAdaptiveCg {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaEhlersAdaptiveCg {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersAdaptiveCgError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ehlers_adaptive_cg_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEhlersAdaptiveCgError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, CudaEhlersAdaptiveCgError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(CudaEhlersAdaptiveCgError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let step_abs = step.abs();
        let mut values = Vec::new();
        if start < end {
            let mut current = start;
            while current <= end + 1e-12 {
                values.push(current);
                current += step_abs;
            }
        } else {
            let mut current = start;
            while current + 1e-12 >= end {
                values.push(current);
                current -= step_abs;
            }
        }
        if values.is_empty() {
            return Err(CudaEhlersAdaptiveCgError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        Ok(values)
    }

    fn expand_grid(
        sweep: &EhlersAdaptiveCgBatchRange,
    ) -> Result<Vec<EhlersAdaptiveCgParams>, CudaEhlersAdaptiveCgError> {
        Self::axis_f64(sweep.alpha)?
            .into_iter()
            .map(|alpha| {
                if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
                    return Err(CudaEhlersAdaptiveCgError::InvalidInput(format!(
                        "invalid alpha: {alpha}"
                    )));
                }
                Ok(EhlersAdaptiveCgParams { alpha: Some(alpha) })
            })
            .collect()
    }

    fn first_valid_index(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| !value.is_nan())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaEhlersAdaptiveCgError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEhlersAdaptiveCgError::OutOfMemory {
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
    ) -> Result<(), CudaEhlersAdaptiveCgError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaEhlersAdaptiveCgError::LaunchConfigTooLarge {
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
        sweep: &EhlersAdaptiveCgBatchRange,
    ) -> Result<CudaEhlersAdaptiveCgBatchResult, CudaEhlersAdaptiveCgError> {
        if data.is_empty() {
            return Err(CudaEhlersAdaptiveCgError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaEhlersAdaptiveCgError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let first_valid = Self::first_valid_index(data)
            .ok_or_else(|| CudaEhlersAdaptiveCgError::InvalidInput("all values are NaN".into()))?;
        let valid = data.len().saturating_sub(first_valid);
        if valid < 14 {
            return Err(CudaEhlersAdaptiveCgError::InvalidInput(format!(
                "not enough valid data: needed=14, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let alphas: Vec<f64> = combos
            .iter()
            .map(|params| params.alpha.unwrap_or(0.07))
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersAdaptiveCgError::InvalidInput("input bytes overflow".into())
            })?;
        let alpha_bytes = alphas
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersAdaptiveCgError::InvalidInput("alpha bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaEhlersAdaptiveCgError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaEhlersAdaptiveCgError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(alpha_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEhlersAdaptiveCgError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_alphas = DeviceBuffer::from_slice(&alphas)?;
        let d_out_cg = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_trigger = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("ehlers_adaptive_cg_batch_f64")
            .map_err(|_| CudaEhlersAdaptiveCgError::MissingKernelSymbol {
                name: "ehlers_adaptive_cg_batch_f64",
            })?;
        let grid_x = ((rows as u32) + EHLERS_ADAPTIVE_CG_BLOCK_X - 1) / EHLERS_ADAPTIVE_CG_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EHLERS_ADAPTIVE_CG_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_alphas.as_device_ptr(),
                rows as i32,
                d_out_cg.as_device_ptr(),
                d_out_trigger.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaEhlersAdaptiveCgBatchResult {
            outputs: EhlersAdaptiveCgDeviceArrayF64Pair {
                cg: EhlersAdaptiveCgDeviceArrayF64 {
                    buf: d_out_cg,
                    rows,
                    cols,
                },
                trigger: EhlersAdaptiveCgDeviceArrayF64 {
                    buf: d_out_trigger,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
