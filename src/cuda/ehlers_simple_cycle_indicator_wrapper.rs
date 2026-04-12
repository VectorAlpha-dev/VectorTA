#![cfg(feature = "cuda")]

use crate::indicators::ehlers_simple_cycle_indicator::{
    EhlersSimpleCycleIndicatorBatchRange, EhlersSimpleCycleIndicatorParams,
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

const EHLERS_SIMPLE_CYCLE_INDICATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaEhlersSimpleCycleIndicatorError {
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

pub struct EhlersSimpleCycleIndicatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersSimpleCycleIndicatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct EhlersSimpleCycleIndicatorDeviceArrayF64Pair {
    pub cycle: EhlersSimpleCycleIndicatorDeviceArrayF64,
    pub trigger: EhlersSimpleCycleIndicatorDeviceArrayF64,
}

pub struct CudaEhlersSimpleCycleIndicatorBatchResult {
    pub outputs: EhlersSimpleCycleIndicatorDeviceArrayF64Pair,
    pub combos: Vec<EhlersSimpleCycleIndicatorParams>,
}

pub struct CudaEhlersSimpleCycleIndicator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaEhlersSimpleCycleIndicator {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersSimpleCycleIndicatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ehlers_simple_cycle_indicator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEhlersSimpleCycleIndicatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn expand_float_range(
        start: f64,
        end: f64,
        step: f64,
    ) -> Result<Vec<f64>, CudaEhlersSimpleCycleIndicatorError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(CudaEhlersSimpleCycleIndicatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }
        if step == 0.0 {
            if (start - end).abs() > 1e-12 {
                return Err(CudaEhlersSimpleCycleIndicatorError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            return Ok(vec![start]);
        }
        if start > end || step < 0.0 {
            return Err(CudaEhlersSimpleCycleIndicatorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }

        let mut out = Vec::new();
        let mut current = start;
        while current <= end + 1e-12 {
            out.push(current);
            if out.len() > 1_000_000 {
                return Err(CudaEhlersSimpleCycleIndicatorError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            current += step;
        }
        Ok(out)
    }

    fn expand_grid(
        sweep: &EhlersSimpleCycleIndicatorBatchRange,
    ) -> Result<Vec<EhlersSimpleCycleIndicatorParams>, CudaEhlersSimpleCycleIndicatorError> {
        Ok(
            Self::expand_float_range(sweep.alpha.0, sweep.alpha.1, sweep.alpha.2)?
                .into_iter()
                .map(|alpha| EhlersSimpleCycleIndicatorParams { alpha: Some(alpha) })
                .collect(),
        )
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaEhlersSimpleCycleIndicatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEhlersSimpleCycleIndicatorError::OutOfMemory {
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
    ) -> Result<(), CudaEhlersSimpleCycleIndicatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaEhlersSimpleCycleIndicatorError::LaunchConfigTooLarge {
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
        sweep: &EhlersSimpleCycleIndicatorBatchRange,
    ) -> Result<CudaEhlersSimpleCycleIndicatorBatchResult, CudaEhlersSimpleCycleIndicatorError>
    {
        if data.is_empty() {
            return Err(CudaEhlersSimpleCycleIndicatorError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = Self::expand_grid(sweep)?;
        let first = Self::first_valid(data).ok_or_else(|| {
            CudaEhlersSimpleCycleIndicatorError::InvalidInput("all values are NaN".into())
        })?;
        let valid = data.len().saturating_sub(first);
        if valid < 3 {
            return Err(CudaEhlersSimpleCycleIndicatorError::InvalidInput(format!(
                "not enough valid data: needed=3, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let alphas: Vec<f64> = combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(0.07))
            .collect();
        if alphas
            .iter()
            .any(|alpha| !alpha.is_finite() || *alpha < 0.0 || *alpha > 1.0)
        {
            return Err(CudaEhlersSimpleCycleIndicatorError::InvalidInput(
                "invalid alpha".into(),
            ));
        }

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaEhlersSimpleCycleIndicatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersSimpleCycleIndicatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = alphas
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersSimpleCycleIndicatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaEhlersSimpleCycleIndicatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEhlersSimpleCycleIndicatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_alphas = DeviceBuffer::from_slice(&alphas)?;
        let d_cycle = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_trigger = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("ehlers_simple_cycle_indicator_batch_f64")
            .map_err(
                |_| CudaEhlersSimpleCycleIndicatorError::MissingKernelSymbol {
                    name: "ehlers_simple_cycle_indicator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + EHLERS_SIMPLE_CYCLE_INDICATOR_BLOCK_X - 1)
            / EHLERS_SIMPLE_CYCLE_INDICATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EHLERS_SIMPLE_CYCLE_INDICATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_alphas.as_device_ptr(),
                rows as i32,
                d_cycle.as_device_ptr(),
                d_trigger.as_device_ptr()
            ))?;
        }

        Ok(CudaEhlersSimpleCycleIndicatorBatchResult {
            outputs: EhlersSimpleCycleIndicatorDeviceArrayF64Pair {
                cycle: EhlersSimpleCycleIndicatorDeviceArrayF64 {
                    buf: d_cycle,
                    rows,
                    cols,
                },
                trigger: EhlersSimpleCycleIndicatorDeviceArrayF64 {
                    buf: d_trigger,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
