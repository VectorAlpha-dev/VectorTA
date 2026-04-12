#![cfg(feature = "cuda")]

use crate::indicators::l1_ehlers_phasor::{L1EhlersPhasorBatchRange, L1EhlersPhasorParams};
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

const L1_EHLERS_PHASOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_DOMESTIC_CYCLE_LENGTH: usize = 15;

#[derive(Debug, Error)]
pub enum CudaL1EhlersPhasorError {
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

pub struct L1EhlersPhasorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl L1EhlersPhasorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaL1EhlersPhasorBatchResult {
    pub outputs: L1EhlersPhasorDeviceArrayF64,
    pub combos: Vec<L1EhlersPhasorParams>,
}

pub struct CudaL1EhlersPhasor {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaL1EhlersPhasor {
    pub fn new(device_id: usize) -> Result<Self, CudaL1EhlersPhasorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("l1_ehlers_phasor_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaL1EhlersPhasorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn expand_grid(
        sweep: &L1EhlersPhasorBatchRange,
    ) -> Result<Vec<L1EhlersPhasorParams>, CudaL1EhlersPhasorError> {
        let (start, end, step) = sweep.domestic_cycle_length;
        if start == 0 {
            return Err(CudaL1EhlersPhasorError::InvalidInput(format!(
                "invalid range: start={start}, end={end}, step={step}"
            )));
        }

        let mut values = Vec::new();
        if step == 0 {
            if start != end {
                return Err(CudaL1EhlersPhasorError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            values.push(start);
        } else {
            if start > end {
                return Err(CudaL1EhlersPhasorError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                )));
            }
            let mut current = start;
            while current <= end {
                values.push(current);
                current = match current.checked_add(step) {
                    Some(next) => next,
                    None => break,
                };
            }
        }

        Ok(values
            .into_iter()
            .map(|domestic_cycle_length| L1EhlersPhasorParams {
                domestic_cycle_length: Some(domestic_cycle_length),
            })
            .collect())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaL1EhlersPhasorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaL1EhlersPhasorError::OutOfMemory {
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
    ) -> Result<(), CudaL1EhlersPhasorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaL1EhlersPhasorError::LaunchConfigTooLarge {
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
        sweep: &L1EhlersPhasorBatchRange,
    ) -> Result<CudaL1EhlersPhasorBatchResult, CudaL1EhlersPhasorError> {
        if data.is_empty() {
            return Err(CudaL1EhlersPhasorError::InvalidInput("empty input".into()));
        }

        let combos = Self::expand_grid(sweep)?;
        let first = Self::first_valid(data)
            .ok_or_else(|| CudaL1EhlersPhasorError::InvalidInput("all values are NaN".into()))?;
        let rows = combos.len();
        let cols = data.len();
        let max_length = combos
            .iter()
            .map(|combo| {
                combo
                    .domestic_cycle_length
                    .unwrap_or(DEFAULT_DOMESTIC_CYCLE_LENGTH)
            })
            .max()
            .unwrap_or(DEFAULT_DOMESTIC_CYCLE_LENGTH);
        let valid = cols.saturating_sub(first);
        if max_length == 0 || valid < max_length {
            return Err(CudaL1EhlersPhasorError::InvalidInput(format!(
                "not enough valid data: needed={max_length}, valid={valid}"
            )));
        }

        let lengths: Vec<i32> = combos
            .iter()
            .map(|combo| {
                combo
                    .domestic_cycle_length
                    .unwrap_or(DEFAULT_DOMESTIC_CYCLE_LENGTH) as i32
            })
            .collect();
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaL1EhlersPhasorError::InvalidInput("rows*cols overflow".into()))?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaL1EhlersPhasorError::InvalidInput("input bytes overflow".into()))?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaL1EhlersPhasorError::InvalidInput("params bytes overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaL1EhlersPhasorError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaL1EhlersPhasorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("l1_ehlers_phasor_batch_f64")
            .map_err(|_| CudaL1EhlersPhasorError::MissingKernelSymbol {
                name: "l1_ehlers_phasor_batch_f64",
            })?;
        let grid_x = ((rows as u32) + L1_EHLERS_PHASOR_BLOCK_X - 1) / L1_EHLERS_PHASOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(L1_EHLERS_PHASOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        Ok(CudaL1EhlersPhasorBatchResult {
            outputs: L1EhlersPhasorDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
