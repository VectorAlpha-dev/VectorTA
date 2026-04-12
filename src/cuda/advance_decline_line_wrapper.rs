#![cfg(feature = "cuda")]

use crate::indicators::advance_decline_line::{
    AdvanceDeclineLineBatchRange, AdvanceDeclineLineParams,
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

const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaAdvanceDeclineLineError {
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

pub struct DeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaAdvanceDeclineLineBatchResult {
    pub outputs: DeviceArrayF64,
    pub combos: Vec<AdvanceDeclineLineParams>,
}

pub struct CudaAdvanceDeclineLine {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAdvanceDeclineLine {
    pub fn new(device_id: usize) -> Result<Self, CudaAdvanceDeclineLineError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("advance_decline_line_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAdvanceDeclineLineError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn expand_grid(
        _sweep: &AdvanceDeclineLineBatchRange,
    ) -> Result<Vec<AdvanceDeclineLineParams>, CudaAdvanceDeclineLineError> {
        Ok(vec![AdvanceDeclineLineParams])
    }

    fn first_valid_index(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaAdvanceDeclineLineError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAdvanceDeclineLineError::OutOfMemory {
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
    ) -> Result<(), CudaAdvanceDeclineLineError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaAdvanceDeclineLineError::LaunchConfigTooLarge {
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
        sweep: &AdvanceDeclineLineBatchRange,
    ) -> Result<CudaAdvanceDeclineLineBatchResult, CudaAdvanceDeclineLineError> {
        if data.is_empty() {
            return Err(CudaAdvanceDeclineLineError::InvalidInput(
                "empty data".into(),
            ));
        }
        Self::first_valid_index(data).ok_or_else(|| {
            CudaAdvanceDeclineLineError::InvalidInput("all values are NaN".into())
        })?;

        let combos = Self::expand_grid(sweep)?;
        let rows = combos.len();
        let cols = data.len();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaAdvanceDeclineLineError::InvalidInput("input bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaAdvanceDeclineLineError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaAdvanceDeclineLineError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes.checked_add(output_bytes).ok_or_else(|| {
            CudaAdvanceDeclineLineError::InvalidInput("required bytes overflow".into())
        })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let mut d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("advance_decline_line_batch_f64")
            .map_err(|_| CudaAdvanceDeclineLineError::MissingKernelSymbol {
                name: "advance_decline_line_batch_f64",
            })?;
        let grid = GridSize::x(1);
        let block = BlockSize::x(1);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaAdvanceDeclineLineBatchResult {
            outputs: DeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
