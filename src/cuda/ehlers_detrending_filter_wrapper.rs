#![cfg(feature = "cuda")]

use crate::indicators::ehlers_detrending_filter::{
    expand_grid_ehlers_detrending_filter, EhlersDetrendingFilterBatchRange,
    EhlersDetrendingFilterParams,
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

const EHLERS_DETRENDING_FILTER_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaEhlersDetrendingFilterError {
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

pub struct EhlersDetrendingFilterDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersDetrendingFilterDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct EhlersDetrendingFilterDeviceArrayF64Pair {
    pub edf: EhlersDetrendingFilterDeviceArrayF64,
    pub signal: EhlersDetrendingFilterDeviceArrayF64,
}

impl EhlersDetrendingFilterDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.edf.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.edf.cols
    }
}

pub struct CudaEhlersDetrendingFilterBatchResult {
    pub outputs: EhlersDetrendingFilterDeviceArrayF64Pair,
    pub combos: Vec<EhlersDetrendingFilterParams>,
}

pub struct CudaEhlersDetrendingFilter {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaEhlersDetrendingFilter {
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersDetrendingFilterError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("ehlers_detrending_filter_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaEhlersDetrendingFilterError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_source(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn max_consecutive_finite_from(data: &[f64], start: usize) -> usize {
        let mut best = 0usize;
        let mut run = 0usize;
        for &value in &data[start..] {
            if value.is_finite() {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        best
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaEhlersDetrendingFilterError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaEhlersDetrendingFilterError::OutOfMemory {
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
    ) -> Result<(), CudaEhlersDetrendingFilterError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaEhlersDetrendingFilterError::LaunchConfigTooLarge {
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
        sweep: &EhlersDetrendingFilterBatchRange,
    ) -> Result<CudaEhlersDetrendingFilterBatchResult, CudaEhlersDetrendingFilterError> {
        if data.is_empty() {
            return Err(CudaEhlersDetrendingFilterError::InvalidInput(
                "empty input".into(),
            ));
        }

        let first = Self::first_valid_source(data).ok_or_else(|| {
            CudaEhlersDetrendingFilterError::InvalidInput("all values are NaN".into())
        })?;
        let combos = expand_grid_ehlers_detrending_filter(sweep)
            .map_err(|err| CudaEhlersDetrendingFilterError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaEhlersDetrendingFilterError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let max_length = combos
            .iter()
            .map(|params| params.length.unwrap_or(10))
            .max()
            .unwrap_or(0);
        if max_length == 0 || max_length > data.len() {
            return Err(CudaEhlersDetrendingFilterError::InvalidInput(format!(
                "invalid length: length={max_length}, data_len={}",
                data.len()
            )));
        }
        let valid = Self::max_consecutive_finite_from(data, first);
        if valid < max_length {
            return Err(CudaEhlersDetrendingFilterError::InvalidInput(format!(
                "not enough valid data: needed={max_length}, valid={valid}"
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let lengths: Vec<i32> = combos
            .iter()
            .map(|params| params.length.unwrap_or(10) as i32)
            .collect();

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaEhlersDetrendingFilterError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaEhlersDetrendingFilterError::InvalidInput("parameter bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaEhlersDetrendingFilterError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaEhlersDetrendingFilterError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEhlersDetrendingFilterError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_out_edf = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("ehlers_detrending_filter_batch_f64")
            .map_err(|_| CudaEhlersDetrendingFilterError::MissingKernelSymbol {
                name: "ehlers_detrending_filter_batch_f64",
            })?;
        let grid_x = ((rows as u32) + EHLERS_DETRENDING_FILTER_BLOCK_X - 1)
            / EHLERS_DETRENDING_FILTER_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EHLERS_DETRENDING_FILTER_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                d_out_edf.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaEhlersDetrendingFilterBatchResult {
            outputs: EhlersDetrendingFilterDeviceArrayF64Pair {
                edf: EhlersDetrendingFilterDeviceArrayF64 {
                    buf: d_out_edf,
                    rows,
                    cols,
                },
                signal: EhlersDetrendingFilterDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
