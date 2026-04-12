#![cfg(feature = "cuda")]

use crate::indicators::spearman_correlation::{
    expand_grid, SpearmanCorrelationBatchRange, SpearmanCorrelationParams,
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

const SPEARMAN_CORRELATION_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaSpearmanCorrelationError {
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

pub struct SpearmanCorrelationDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl SpearmanCorrelationDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct SpearmanCorrelationDeviceArrayF64Pair {
    pub raw: SpearmanCorrelationDeviceArrayF64,
    pub smoothed: SpearmanCorrelationDeviceArrayF64,
}

impl SpearmanCorrelationDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.raw.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.raw.cols
    }
}

pub struct CudaSpearmanCorrelationBatchResult {
    pub outputs: SpearmanCorrelationDeviceArrayF64Pair,
    pub combos: Vec<SpearmanCorrelationParams>,
}

pub struct CudaSpearmanCorrelation {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaSpearmanCorrelation {
    pub fn new(device_id: usize) -> Result<Self, CudaSpearmanCorrelationError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("spearman_correlation_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaSpearmanCorrelationError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_pair(main: &[f64], compare: &[f64]) -> Option<usize> {
        (0..main.len()).find(|&i| main[i].is_finite() && compare[i].is_finite())
    }

    fn first_valid_return_idx(main: &[f64], compare: &[f64]) -> Option<usize> {
        (1..main.len()).find(|&i| {
            main[i - 1].is_finite()
                && main[i].is_finite()
                && compare[i - 1].is_finite()
                && compare[i].is_finite()
        })
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaSpearmanCorrelationError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaSpearmanCorrelationError::OutOfMemory {
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
    ) -> Result<(), CudaSpearmanCorrelationError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaSpearmanCorrelationError::LaunchConfigTooLarge {
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
        main: &[f64],
        compare: &[f64],
        sweep: &SpearmanCorrelationBatchRange,
    ) -> Result<CudaSpearmanCorrelationBatchResult, CudaSpearmanCorrelationError> {
        if main.is_empty() || compare.is_empty() {
            return Err(CudaSpearmanCorrelationError::InvalidInput(
                "empty input".into(),
            ));
        }
        if main.len() != compare.len() {
            return Err(CudaSpearmanCorrelationError::InvalidInput(format!(
                "input length mismatch: main={}, compare={}",
                main.len(),
                compare.len()
            )));
        }

        let combos = expand_grid(sweep)
            .map_err(|err| CudaSpearmanCorrelationError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaSpearmanCorrelationError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let _ = Self::first_valid_pair(main, compare).ok_or_else(|| {
            CudaSpearmanCorrelationError::InvalidInput("all values are NaN".into())
        })?;
        let first_return = Self::first_valid_return_idx(main, compare).ok_or_else(|| {
            CudaSpearmanCorrelationError::InvalidInput(
                "not enough valid data: needed=2, valid=0".into(),
            )
        })?;

        let rows = combos.len();
        let cols = main.len();
        let mut lookbacks = Vec::with_capacity(rows);
        let mut smoothing_lengths = Vec::with_capacity(rows);
        for combo in &combos {
            let lookback = combo.lookback.unwrap_or(12);
            let smoothing_length = combo.smoothing_length.unwrap_or(4);
            if lookback == 0 || lookback >= cols {
                return Err(CudaSpearmanCorrelationError::InvalidInput(format!(
                    "invalid lookback: lookback={lookback}, data_len={cols}"
                )));
            }
            if smoothing_length == 0 {
                return Err(CudaSpearmanCorrelationError::InvalidInput(format!(
                    "invalid smoothing_length: smoothing_length={smoothing_length}"
                )));
            }
            let valid = cols.saturating_sub(first_return);
            if valid < lookback {
                return Err(CudaSpearmanCorrelationError::InvalidInput(format!(
                    "not enough valid data: needed={lookback}, valid={valid}"
                )));
            }
            lookbacks.push(lookback as i32);
            smoothing_lengths.push(smoothing_length as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaSpearmanCorrelationError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lookbacks
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                smoothing_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaSpearmanCorrelationError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaSpearmanCorrelationError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaSpearmanCorrelationError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaSpearmanCorrelationError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_main = DeviceBuffer::from_slice(main)?;
        let d_compare = DeviceBuffer::from_slice(compare)?;
        let d_lookbacks = DeviceBuffer::from_slice(&lookbacks)?;
        let d_smoothing_lengths = DeviceBuffer::from_slice(&smoothing_lengths)?;
        let d_out_raw = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_smoothed = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("spearman_correlation_batch_f64")
            .map_err(|_| CudaSpearmanCorrelationError::MissingKernelSymbol {
                name: "spearman_correlation_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + SPEARMAN_CORRELATION_BLOCK_X - 1) / SPEARMAN_CORRELATION_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(SPEARMAN_CORRELATION_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_main.as_device_ptr(),
                d_compare.as_device_ptr(),
                cols as i32,
                d_lookbacks.as_device_ptr(),
                d_smoothing_lengths.as_device_ptr(),
                rows as i32,
                d_out_raw.as_device_ptr(),
                d_out_smoothed.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaSpearmanCorrelationBatchResult {
            outputs: SpearmanCorrelationDeviceArrayF64Pair {
                raw: SpearmanCorrelationDeviceArrayF64 {
                    buf: d_out_raw,
                    rows,
                    cols,
                },
                smoothed: SpearmanCorrelationDeviceArrayF64 {
                    buf: d_out_smoothed,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
