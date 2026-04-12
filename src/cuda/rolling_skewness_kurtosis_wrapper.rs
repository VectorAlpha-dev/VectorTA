#![cfg(feature = "cuda")]

use crate::indicators::rolling_skewness_kurtosis::{
    expand_grid_rolling_skewness_kurtosis, RollingSkewnessKurtosisBatchRange,
    RollingSkewnessKurtosisParams,
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

const ROLLING_SKEWNESS_KURTOSIS_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaRollingSkewnessKurtosisError {
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

pub struct RollingSkewnessKurtosisDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl RollingSkewnessKurtosisDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct RollingSkewnessKurtosisDeviceArrayF64Pair {
    pub skewness: RollingSkewnessKurtosisDeviceArrayF64,
    pub kurtosis: RollingSkewnessKurtosisDeviceArrayF64,
}

impl RollingSkewnessKurtosisDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.skewness.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.skewness.cols
    }
}

pub struct CudaRollingSkewnessKurtosisBatchResult {
    pub outputs: RollingSkewnessKurtosisDeviceArrayF64Pair,
    pub combos: Vec<RollingSkewnessKurtosisParams>,
}

pub struct CudaRollingSkewnessKurtosis {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaRollingSkewnessKurtosis {
    pub fn new(device_id: usize) -> Result<Self, CudaRollingSkewnessKurtosisError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("rolling_skewness_kurtosis_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaRollingSkewnessKurtosisError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn longest_valid_run(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for &value in data {
            if value.is_finite() {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        best
    }

    fn warmup_needed(length: usize, smooth_length: usize) -> usize {
        length.saturating_add(smooth_length).saturating_sub(1)
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaRollingSkewnessKurtosisError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaRollingSkewnessKurtosisError::OutOfMemory {
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
    ) -> Result<(), CudaRollingSkewnessKurtosisError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaRollingSkewnessKurtosisError::LaunchConfigTooLarge {
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
        sweep: &RollingSkewnessKurtosisBatchRange,
    ) -> Result<CudaRollingSkewnessKurtosisBatchResult, CudaRollingSkewnessKurtosisError> {
        if data.is_empty() {
            return Err(CudaRollingSkewnessKurtosisError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = expand_grid_rolling_skewness_kurtosis(sweep);
        if combos.is_empty() {
            return Err(CudaRollingSkewnessKurtosisError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let longest = Self::longest_valid_run(data);
        if longest == 0 {
            return Err(CudaRollingSkewnessKurtosisError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let mut max_length = 0usize;
        let mut max_smooth_length = 0usize;
        let mut max_needed = 0usize;
        let mut lengths = Vec::with_capacity(rows);
        let mut smooth_lengths = Vec::with_capacity(rows);

        for combo in &combos {
            let length = combo.length.unwrap_or(50);
            let smooth_length = combo.smooth_length.unwrap_or(3);
            if length == 0 || smooth_length == 0 {
                return Err(CudaRollingSkewnessKurtosisError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            max_length = max_length.max(length);
            max_smooth_length = max_smooth_length.max(smooth_length);
            max_needed = max_needed.max(Self::warmup_needed(length, smooth_length));
            lengths.push(length as i32);
            smooth_lengths.push(smooth_length as i32);
        }

        if longest < max_needed {
            return Err(CudaRollingSkewnessKurtosisError::InvalidInput(format!(
                "not enough valid data: needed={max_needed}, valid={longest}"
            )));
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaRollingSkewnessKurtosisError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                smooth_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaRollingSkewnessKurtosisError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaRollingSkewnessKurtosisError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaRollingSkewnessKurtosisError::InvalidInput("output bytes overflow".into())
            })?;
        let source_scratch = rows.checked_mul(max_length).ok_or_else(|| {
            CudaRollingSkewnessKurtosisError::InvalidInput("source scratch overflow".into())
        })?;
        let smooth_scratch = rows.checked_mul(max_smooth_length).ok_or_else(|| {
            CudaRollingSkewnessKurtosisError::InvalidInput("smooth scratch overflow".into())
        })?;
        let scratch_bytes = source_scratch
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                smooth_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                smooth_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaRollingSkewnessKurtosisError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaRollingSkewnessKurtosisError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_smooth_lengths = DeviceBuffer::from_slice(&smooth_lengths)?;
        let d_source_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(source_scratch)? };
        let d_skew_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(smooth_scratch)? };
        let d_kurt_buffer = unsafe { DeviceBuffer::<f64>::uninitialized(smooth_scratch)? };
        let d_out_skewness = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_kurtosis = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("rolling_skewness_kurtosis_batch_f64")
            .map_err(|_| CudaRollingSkewnessKurtosisError::MissingKernelSymbol {
                name: "rolling_skewness_kurtosis_batch_f64",
            })?;
        let grid_x = ((rows as u32) + ROLLING_SKEWNESS_KURTOSIS_BLOCK_X - 1)
            / ROLLING_SKEWNESS_KURTOSIS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ROLLING_SKEWNESS_KURTOSIS_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_smooth_lengths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                max_smooth_length as i32,
                d_source_buffer.as_device_ptr(),
                d_skew_buffer.as_device_ptr(),
                d_kurt_buffer.as_device_ptr(),
                d_out_skewness.as_device_ptr(),
                d_out_kurtosis.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaRollingSkewnessKurtosisBatchResult {
            outputs: RollingSkewnessKurtosisDeviceArrayF64Pair {
                skewness: RollingSkewnessKurtosisDeviceArrayF64 {
                    buf: d_out_skewness,
                    rows,
                    cols,
                },
                kurtosis: RollingSkewnessKurtosisDeviceArrayF64 {
                    buf: d_out_kurtosis,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
