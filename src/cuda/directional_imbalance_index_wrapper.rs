#![cfg(feature = "cuda")]

use crate::indicators::directional_imbalance_index::{
    expand_grid_directional_imbalance_index, DirectionalImbalanceIndexBatchRange,
    DirectionalImbalanceIndexParams,
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

const DIRECTIONAL_IMBALANCE_INDEX_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaDirectionalImbalanceIndexError {
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

pub struct DirectionalImbalanceIndexDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DirectionalImbalanceIndexDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct DirectionalImbalanceIndexDeviceArrayF64Six {
    pub up: DirectionalImbalanceIndexDeviceArrayF64,
    pub down: DirectionalImbalanceIndexDeviceArrayF64,
    pub bulls: DirectionalImbalanceIndexDeviceArrayF64,
    pub bears: DirectionalImbalanceIndexDeviceArrayF64,
    pub upper: DirectionalImbalanceIndexDeviceArrayF64,
    pub lower: DirectionalImbalanceIndexDeviceArrayF64,
}

impl DirectionalImbalanceIndexDeviceArrayF64Six {
    #[inline]
    pub fn rows(&self) -> usize {
        self.up.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.up.cols
    }
}

pub struct CudaDirectionalImbalanceIndexBatchResult {
    pub outputs: DirectionalImbalanceIndexDeviceArrayF64Six,
    pub combos: Vec<DirectionalImbalanceIndexParams>,
}

pub struct CudaDirectionalImbalanceIndex {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaDirectionalImbalanceIndex {
    pub fn new(device_id: usize) -> Result<Self, CudaDirectionalImbalanceIndexError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("directional_imbalance_index_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaDirectionalImbalanceIndexError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn any_valid_pair(high: &[f64], low: &[f64]) -> bool {
        high.iter()
            .zip(low.iter())
            .any(|(&h, &l)| h.is_finite() && l.is_finite())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaDirectionalImbalanceIndexError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaDirectionalImbalanceIndexError::OutOfMemory {
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
    ) -> Result<(), CudaDirectionalImbalanceIndexError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaDirectionalImbalanceIndexError::LaunchConfigTooLarge {
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
        high: &[f64],
        low: &[f64],
        sweep: &DirectionalImbalanceIndexBatchRange,
        fixed: &DirectionalImbalanceIndexParams,
    ) -> Result<CudaDirectionalImbalanceIndexBatchResult, CudaDirectionalImbalanceIndexError> {
        let len = high.len();
        if len == 0 || low.is_empty() {
            return Err(CudaDirectionalImbalanceIndexError::InvalidInput(
                "empty input".into(),
            ));
        }
        if low.len() != len {
            return Err(CudaDirectionalImbalanceIndexError::InvalidInput(format!(
                "input length mismatch: high={len}, low={}",
                low.len()
            )));
        }
        if !Self::any_valid_pair(high, low) {
            return Err(CudaDirectionalImbalanceIndexError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_directional_imbalance_index(sweep, fixed)
            .map_err(|err| CudaDirectionalImbalanceIndexError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaDirectionalImbalanceIndexError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = len;
        let mut max_window = 0usize;
        let mut max_period = 0usize;
        let mut lengths = Vec::with_capacity(rows);
        let mut periods = Vec::with_capacity(rows);

        for combo in &combos {
            let length = combo.length.unwrap_or(10);
            let period = combo.period.unwrap_or(70);
            if length == 0 || period == 0 {
                return Err(CudaDirectionalImbalanceIndexError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            max_window = max_window.max(length.saturating_add(1));
            max_period = max_period.max(period);
            lengths.push(length as i32);
            periods.push(period as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaDirectionalImbalanceIndexError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                periods
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaDirectionalImbalanceIndexError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaDirectionalImbalanceIndexError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(6))
            .ok_or_else(|| {
                CudaDirectionalImbalanceIndexError::InvalidInput("output bytes overflow".into())
            })?;
        let length_scratch = rows.checked_mul(max_window).ok_or_else(|| {
            CudaDirectionalImbalanceIndexError::InvalidInput("length scratch overflow".into())
        })?;
        let period_scratch = rows.checked_mul(max_period).ok_or_else(|| {
            CudaDirectionalImbalanceIndexError::InvalidInput("period scratch overflow".into())
        })?;
        let scratch_bytes = length_scratch
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                length_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                period_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                period_scratch
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaDirectionalImbalanceIndexError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaDirectionalImbalanceIndexError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_high_ring = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_low_ring = unsafe { DeviceBuffer::<f64>::uninitialized(length_scratch)? };
        let d_up_hits = unsafe { DeviceBuffer::<f64>::uninitialized(period_scratch)? };
        let d_down_hits = unsafe { DeviceBuffer::<f64>::uninitialized(period_scratch)? };
        let d_out_up = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_down = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bulls = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bears = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("directional_imbalance_index_batch_f64")
            .map_err(
                |_| CudaDirectionalImbalanceIndexError::MissingKernelSymbol {
                    name: "directional_imbalance_index_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + DIRECTIONAL_IMBALANCE_INDEX_BLOCK_X - 1)
            / DIRECTIONAL_IMBALANCE_INDEX_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(DIRECTIONAL_IMBALANCE_INDEX_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_periods.as_device_ptr(),
                rows as i32,
                max_window as i32,
                max_period as i32,
                d_high_ring.as_device_ptr(),
                d_low_ring.as_device_ptr(),
                d_up_hits.as_device_ptr(),
                d_down_hits.as_device_ptr(),
                d_out_up.as_device_ptr(),
                d_out_down.as_device_ptr(),
                d_out_bulls.as_device_ptr(),
                d_out_bears.as_device_ptr(),
                d_out_upper.as_device_ptr(),
                d_out_lower.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaDirectionalImbalanceIndexBatchResult {
            outputs: DirectionalImbalanceIndexDeviceArrayF64Six {
                up: DirectionalImbalanceIndexDeviceArrayF64 {
                    buf: d_out_up,
                    rows,
                    cols,
                },
                down: DirectionalImbalanceIndexDeviceArrayF64 {
                    buf: d_out_down,
                    rows,
                    cols,
                },
                bulls: DirectionalImbalanceIndexDeviceArrayF64 {
                    buf: d_out_bulls,
                    rows,
                    cols,
                },
                bears: DirectionalImbalanceIndexDeviceArrayF64 {
                    buf: d_out_bears,
                    rows,
                    cols,
                },
                upper: DirectionalImbalanceIndexDeviceArrayF64 {
                    buf: d_out_upper,
                    rows,
                    cols,
                },
                lower: DirectionalImbalanceIndexDeviceArrayF64 {
                    buf: d_out_lower,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
