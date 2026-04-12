#![cfg(feature = "cuda")]

use crate::indicators::l2_ehlers_signal_to_noise::{
    expand_grid, L2EhlersSignalToNoiseBatchRange, L2EhlersSignalToNoiseParams,
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

const L2_EHLERS_SIGNAL_TO_NOISE_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const MIN_REQUIRED_VALID: usize = 7;

#[derive(Debug, Error)]
pub enum CudaL2EhlersSignalToNoiseError {
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

pub struct L2EhlersSignalToNoiseDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl L2EhlersSignalToNoiseDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaL2EhlersSignalToNoiseBatchResult {
    pub outputs: L2EhlersSignalToNoiseDeviceArrayF64,
    pub combos: Vec<L2EhlersSignalToNoiseParams>,
}

pub struct CudaL2EhlersSignalToNoise {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaL2EhlersSignalToNoise {
    pub fn new(device_id: usize) -> Result<Self, CudaL2EhlersSignalToNoiseError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("l2_ehlers_signal_to_noise_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaL2EhlersSignalToNoiseError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_triple(source: &[f64], high: &[f64], low: &[f64]) -> Option<usize> {
        (0..source.len())
            .find(|&i| source[i].is_finite() && high[i].is_finite() && low[i].is_finite())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaL2EhlersSignalToNoiseError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaL2EhlersSignalToNoiseError::OutOfMemory {
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
    ) -> Result<(), CudaL2EhlersSignalToNoiseError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaL2EhlersSignalToNoiseError::LaunchConfigTooLarge {
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
        source: &[f64],
        high: &[f64],
        low: &[f64],
        sweep: &L2EhlersSignalToNoiseBatchRange,
    ) -> Result<CudaL2EhlersSignalToNoiseBatchResult, CudaL2EhlersSignalToNoiseError> {
        let len = source.len();
        if len == 0 || high.is_empty() || low.is_empty() {
            return Err(CudaL2EhlersSignalToNoiseError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != len || low.len() != len {
            return Err(CudaL2EhlersSignalToNoiseError::InvalidInput(format!(
                "input length mismatch: source={len}, high={}, low={}",
                high.len(),
                low.len()
            )));
        }

        let first = Self::first_valid_triple(source, high, low).ok_or_else(|| {
            CudaL2EhlersSignalToNoiseError::InvalidInput("all values are NaN".into())
        })?;
        let valid = len.saturating_sub(first);
        if valid < MIN_REQUIRED_VALID {
            return Err(CudaL2EhlersSignalToNoiseError::InvalidInput(format!(
                "not enough valid data: needed={MIN_REQUIRED_VALID}, valid={valid}"
            )));
        }

        let combos = expand_grid(sweep)
            .map_err(|err| CudaL2EhlersSignalToNoiseError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaL2EhlersSignalToNoiseError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = len;
        let mut smooth_periods = Vec::with_capacity(rows);
        for combo in &combos {
            let smooth_period = combo.smooth_period.unwrap_or(10);
            if smooth_period == 0 {
                return Err(CudaL2EhlersSignalToNoiseError::InvalidInput(
                    "invalid smooth_period: 0".into(),
                ));
            }
            smooth_periods.push(smooth_period as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaL2EhlersSignalToNoiseError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaL2EhlersSignalToNoiseError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaL2EhlersSignalToNoiseError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaL2EhlersSignalToNoiseError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaL2EhlersSignalToNoiseError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_source = DeviceBuffer::from_slice(source)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_smooth_periods = DeviceBuffer::from_slice(&smooth_periods)?;
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("l2_ehlers_signal_to_noise_batch_f64")
            .map_err(|_| CudaL2EhlersSignalToNoiseError::MissingKernelSymbol {
                name: "l2_ehlers_signal_to_noise_batch_f64",
            })?;
        let grid_x = ((rows as u32) + L2_EHLERS_SIGNAL_TO_NOISE_BLOCK_X - 1)
            / L2_EHLERS_SIGNAL_TO_NOISE_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(L2_EHLERS_SIGNAL_TO_NOISE_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_source.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                cols as i32,
                d_smooth_periods.as_device_ptr(),
                rows as i32,
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaL2EhlersSignalToNoiseBatchResult {
            outputs: L2EhlersSignalToNoiseDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
