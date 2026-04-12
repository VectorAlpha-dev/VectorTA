#![cfg(feature = "cuda")]

use crate::indicators::stochastic_adaptive_d::{
    expand_grid_stochastic_adaptive_d, StochasticAdaptiveDBatchRange, StochasticAdaptiveDParams,
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

const STOCHASTIC_ADAPTIVE_D_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_K_LENGTH: usize = 20;
const DEFAULT_D_SMOOTHING: usize = 9;
const DEFAULT_PRE_SMOOTH: usize = 20;
const DEFAULT_ATTENUATION: f64 = 2.0;

#[derive(Debug, Error)]
pub enum CudaStochasticAdaptiveDError {
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

pub struct StochasticAdaptiveDDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl StochasticAdaptiveDDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct StochasticAdaptiveDDeviceArrayF64Triple {
    pub standard_d: StochasticAdaptiveDDeviceArrayF64,
    pub adaptive_d: StochasticAdaptiveDDeviceArrayF64,
    pub difference: StochasticAdaptiveDDeviceArrayF64,
}

impl StochasticAdaptiveDDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.standard_d.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.standard_d.cols
    }
}

pub struct CudaStochasticAdaptiveDBatchResult {
    pub outputs: StochasticAdaptiveDDeviceArrayF64Triple,
    pub combos: Vec<StochasticAdaptiveDParams>,
}

pub struct CudaStochasticAdaptiveD {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaStochasticAdaptiveD {
    pub fn new(device_id: usize) -> Result<Self, CudaStochasticAdaptiveDError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("stochastic_adaptive_d_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaStochasticAdaptiveDError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
        (0..close.len()).find(|&i| {
            high[i].is_finite() && low[i].is_finite() && close[i].is_finite() && high[i] >= low[i]
        })
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaStochasticAdaptiveDError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaStochasticAdaptiveDError::OutOfMemory {
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
    ) -> Result<(), CudaStochasticAdaptiveDError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaStochasticAdaptiveDError::LaunchConfigTooLarge {
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
        close: &[f64],
        sweep: &StochasticAdaptiveDBatchRange,
    ) -> Result<CudaStochasticAdaptiveDBatchResult, CudaStochasticAdaptiveDError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaStochasticAdaptiveDError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || low.len() != close.len() {
            return Err(CudaStochasticAdaptiveDError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let combos = expand_grid_stochastic_adaptive_d(sweep)
            .map_err(|err| CudaStochasticAdaptiveDError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaStochasticAdaptiveDError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let first_valid = Self::first_valid_bar(high, low, close).ok_or_else(|| {
            CudaStochasticAdaptiveDError::InvalidInput("all values are NaN".into())
        })?;
        let valid = close.len() - first_valid;
        let rows = combos.len();
        let cols = close.len();
        let mut k_lengths = Vec::with_capacity(rows);
        let mut d_smoothings = Vec::with_capacity(rows);
        let mut pre_smooths = Vec::with_capacity(rows);
        let mut attenuations = Vec::with_capacity(rows);

        for combo in &combos {
            let k_length = combo.k_length.unwrap_or(DEFAULT_K_LENGTH);
            let d_smoothing = combo.d_smoothing.unwrap_or(DEFAULT_D_SMOOTHING);
            let pre_smooth = combo.pre_smooth.unwrap_or(DEFAULT_PRE_SMOOTH);
            let attenuation = combo.attenuation.unwrap_or(DEFAULT_ATTENUATION);
            if k_length == 0
                || k_length > cols
                || d_smoothing == 0
                || d_smoothing > cols
                || pre_smooth == 0
                || pre_smooth > cols
                || !attenuation.is_finite()
                || attenuation < 0.1
            {
                return Err(CudaStochasticAdaptiveDError::InvalidInput(
                    "invalid parameters".into(),
                ));
            }
            let needed = pre_smooth
                .checked_add(k_length)
                .and_then(|v| v.checked_add(d_smoothing))
                .and_then(|v| v.checked_sub(2))
                .ok_or_else(|| {
                    CudaStochasticAdaptiveDError::InvalidInput("needed bars overflow".into())
                })?;
            if valid < needed {
                return Err(CudaStochasticAdaptiveDError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            k_lengths.push(k_length as i32);
            d_smoothings.push(d_smoothing as i32);
            pre_smooths.push(pre_smooth as i32);
            attenuations.push(attenuation);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| {
                CudaStochasticAdaptiveDError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| {
                rows.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| v.checked_add(other))
            })
            .ok_or_else(|| {
                CudaStochasticAdaptiveDError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaStochasticAdaptiveDError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| {
                CudaStochasticAdaptiveDError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaStochasticAdaptiveDError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_k_lengths = DeviceBuffer::from_slice(&k_lengths)?;
        let d_d_smoothings = DeviceBuffer::from_slice(&d_smoothings)?;
        let d_pre_smooths = DeviceBuffer::from_slice(&pre_smooths)?;
        let d_attenuations = DeviceBuffer::from_slice(&attenuations)?;
        let d_out_standard = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_adaptive = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_difference = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("stochastic_adaptive_d_batch_f64")
            .map_err(|_| CudaStochasticAdaptiveDError::MissingKernelSymbol {
                name: "stochastic_adaptive_d_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + STOCHASTIC_ADAPTIVE_D_BLOCK_X - 1) / STOCHASTIC_ADAPTIVE_D_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(STOCHASTIC_ADAPTIVE_D_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_k_lengths.as_device_ptr(),
                d_d_smoothings.as_device_ptr(),
                d_pre_smooths.as_device_ptr(),
                d_attenuations.as_device_ptr(),
                rows as i32,
                d_out_standard.as_device_ptr(),
                d_out_adaptive.as_device_ptr(),
                d_out_difference.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaStochasticAdaptiveDBatchResult {
            outputs: StochasticAdaptiveDDeviceArrayF64Triple {
                standard_d: StochasticAdaptiveDDeviceArrayF64 {
                    buf: d_out_standard,
                    rows,
                    cols,
                },
                adaptive_d: StochasticAdaptiveDDeviceArrayF64 {
                    buf: d_out_adaptive,
                    rows,
                    cols,
                },
                difference: StochasticAdaptiveDDeviceArrayF64 {
                    buf: d_out_difference,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
