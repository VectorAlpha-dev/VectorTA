#![cfg(feature = "cuda")]

use crate::indicators::adaptive_bounds_rsi::{
    expand_grid, AdaptiveBoundsRsiBatchRange, AdaptiveBoundsRsiParams,
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

const ADAPTIVE_BOUNDS_RSI_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_ALPHA: f64 = 0.1;
const MIN_ALPHA: f64 = 0.001;
const MAX_ALPHA: f64 = 1.0;

#[derive(Debug, Error)]
pub enum CudaAdaptiveBoundsRsiError {
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

pub struct AdaptiveBoundsRsiDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AdaptiveBoundsRsiDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct AdaptiveBoundsRsiDeviceArrayF64Deca {
    pub rsi: AdaptiveBoundsRsiDeviceArrayF64,
    pub lower_bound: AdaptiveBoundsRsiDeviceArrayF64,
    pub lower_mid: AdaptiveBoundsRsiDeviceArrayF64,
    pub mid: AdaptiveBoundsRsiDeviceArrayF64,
    pub upper_mid: AdaptiveBoundsRsiDeviceArrayF64,
    pub upper_bound: AdaptiveBoundsRsiDeviceArrayF64,
    pub regime: AdaptiveBoundsRsiDeviceArrayF64,
    pub regime_flip: AdaptiveBoundsRsiDeviceArrayF64,
    pub lower_signal: AdaptiveBoundsRsiDeviceArrayF64,
    pub upper_signal: AdaptiveBoundsRsiDeviceArrayF64,
}

impl AdaptiveBoundsRsiDeviceArrayF64Deca {
    #[inline]
    pub fn rows(&self) -> usize {
        self.rsi.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.rsi.cols
    }
}

pub struct CudaAdaptiveBoundsRsiBatchResult {
    pub outputs: AdaptiveBoundsRsiDeviceArrayF64Deca,
    pub combos: Vec<AdaptiveBoundsRsiParams>,
}

pub struct CudaAdaptiveBoundsRsi {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaAdaptiveBoundsRsi {
    pub fn new(device_id: usize) -> Result<Self, CudaAdaptiveBoundsRsiError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("adaptive_bounds_rsi_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAdaptiveBoundsRsiError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn first_valid_value(data: &[f64]) -> Option<usize> {
        data.iter().position(|value| value.is_finite())
    }

    fn valid_count_from(data: &[f64], first: usize) -> usize {
        data[first..]
            .iter()
            .filter(|value| value.is_finite())
            .count()
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaAdaptiveBoundsRsiError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAdaptiveBoundsRsiError::OutOfMemory {
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
    ) -> Result<(), CudaAdaptiveBoundsRsiError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaAdaptiveBoundsRsiError::LaunchConfigTooLarge {
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
        sweep: &AdaptiveBoundsRsiBatchRange,
    ) -> Result<CudaAdaptiveBoundsRsiBatchResult, CudaAdaptiveBoundsRsiError> {
        if data.is_empty() {
            return Err(CudaAdaptiveBoundsRsiError::InvalidInput(
                "empty input".into(),
            ));
        }
        let first = Self::first_valid_value(data)
            .ok_or_else(|| CudaAdaptiveBoundsRsiError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep)
            .map_err(|err| CudaAdaptiveBoundsRsiError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaAdaptiveBoundsRsiError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = data.len();
        let valid = Self::valid_count_from(data, first);
        let mut rsi_lengths = Vec::with_capacity(rows);
        let mut alphas = Vec::with_capacity(rows);

        for combo in &combos {
            let rsi_length = combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
            let alpha = combo.alpha.unwrap_or(DEFAULT_ALPHA);
            if rsi_length == 0 || rsi_length > cols {
                return Err(CudaAdaptiveBoundsRsiError::InvalidInput(format!(
                    "invalid rsi_length: {rsi_length}"
                )));
            }
            if !alpha.is_finite() || !(MIN_ALPHA..=MAX_ALPHA).contains(&alpha) {
                return Err(CudaAdaptiveBoundsRsiError::InvalidInput(format!(
                    "invalid alpha: {alpha}"
                )));
            }
            let needed = rsi_length + 1;
            if valid < needed {
                return Err(CudaAdaptiveBoundsRsiError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={valid}"
                )));
            }
            rsi_lengths.push(rsi_length as i32);
            alphas.push(alpha);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaAdaptiveBoundsRsiError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| value.checked_add(rows.checked_mul(std::mem::size_of::<f64>())?))
            .ok_or_else(|| {
                CudaAdaptiveBoundsRsiError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaAdaptiveBoundsRsiError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(10))
            .ok_or_else(|| {
                CudaAdaptiveBoundsRsiError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaAdaptiveBoundsRsiError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_rsi_lengths = DeviceBuffer::from_slice(&rsi_lengths)?;
        let d_alphas = DeviceBuffer::from_slice(&alphas)?;
        let d_out_rsi = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_lower_bound = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_lower_mid = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_mid = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_upper_mid = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_upper_bound = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_regime = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_regime_flip = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_lower_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_upper_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("adaptive_bounds_rsi_batch_f64")
            .map_err(|_| CudaAdaptiveBoundsRsiError::MissingKernelSymbol {
                name: "adaptive_bounds_rsi_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + ADAPTIVE_BOUNDS_RSI_BLOCK_X - 1) / ADAPTIVE_BOUNDS_RSI_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(ADAPTIVE_BOUNDS_RSI_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_rsi_lengths.as_device_ptr(),
                d_alphas.as_device_ptr(),
                rows as i32,
                d_out_rsi.as_device_ptr(),
                d_out_lower_bound.as_device_ptr(),
                d_out_lower_mid.as_device_ptr(),
                d_out_mid.as_device_ptr(),
                d_out_upper_mid.as_device_ptr(),
                d_out_upper_bound.as_device_ptr(),
                d_out_regime.as_device_ptr(),
                d_out_regime_flip.as_device_ptr(),
                d_out_lower_signal.as_device_ptr(),
                d_out_upper_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaAdaptiveBoundsRsiBatchResult {
            outputs: AdaptiveBoundsRsiDeviceArrayF64Deca {
                rsi: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_rsi,
                    rows,
                    cols,
                },
                lower_bound: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_lower_bound,
                    rows,
                    cols,
                },
                lower_mid: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_lower_mid,
                    rows,
                    cols,
                },
                mid: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_mid,
                    rows,
                    cols,
                },
                upper_mid: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_upper_mid,
                    rows,
                    cols,
                },
                upper_bound: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_upper_bound,
                    rows,
                    cols,
                },
                regime: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_regime,
                    rows,
                    cols,
                },
                regime_flip: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_regime_flip,
                    rows,
                    cols,
                },
                lower_signal: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_lower_signal,
                    rows,
                    cols,
                },
                upper_signal: AdaptiveBoundsRsiDeviceArrayF64 {
                    buf: d_out_upper_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
