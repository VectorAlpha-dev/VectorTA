#![cfg(feature = "cuda")]

use crate::indicators::geometric_bias_oscillator::{
    expand_grid_geometric_bias_oscillator, GeometricBiasOscillatorBatchRange,
    GeometricBiasOscillatorParams,
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

const GEOMETRIC_BIAS_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaGeometricBiasOscillatorError {
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

pub struct GeometricBiasOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl GeometricBiasOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaGeometricBiasOscillatorBatchResult {
    pub outputs: GeometricBiasOscillatorDeviceArrayF64,
    pub combos: Vec<GeometricBiasOscillatorParams>,
}

pub struct CudaGeometricBiasOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaGeometricBiasOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaGeometricBiasOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("geometric_bias_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaGeometricBiasOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn analyze_valid_segments(
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<(usize, usize), CudaGeometricBiasOscillatorError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaGeometricBiasOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaGeometricBiasOscillatorError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let mut first_valid = None;
        let mut max_run = 0usize;
        let mut run = 0usize;
        for i in 0..close.len() {
            let valid = high[i].is_finite() && low[i].is_finite() && close[i].is_finite();
            if valid {
                if first_valid.is_none() {
                    first_valid = Some(i);
                }
                run += 1;
                max_run = max_run.max(run);
            } else {
                run = 0;
            }
        }

        match first_valid {
            Some(first) => Ok((first, max_run)),
            None => Err(CudaGeometricBiasOscillatorError::InvalidInput(
                "all values are NaN".into(),
            )),
        }
    }

    fn required_valid_bars(length: usize, atr_length: usize, smooth: usize) -> usize {
        length.max(atr_length) + smooth - 1
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaGeometricBiasOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaGeometricBiasOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaGeometricBiasOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaGeometricBiasOscillatorError::LaunchConfigTooLarge {
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
        sweep: &GeometricBiasOscillatorBatchRange,
    ) -> Result<CudaGeometricBiasOscillatorBatchResult, CudaGeometricBiasOscillatorError> {
        let (_, max_run) = Self::analyze_valid_segments(high, low, close)?;
        let combos = expand_grid_geometric_bias_oscillator(sweep)
            .map_err(|err| CudaGeometricBiasOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaGeometricBiasOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut max_length = 0usize;
        let mut max_smooth = 0usize;
        let mut lengths = Vec::with_capacity(rows);
        let mut multipliers = Vec::with_capacity(rows);
        let mut atr_lengths = Vec::with_capacity(rows);
        let mut smooths = Vec::with_capacity(rows);
        for combo in &combos {
            let length = combo.length.unwrap_or(100);
            let multiplier = combo.multiplier.unwrap_or(2.0);
            let atr_length = combo.atr_length.unwrap_or(14);
            let smooth = combo.smooth.unwrap_or(1);
            if length < 10 || length > 500 || length > cols {
                return Err(CudaGeometricBiasOscillatorError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            if atr_length == 0 || atr_length > cols {
                return Err(CudaGeometricBiasOscillatorError::InvalidInput(format!(
                    "invalid atr_length: atr_length={atr_length}, data_len={cols}"
                )));
            }
            if !multiplier.is_finite() || multiplier < 0.1 {
                return Err(CudaGeometricBiasOscillatorError::InvalidInput(format!(
                    "invalid multiplier: multiplier={multiplier}"
                )));
            }
            if smooth == 0 {
                return Err(CudaGeometricBiasOscillatorError::InvalidInput(format!(
                    "invalid smooth: smooth={smooth}"
                )));
            }
            let needed = Self::required_valid_bars(length, atr_length, smooth);
            if max_run < needed {
                return Err(CudaGeometricBiasOscillatorError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={max_run}"
                )));
            }
            max_length = max_length.max(length);
            max_smooth = max_smooth.max(smooth);
            lengths.push(length as i32);
            multipliers.push(multiplier);
            atr_lengths.push(atr_length as i32);
            smooths.push(smooth as i32);
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaGeometricBiasOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                multipliers
                    .len()
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                atr_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .and_then(|value| {
                smooths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaGeometricBiasOscillatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaGeometricBiasOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaGeometricBiasOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_elems = rows
            .checked_mul(max_length)
            .and_then(|value| value.checked_mul(5))
            .and_then(|value| {
                rows.checked_mul(max_smooth)
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaGeometricBiasOscillatorError::InvalidInput("scratch elems overflow".into())
            })?;
        let scratch_bytes = rows
            .checked_mul(max_length)
            .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
            .and_then(|ints| ints.checked_mul(3))
            .and_then(|ints| {
                rows.checked_mul(max_length)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
                    .and_then(|floats| floats.checked_mul(2))
                    .and_then(|floats| ints.checked_add(floats))
            })
            .and_then(|value| {
                rows.checked_mul(max_smooth)
                    .and_then(|other| other.checked_mul(std::mem::size_of::<f64>()))
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaGeometricBiasOscillatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let _ = scratch_elems;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaGeometricBiasOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_multipliers = DeviceBuffer::from_slice(&multipliers)?;
        let d_atr_lengths = DeviceBuffer::from_slice(&atr_lengths)?;
        let d_smooths = DeviceBuffer::from_slice(&smooths)?;
        let d_price_ring = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_length)? };
        let d_ordered = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_length)? };
        let d_keep = unsafe { DeviceBuffer::<i32>::uninitialized(rows * max_length)? };
        let d_stack_start = unsafe { DeviceBuffer::<i32>::uninitialized(rows * max_length)? };
        let d_stack_end = unsafe { DeviceBuffer::<i32>::uninitialized(rows * max_length)? };
        let d_smooth_ring = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_smooth)? };
        let d_out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("geometric_bias_oscillator_batch_f64")
            .map_err(|_| CudaGeometricBiasOscillatorError::MissingKernelSymbol {
                name: "geometric_bias_oscillator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + GEOMETRIC_BIAS_OSCILLATOR_BLOCK_X - 1)
            / GEOMETRIC_BIAS_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(GEOMETRIC_BIAS_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                d_multipliers.as_device_ptr(),
                d_atr_lengths.as_device_ptr(),
                d_smooths.as_device_ptr(),
                rows as i32,
                max_length as i32,
                max_smooth as i32,
                d_price_ring.as_device_ptr(),
                d_ordered.as_device_ptr(),
                d_keep.as_device_ptr(),
                d_stack_start.as_device_ptr(),
                d_stack_end.as_device_ptr(),
                d_smooth_ring.as_device_ptr(),
                d_out.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaGeometricBiasOscillatorBatchResult {
            outputs: GeometricBiasOscillatorDeviceArrayF64 {
                buf: d_out,
                rows,
                cols,
            },
            combos,
        })
    }
}
