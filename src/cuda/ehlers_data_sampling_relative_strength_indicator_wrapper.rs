#![cfg(feature = "cuda")]

use crate::indicators::ehlers_data_sampling_relative_strength_indicator::{
    expand_grid_ehlers_data_sampling_relative_strength_indicator,
    EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
    EhlersDataSamplingRelativeStrengthIndicatorParams,
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

const EHLERS_DATA_SAMPLING_RELATIVE_STRENGTH_INDICATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CudaEhlersDataSamplingRelativeStrengthIndicatorError {
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

pub struct EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64Triple {
    pub ds_rsi: EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64,
    pub original_rsi: EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64,
    pub signal: EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64,
}

impl EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64Triple {
    #[inline]
    pub fn rows(&self) -> usize {
        self.ds_rsi.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.ds_rsi.cols
    }
}

pub struct CudaEhlersDataSamplingRelativeStrengthIndicatorBatchResult {
    pub outputs: EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64Triple,
    pub combos: Vec<EhlersDataSamplingRelativeStrengthIndicatorParams>,
}

pub struct CudaEhlersDataSamplingRelativeStrengthIndicator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaEhlersDataSamplingRelativeStrengthIndicator {
    pub fn new(
        device_id: usize,
    ) -> Result<Self, CudaEhlersDataSamplingRelativeStrengthIndicatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!(
            "ehlers_data_sampling_relative_strength_indicator_kernel"
        )?;
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

    pub fn synchronize(&self) -> Result<(), CudaEhlersDataSamplingRelativeStrengthIndicatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn longest_valid_run_pair(open: &[f64], close: &[f64]) -> usize {
        let mut best = 0usize;
        let mut cur = 0usize;
        for (&open_value, &close_value) in open.iter().zip(close.iter()) {
            if open_value.is_finite() && close_value.is_finite() {
                cur += 1;
                best = best.max(cur);
            } else {
                cur = 0;
            }
        }
        best
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaEhlersDataSamplingRelativeStrengthIndicatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(
                    CudaEhlersDataSamplingRelativeStrengthIndicatorError::OutOfMemory {
                        required,
                        free,
                        headroom,
                    },
                );
            }
        }
        Ok(())
    }

    fn validate_launch(
        &self,
        grid: GridSize,
        block: BlockSize,
    ) -> Result<(), CudaEhlersDataSamplingRelativeStrengthIndicatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::LaunchConfigTooLarge {
                    gx: grid.x,
                    gy: grid.y,
                    gz: grid.z,
                    bx: block.x,
                    by: block.y,
                    bz: block.z,
                },
            );
        }
        Ok(())
    }

    pub fn batch_dev(
        &self,
        open: &[f64],
        close: &[f64],
        sweep: &EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
    ) -> Result<
        CudaEhlersDataSamplingRelativeStrengthIndicatorBatchResult,
        CudaEhlersDataSamplingRelativeStrengthIndicatorError,
    > {
        if open.is_empty() || close.is_empty() {
            return Err(
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(
                    "empty input".into(),
                ),
            );
        }
        if open.len() != close.len() {
            return Err(
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(format!(
                    "input length mismatch: open={}, close={}",
                    open.len(),
                    close.len()
                )),
            );
        }

        let combos = expand_grid_ehlers_data_sampling_relative_strength_indicator(sweep);
        if combos.is_empty() {
            return Err(
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(
                    "empty parameter grid".into(),
                ),
            );
        }

        let rows = combos.len();
        let cols = close.len();
        let mut max_length = 0usize;
        let mut lengths = Vec::with_capacity(rows);
        for combo in &combos {
            let length = combo.length.unwrap_or(14);
            if length == 0 || length > cols {
                return Err(
                    CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(format!(
                        "invalid length: length={length}, data_len={cols}"
                    )),
                );
            }
            max_length = max_length.max(length);
            lengths.push(length as i32);
        }

        let longest = Self::longest_valid_run_pair(open, close);
        if longest == 0 {
            return Err(
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(
                    "all values are NaN".into(),
                ),
            );
        }
        if longest < max_length {
            return Err(
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(format!(
                    "not enough valid data: needed={max_length}, valid={longest}"
                )),
            );
        }

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(
                    "input bytes overflow".into(),
                )
            })?;
        let params_bytes = lengths
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(
                    "params bytes overflow".into(),
                )
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(
                "rows*cols overflow".into(),
            )
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(
                    "output bytes overflow".into(),
                )
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput(
                    "required bytes overflow".into(),
                )
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_out_ds_rsi = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_original_rsi = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("ehlers_data_sampling_relative_strength_indicator_batch_f64")
            .map_err(|_| {
                CudaEhlersDataSamplingRelativeStrengthIndicatorError::MissingKernelSymbol {
                    name: "ehlers_data_sampling_relative_strength_indicator_batch_f64",
                }
            })?;
        let grid_x = ((rows as u32) + EHLERS_DATA_SAMPLING_RELATIVE_STRENGTH_INDICATOR_BLOCK_X - 1)
            / EHLERS_DATA_SAMPLING_RELATIVE_STRENGTH_INDICATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(EHLERS_DATA_SAMPLING_RELATIVE_STRENGTH_INDICATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                d_out_ds_rsi.as_device_ptr(),
                d_out_original_rsi.as_device_ptr(),
                d_out_signal.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaEhlersDataSamplingRelativeStrengthIndicatorBatchResult {
            outputs: EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64Triple {
                ds_rsi: EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64 {
                    buf: d_out_ds_rsi,
                    rows,
                    cols,
                },
                original_rsi: EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64 {
                    buf: d_out_original_rsi,
                    rows,
                    cols,
                },
                signal: EhlersDataSamplingRelativeStrengthIndicatorDeviceArrayF64 {
                    buf: d_out_signal,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
