#![cfg(feature = "cuda")]

use crate::indicators::autocorrelation_indicator::{
    expand_grid_autocorrelation_indicator, AutocorrelationIndicatorBatchRange,
    AutocorrelationIndicatorParams,
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

const AUTOCORRELATION_INDICATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LENGTH: usize = 20;
const DEFAULT_MAX_LAG: usize = 99;

#[derive(Debug, Error)]
pub enum CudaAutocorrelationIndicatorError {
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

pub struct AutocorrelationIndicatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl AutocorrelationIndicatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct AutocorrelationIndicatorCorrelationDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
    pub lag_count: usize,
}

impl AutocorrelationIndicatorCorrelationDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols * self.lag_count
    }
}

pub struct AutocorrelationIndicatorDeviceArrayF64Pair {
    pub filtered: AutocorrelationIndicatorDeviceArrayF64,
    pub correlations: AutocorrelationIndicatorCorrelationDeviceArrayF64,
}

impl AutocorrelationIndicatorDeviceArrayF64Pair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.filtered.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.filtered.cols
    }
}

pub struct CudaAutocorrelationIndicatorBatchResult {
    pub outputs: AutocorrelationIndicatorDeviceArrayF64Pair,
    pub combos: Vec<AutocorrelationIndicatorParams>,
}

pub struct CudaAutocorrelationIndicator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
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

impl CudaAutocorrelationIndicator {
    pub fn new(device_id: usize) -> Result<Self, CudaAutocorrelationIndicatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("autocorrelation_indicator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaAutocorrelationIndicatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaAutocorrelationIndicatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaAutocorrelationIndicatorError::OutOfMemory {
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
    ) -> Result<(), CudaAutocorrelationIndicatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaAutocorrelationIndicatorError::LaunchConfigTooLarge {
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
        sweep: &AutocorrelationIndicatorBatchRange,
    ) -> Result<CudaAutocorrelationIndicatorBatchResult, CudaAutocorrelationIndicatorError> {
        if data.is_empty() {
            return Err(CudaAutocorrelationIndicatorError::InvalidInput(
                "empty input".into(),
            ));
        }

        let combos = expand_grid_autocorrelation_indicator(sweep)
            .map_err(|err| CudaAutocorrelationIndicatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaAutocorrelationIndicatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let max_lag = sweep.max_lag.unwrap_or(DEFAULT_MAX_LAG);
        if max_lag == 0 {
            return Err(CudaAutocorrelationIndicatorError::InvalidInput(
                "invalid max_lag: 0".into(),
            ));
        }
        let use_test_signal = sweep.use_test_signal.unwrap_or(false);

        let valid = longest_valid_run(data);
        let rows = combos.len();
        let cols = data.len();
        let mut lengths = Vec::with_capacity(rows);
        let mut max_length = 0usize;
        for combo in &combos {
            let length = combo.length.unwrap_or(DEFAULT_LENGTH);
            if length == 0 || length > cols {
                return Err(CudaAutocorrelationIndicatorError::InvalidInput(format!(
                    "invalid length: length={length}, data_len={cols}"
                )));
            }
            max_length = max_length.max(length);
            lengths.push(i32::try_from(length).map_err(|_| {
                CudaAutocorrelationIndicatorError::InvalidInput(format!(
                    "length out of range: {length}"
                ))
            })?);
        }
        if !use_test_signal {
            if valid == 0 {
                return Err(CudaAutocorrelationIndicatorError::InvalidInput(
                    "all values are NaN".into(),
                ));
            }
            if valid < max_length {
                return Err(CudaAutocorrelationIndicatorError::InvalidInput(format!(
                    "not enough valid data: needed={max_length}, valid={valid}"
                )));
            }
        }

        let filtered_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaAutocorrelationIndicatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let corr_elems = filtered_elems.checked_mul(max_lag).ok_or_else(|| {
            CudaAutocorrelationIndicatorError::InvalidInput("rows*cols*lag_count overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                CudaAutocorrelationIndicatorError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| {
                CudaAutocorrelationIndicatorError::InvalidInput("params bytes overflow".into())
            })?;
        let output_bytes = filtered_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                corr_elems
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaAutocorrelationIndicatorError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaAutocorrelationIndicatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_data = DeviceBuffer::from_slice(data)?;
        let d_lengths = DeviceBuffer::from_slice(&lengths)?;
        let d_out_filtered = unsafe { DeviceBuffer::<f64>::uninitialized(filtered_elems)? };
        let d_out_correlations = unsafe { DeviceBuffer::<f64>::uninitialized(corr_elems)? };

        let func = self
            .module
            .get_function("autocorrelation_indicator_batch_f64")
            .map_err(|_| CudaAutocorrelationIndicatorError::MissingKernelSymbol {
                name: "autocorrelation_indicator_batch_f64",
            })?;
        let grid_x = ((rows as u32) + AUTOCORRELATION_INDICATOR_BLOCK_X - 1)
            / AUTOCORRELATION_INDICATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(AUTOCORRELATION_INDICATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_data.as_device_ptr(),
                cols as i32,
                d_lengths.as_device_ptr(),
                rows as i32,
                max_lag as i32,
                if use_test_signal { 1 } else { 0 },
                d_out_filtered.as_device_ptr(),
                d_out_correlations.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaAutocorrelationIndicatorBatchResult {
            outputs: AutocorrelationIndicatorDeviceArrayF64Pair {
                filtered: AutocorrelationIndicatorDeviceArrayF64 {
                    buf: d_out_filtered,
                    rows,
                    cols,
                },
                correlations: AutocorrelationIndicatorCorrelationDeviceArrayF64 {
                    buf: d_out_correlations,
                    rows,
                    cols,
                    lag_count: max_lag,
                },
            },
            combos,
        })
    }
}
