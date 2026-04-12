#![cfg(feature = "cuda")]

use crate::indicators::standardized_psar_oscillator::{
    expand_grid_standardized_psar_oscillator, StandardizedPsarOscillatorBatchRange,
    StandardizedPsarOscillatorParams,
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

const STANDARDIZED_PSAR_OSCILLATOR_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_START: f64 = 0.02;
const DEFAULT_INCREMENT: f64 = 0.0005;
const DEFAULT_MAXIMUM: f64 = 0.2;
const DEFAULT_STANDARDIZATION_LENGTH: usize = 21;
const DEFAULT_WMA_LENGTH: usize = 40;
const DEFAULT_WMA_LAG: usize = 3;
const DEFAULT_PIVOT_LEFT: usize = 15;
const DEFAULT_PIVOT_RIGHT: usize = 1;

#[derive(Debug, Error)]
pub enum CudaStandardizedPsarOscillatorError {
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

pub struct StandardizedPsarOscillatorDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl StandardizedPsarOscillatorDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct StandardizedPsarOscillatorDeviceArrayF64Octet {
    pub oscillator: StandardizedPsarOscillatorDeviceArrayF64,
    pub ma: StandardizedPsarOscillatorDeviceArrayF64,
    pub bullish_reversal: StandardizedPsarOscillatorDeviceArrayF64,
    pub bearish_reversal: StandardizedPsarOscillatorDeviceArrayF64,
    pub regular_bullish: StandardizedPsarOscillatorDeviceArrayF64,
    pub regular_bearish: StandardizedPsarOscillatorDeviceArrayF64,
    pub bullish_weakening: StandardizedPsarOscillatorDeviceArrayF64,
    pub bearish_weakening: StandardizedPsarOscillatorDeviceArrayF64,
}

impl StandardizedPsarOscillatorDeviceArrayF64Octet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.oscillator.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.oscillator.cols
    }
}

pub struct CudaStandardizedPsarOscillatorBatchResult {
    pub outputs: StandardizedPsarOscillatorDeviceArrayF64Octet,
    pub combos: Vec<StandardizedPsarOscillatorParams>,
}

pub struct CudaStandardizedPsarOscillator {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaStandardizedPsarOscillator {
    pub fn new(device_id: usize) -> Result<Self, CudaStandardizedPsarOscillatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("standardized_psar_oscillator_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaStandardizedPsarOscillatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn max_valid_run(high: &[f64], low: &[f64], close: &[f64]) -> usize {
        let mut best = 0usize;
        let mut run = 0usize;
        for i in 0..close.len() {
            if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        best
    }

    fn required_valid_bars(standardization_length: usize, wma_length: usize) -> usize {
        standardization_length.max(2) + wma_length - 1
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaStandardizedPsarOscillatorError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaStandardizedPsarOscillatorError::OutOfMemory {
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
    ) -> Result<(), CudaStandardizedPsarOscillatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaStandardizedPsarOscillatorError::LaunchConfigTooLarge {
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
        sweep: &StandardizedPsarOscillatorBatchRange,
    ) -> Result<CudaStandardizedPsarOscillatorBatchResult, CudaStandardizedPsarOscillatorError>
    {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaStandardizedPsarOscillatorError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let max_run = Self::max_valid_run(high, low, close);
        if max_run == 0 {
            return Err(CudaStandardizedPsarOscillatorError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_standardized_psar_oscillator(sweep)
            .map_err(|err| CudaStandardizedPsarOscillatorError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaStandardizedPsarOscillatorError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut starts = Vec::with_capacity(rows);
        let mut increments = Vec::with_capacity(rows);
        let mut maximums = Vec::with_capacity(rows);
        let mut standardization_lengths = Vec::with_capacity(rows);
        let mut wma_lengths = Vec::with_capacity(rows);
        let mut wma_lags = Vec::with_capacity(rows);
        let mut pivot_lefts = Vec::with_capacity(rows);
        let mut pivot_rights = Vec::with_capacity(rows);
        let mut max_wma_length = 0usize;

        for combo in &combos {
            let start = combo.start.unwrap_or(DEFAULT_START);
            let increment = combo.increment.unwrap_or(DEFAULT_INCREMENT);
            let maximum = combo.maximum.unwrap_or(DEFAULT_MAXIMUM);
            let standardization_length = combo
                .standardization_length
                .unwrap_or(DEFAULT_STANDARDIZATION_LENGTH);
            let wma_length = combo.wma_length.unwrap_or(DEFAULT_WMA_LENGTH);
            let wma_lag = combo.wma_lag.unwrap_or(DEFAULT_WMA_LAG);
            let pivot_left = combo.pivot_left.unwrap_or(DEFAULT_PIVOT_LEFT);
            let pivot_right = combo.pivot_right.unwrap_or(DEFAULT_PIVOT_RIGHT);

            if !start.is_finite() || start <= 0.0 {
                return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "invalid start: {start}"
                )));
            }
            if !increment.is_finite() || increment <= 0.0 {
                return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "invalid increment: {increment}"
                )));
            }
            if !maximum.is_finite() || maximum <= 0.0 || maximum < start {
                return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "invalid maximum: {maximum}"
                )));
            }
            if standardization_length == 0 || standardization_length > cols {
                return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "invalid standardization_length: standardization_length={standardization_length}, data_len={cols}"
                )));
            }
            if wma_length == 0 || wma_length > cols {
                return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "invalid wma_length: wma_length={wma_length}, data_len={cols}"
                )));
            }
            if wma_lag > cols {
                return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "invalid wma_lag: {wma_lag}"
                )));
            }
            if pivot_left == 0 || pivot_left > cols {
                return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "invalid pivot_left: pivot_left={pivot_left}, data_len={cols}"
                )));
            }
            if pivot_right > cols {
                return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "invalid pivot_right: pivot_right={pivot_right}, data_len={cols}"
                )));
            }

            let needed = Self::required_valid_bars(standardization_length, wma_length);
            if max_run < needed {
                return Err(CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={max_run}"
                )));
            }

            max_wma_length = max_wma_length.max(wma_length);
            starts.push(start);
            increments.push(increment);
            maximums.push(maximum);
            standardization_lengths.push(i32::try_from(standardization_length).map_err(|_| {
                CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "standardization_length out of range: {standardization_length}"
                ))
            })?);
            wma_lengths.push(i32::try_from(wma_length).map_err(|_| {
                CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "wma_length out of range: {wma_length}"
                ))
            })?);
            wma_lags.push(i32::try_from(wma_lag).map_err(|_| {
                CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "wma_lag out of range: {wma_lag}"
                ))
            })?);
            pivot_lefts.push(i32::try_from(pivot_left).map_err(|_| {
                CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "pivot_left out of range: {pivot_left}"
                ))
            })?);
            pivot_rights.push(i32::try_from(pivot_right).map_err(|_| {
                CudaStandardizedPsarOscillatorError::InvalidInput(format!(
                    "pivot_right out of range: {pivot_right}"
                ))
            })?);
        }

        let rows_i32 = i32::try_from(rows).map_err(|_| {
            CudaStandardizedPsarOscillatorError::InvalidInput("rows out of range".into())
        })?;
        let cols_i32 = i32::try_from(cols).map_err(|_| {
            CudaStandardizedPsarOscillatorError::InvalidInput("cols out of range".into())
        })?;
        let max_wma_length_i32 = i32::try_from(max_wma_length).map_err(|_| {
            CudaStandardizedPsarOscillatorError::InvalidInput("max_wma_length out of range".into())
        })?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaStandardizedPsarOscillatorError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<f64>() * 3 + std::mem::size_of::<i32>() * 5)
            .ok_or_else(|| {
                CudaStandardizedPsarOscillatorError::InvalidInput("param bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaStandardizedPsarOscillatorError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(8))
            .ok_or_else(|| {
                CudaStandardizedPsarOscillatorError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_bytes = rows
            .checked_mul(max_wma_length)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                CudaStandardizedPsarOscillatorError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaStandardizedPsarOscillatorError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_starts = DeviceBuffer::from_slice(&starts)?;
        let d_increments = DeviceBuffer::from_slice(&increments)?;
        let d_maximums = DeviceBuffer::from_slice(&maximums)?;
        let d_standardization_lengths = DeviceBuffer::from_slice(&standardization_lengths)?;
        let d_wma_lengths = DeviceBuffer::from_slice(&wma_lengths)?;
        let d_wma_lags = DeviceBuffer::from_slice(&wma_lags)?;
        let d_pivot_lefts = DeviceBuffer::from_slice(&pivot_lefts)?;
        let d_pivot_rights = DeviceBuffer::from_slice(&pivot_rights)?;
        let d_out_oscillator = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bullish_reversal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bearish_reversal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_regular_bullish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_regular_bearish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bullish_weakening = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bearish_weakening = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_wma_buffers = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_wma_length)? };

        let func = self
            .module
            .get_function("standardized_psar_oscillator_batch_f64")
            .map_err(
                |_| CudaStandardizedPsarOscillatorError::MissingKernelSymbol {
                    name: "standardized_psar_oscillator_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + STANDARDIZED_PSAR_OSCILLATOR_BLOCK_X - 1)
            / STANDARDIZED_PSAR_OSCILLATOR_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(STANDARDIZED_PSAR_OSCILLATOR_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols_i32,
                d_starts.as_device_ptr(),
                d_increments.as_device_ptr(),
                d_maximums.as_device_ptr(),
                d_standardization_lengths.as_device_ptr(),
                d_wma_lengths.as_device_ptr(),
                d_wma_lags.as_device_ptr(),
                d_pivot_lefts.as_device_ptr(),
                d_pivot_rights.as_device_ptr(),
                if sweep.plot_bullish { 1i32 } else { 0i32 },
                if sweep.plot_bearish { 1i32 } else { 0i32 },
                rows_i32,
                max_wma_length_i32,
                d_out_oscillator.as_device_ptr(),
                d_out_ma.as_device_ptr(),
                d_out_bullish_reversal.as_device_ptr(),
                d_out_bearish_reversal.as_device_ptr(),
                d_out_regular_bullish.as_device_ptr(),
                d_out_regular_bearish.as_device_ptr(),
                d_out_bullish_weakening.as_device_ptr(),
                d_out_bearish_weakening.as_device_ptr(),
                d_wma_buffers.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaStandardizedPsarOscillatorBatchResult {
            outputs: StandardizedPsarOscillatorDeviceArrayF64Octet {
                oscillator: StandardizedPsarOscillatorDeviceArrayF64 {
                    buf: d_out_oscillator,
                    rows,
                    cols,
                },
                ma: StandardizedPsarOscillatorDeviceArrayF64 {
                    buf: d_out_ma,
                    rows,
                    cols,
                },
                bullish_reversal: StandardizedPsarOscillatorDeviceArrayF64 {
                    buf: d_out_bullish_reversal,
                    rows,
                    cols,
                },
                bearish_reversal: StandardizedPsarOscillatorDeviceArrayF64 {
                    buf: d_out_bearish_reversal,
                    rows,
                    cols,
                },
                regular_bullish: StandardizedPsarOscillatorDeviceArrayF64 {
                    buf: d_out_regular_bullish,
                    rows,
                    cols,
                },
                regular_bearish: StandardizedPsarOscillatorDeviceArrayF64 {
                    buf: d_out_regular_bearish,
                    rows,
                    cols,
                },
                bullish_weakening: StandardizedPsarOscillatorDeviceArrayF64 {
                    buf: d_out_bullish_weakening,
                    rows,
                    cols,
                },
                bearish_weakening: StandardizedPsarOscillatorDeviceArrayF64 {
                    buf: d_out_bearish_weakening,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
