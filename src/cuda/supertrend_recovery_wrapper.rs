#![cfg(feature = "cuda")]

use crate::indicators::supertrend_recovery::{
    expand_grid, SuperTrendRecoveryBatchRange, SuperTrendRecoveryParams,
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

const SUPERTREND_RECOVERY_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_ATR_LENGTH: usize = 10;
const DEFAULT_MULTIPLIER: f64 = 3.0;
const DEFAULT_ALPHA_PERCENT: f64 = 5.0;
const DEFAULT_THRESHOLD_ATR: f64 = 1.0;
const MIN_ALPHA_PERCENT: f64 = 0.1;
const MAX_ALPHA_PERCENT: f64 = 100.0;
const MIN_MULTIPLIER: f64 = 0.1;

#[derive(Debug, Error)]
pub enum CudaSuperTrendRecoveryError {
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

pub struct SuperTrendRecoveryDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl SuperTrendRecoveryDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct SuperTrendRecoveryDeviceArrayF64Quad {
    pub band: SuperTrendRecoveryDeviceArrayF64,
    pub switch_price: SuperTrendRecoveryDeviceArrayF64,
    pub trend: SuperTrendRecoveryDeviceArrayF64,
    pub changed: SuperTrendRecoveryDeviceArrayF64,
}

impl SuperTrendRecoveryDeviceArrayF64Quad {
    #[inline]
    pub fn rows(&self) -> usize {
        self.band.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.band.cols
    }
}

pub struct CudaSuperTrendRecoveryBatchResult {
    pub outputs: SuperTrendRecoveryDeviceArrayF64Quad,
    pub combos: Vec<SuperTrendRecoveryParams>,
}

pub struct CudaSuperTrendRecovery {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaSuperTrendRecovery {
    pub fn new(device_id: usize) -> Result<Self, CudaSuperTrendRecoveryError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("supertrend_recovery_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaSuperTrendRecoveryError> {
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

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaSuperTrendRecoveryError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaSuperTrendRecoveryError::OutOfMemory {
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
    ) -> Result<(), CudaSuperTrendRecoveryError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaSuperTrendRecoveryError::LaunchConfigTooLarge {
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
        sweep: &SuperTrendRecoveryBatchRange,
    ) -> Result<CudaSuperTrendRecoveryBatchResult, CudaSuperTrendRecoveryError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaSuperTrendRecoveryError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaSuperTrendRecoveryError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let max_run = Self::max_valid_run(high, low, close);
        if max_run == 0 {
            return Err(CudaSuperTrendRecoveryError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid(sweep)
            .map_err(|err| CudaSuperTrendRecoveryError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaSuperTrendRecoveryError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut atr_lengths = Vec::with_capacity(rows);
        let mut multipliers = Vec::with_capacity(rows);
        let mut alpha_percents = Vec::with_capacity(rows);
        let mut threshold_atrs = Vec::with_capacity(rows);

        for combo in &combos {
            let atr_length = combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
            let multiplier = combo.multiplier.unwrap_or(DEFAULT_MULTIPLIER);
            let alpha_percent = combo.alpha_percent.unwrap_or(DEFAULT_ALPHA_PERCENT);
            let threshold_atr = combo.threshold_atr.unwrap_or(DEFAULT_THRESHOLD_ATR);

            if atr_length == 0 || atr_length > cols {
                return Err(CudaSuperTrendRecoveryError::InvalidInput(format!(
                    "invalid atr_length: atr_length={atr_length}, data_len={cols}"
                )));
            }
            if !multiplier.is_finite() || multiplier < MIN_MULTIPLIER {
                return Err(CudaSuperTrendRecoveryError::InvalidInput(format!(
                    "invalid multiplier: {multiplier}"
                )));
            }
            if !alpha_percent.is_finite()
                || !(MIN_ALPHA_PERCENT..=MAX_ALPHA_PERCENT).contains(&alpha_percent)
            {
                return Err(CudaSuperTrendRecoveryError::InvalidInput(format!(
                    "invalid alpha_percent: {alpha_percent}"
                )));
            }
            if !threshold_atr.is_finite() || threshold_atr < 0.0 {
                return Err(CudaSuperTrendRecoveryError::InvalidInput(format!(
                    "invalid threshold_atr: {threshold_atr}"
                )));
            }
            if max_run < atr_length {
                return Err(CudaSuperTrendRecoveryError::InvalidInput(format!(
                    "not enough valid data: needed={atr_length}, valid={max_run}"
                )));
            }

            atr_lengths.push(i32::try_from(atr_length).map_err(|_| {
                CudaSuperTrendRecoveryError::InvalidInput(format!(
                    "atr_length out of range: {atr_length}"
                ))
            })?);
            multipliers.push(multiplier);
            alpha_percents.push(alpha_percent);
            threshold_atrs.push(threshold_atr);
        }

        let rows_i32 = i32::try_from(rows)
            .map_err(|_| CudaSuperTrendRecoveryError::InvalidInput("rows out of range".into()))?;
        let cols_i32 = i32::try_from(cols)
            .map_err(|_| CudaSuperTrendRecoveryError::InvalidInput("cols out of range".into()))?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaSuperTrendRecoveryError::InvalidInput("input bytes overflow".into())
            })?;
        let params_bytes = rows
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|value| {
                rows.checked_mul(std::mem::size_of::<f64>() * 3)
                    .and_then(|other| value.checked_add(other))
            })
            .ok_or_else(|| {
                CudaSuperTrendRecoveryError::InvalidInput("params bytes overflow".into())
            })?;
        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaSuperTrendRecoveryError::InvalidInput("rows*cols overflow".into())
        })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaSuperTrendRecoveryError::InvalidInput("output bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(params_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| {
                CudaSuperTrendRecoveryError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_atr_lengths = DeviceBuffer::from_slice(&atr_lengths)?;
        let d_multipliers = DeviceBuffer::from_slice(&multipliers)?;
        let d_alpha_percents = DeviceBuffer::from_slice(&alpha_percents)?;
        let d_threshold_atrs = DeviceBuffer::from_slice(&threshold_atrs)?;
        let d_out_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_switch_price = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_trend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_changed = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("supertrend_recovery_batch_f64")
            .map_err(|_| CudaSuperTrendRecoveryError::MissingKernelSymbol {
                name: "supertrend_recovery_batch_f64",
            })?;
        let grid_x =
            ((rows as u32) + SUPERTREND_RECOVERY_BLOCK_X - 1) / SUPERTREND_RECOVERY_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(SUPERTREND_RECOVERY_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols_i32,
                d_atr_lengths.as_device_ptr(),
                d_multipliers.as_device_ptr(),
                d_alpha_percents.as_device_ptr(),
                d_threshold_atrs.as_device_ptr(),
                rows_i32,
                d_out_band.as_device_ptr(),
                d_out_switch_price.as_device_ptr(),
                d_out_trend.as_device_ptr(),
                d_out_changed.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaSuperTrendRecoveryBatchResult {
            outputs: SuperTrendRecoveryDeviceArrayF64Quad {
                band: SuperTrendRecoveryDeviceArrayF64 {
                    buf: d_out_band,
                    rows,
                    cols,
                },
                switch_price: SuperTrendRecoveryDeviceArrayF64 {
                    buf: d_out_switch_price,
                    rows,
                    cols,
                },
                trend: SuperTrendRecoveryDeviceArrayF64 {
                    buf: d_out_trend,
                    rows,
                    cols,
                },
                changed: SuperTrendRecoveryDeviceArrayF64 {
                    buf: d_out_changed,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
