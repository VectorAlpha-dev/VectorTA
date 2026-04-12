#![cfg(feature = "cuda")]

use crate::indicators::trend_follower::{
    expand_grid_trend_follower, TrendFollowerBatchRange, TrendFollowerParams,
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

const TREND_FOLLOWER_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

const MATYPE_EMA: i32 = 0;
const MATYPE_SMA: i32 = 1;
const MATYPE_RMA: i32 = 2;
const MATYPE_WMA: i32 = 3;
const MATYPE_VWMA: i32 = 4;

#[derive(Debug, Error)]
pub enum CudaTrendFollowerError {
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

pub struct TrendFollowerDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl TrendFollowerDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct CudaTrendFollowerBatchResult {
    pub outputs: TrendFollowerDeviceArrayF64,
    pub combos: Vec<TrendFollowerParams>,
}

pub struct CudaTrendFollower {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaTrendFollower {
    pub fn new(device_id: usize) -> Result<Self, CudaTrendFollowerError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("trend_follower_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaTrendFollowerError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn parse_matype(value: &str) -> Option<i32> {
        if value.eq_ignore_ascii_case("ema") {
            Some(MATYPE_EMA)
        } else if value.eq_ignore_ascii_case("sma") {
            Some(MATYPE_SMA)
        } else if value.eq_ignore_ascii_case("rma") {
            Some(MATYPE_RMA)
        } else if value.eq_ignore_ascii_case("wma") {
            Some(MATYPE_WMA)
        } else if value.eq_ignore_ascii_case("vwma") {
            Some(MATYPE_VWMA)
        } else {
            None
        }
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaTrendFollowerError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaTrendFollowerError::OutOfMemory {
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
    ) -> Result<(), CudaTrendFollowerError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaTrendFollowerError::LaunchConfigTooLarge {
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
        volume: &[f64],
        sweep: &TrendFollowerBatchRange,
    ) -> Result<CudaTrendFollowerBatchResult, CudaTrendFollowerError> {
        if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
            return Err(CudaTrendFollowerError::InvalidInput("empty input".into()));
        }
        if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
            return Err(CudaTrendFollowerError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}, volume={}",
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }
        if !high
            .iter()
            .zip(low.iter())
            .zip(close.iter())
            .any(|((&h, &l), &c)| h.is_finite() && l.is_finite() && c.is_finite())
        {
            return Err(CudaTrendFollowerError::InvalidInput(
                "all values are invalid".into(),
            ));
        }

        let combos = expand_grid_trend_follower(sweep)
            .map_err(|err| CudaTrendFollowerError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaTrendFollowerError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut trend_periods = Vec::with_capacity(rows);
        let mut ma_periods = Vec::with_capacity(rows);
        let mut channel_rate_fractions = Vec::with_capacity(rows);
        let mut linear_regression_periods = Vec::with_capacity(rows);
        let mut ma_type_ids = Vec::with_capacity(rows);

        for combo in &combos {
            let trend_period = combo.trend_period.unwrap_or(20);
            let ma_period = combo.ma_period.unwrap_or(20);
            let channel_rate_percent = combo.channel_rate_percent.unwrap_or(1.0);
            let linear_regression_period = combo.linear_regression_period.unwrap_or(5);
            let matype = combo.matype.as_deref().unwrap_or("ema");
            let ma_type_id = Self::parse_matype(matype).ok_or_else(|| {
                CudaTrendFollowerError::InvalidInput(format!("invalid matype: {matype}"))
            })?;

            if trend_period < 1 {
                return Err(CudaTrendFollowerError::InvalidInput(format!(
                    "invalid trend_period: {trend_period}"
                )));
            }
            if ma_period == 0 || ma_period > cols {
                return Err(CudaTrendFollowerError::InvalidInput(format!(
                    "invalid ma_period: {ma_period}, data_len={cols}"
                )));
            }
            if sweep.use_linear_regression
                && (linear_regression_period < 2 || linear_regression_period > cols)
            {
                return Err(CudaTrendFollowerError::InvalidInput(format!(
                    "invalid linear_regression_period: {linear_regression_period}, data_len={cols}"
                )));
            }
            if !channel_rate_percent.is_finite() || channel_rate_percent <= 0.0 {
                return Err(CudaTrendFollowerError::InvalidInput(format!(
                    "invalid channel_rate_percent: {channel_rate_percent}"
                )));
            }

            trend_periods.push(i32::try_from(trend_period).map_err(|_| {
                CudaTrendFollowerError::InvalidInput(format!(
                    "trend_period out of range: {trend_period}"
                ))
            })?);
            ma_periods.push(i32::try_from(ma_period).map_err(|_| {
                CudaTrendFollowerError::InvalidInput(format!("ma_period out of range: {ma_period}"))
            })?);
            channel_rate_fractions.push(channel_rate_percent * 0.01);
            linear_regression_periods.push(i32::try_from(linear_regression_period).map_err(
                |_| {
                    CudaTrendFollowerError::InvalidInput(format!(
                        "linear_regression_period out of range: {linear_regression_period}"
                    ))
                },
            )?);
            ma_type_ids.push(ma_type_id);
        }

        let rows_i32 = i32::try_from(rows)
            .map_err(|_| CudaTrendFollowerError::InvalidInput("rows out of range".into()))?;
        let cols_i32 = i32::try_from(cols)
            .map_err(|_| CudaTrendFollowerError::InvalidInput("cols out of range".into()))?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| CudaTrendFollowerError::InvalidInput("input bytes overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() * 4 + std::mem::size_of::<f64>())
            .ok_or_else(|| CudaTrendFollowerError::InvalidInput("param bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaTrendFollowerError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| CudaTrendFollowerError::InvalidInput("output bytes overflow".into()))?;
        let scratch_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| CudaTrendFollowerError::InvalidInput("scratch bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaTrendFollowerError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_trend_periods = DeviceBuffer::from_slice(&trend_periods)?;
        let d_ma_periods = DeviceBuffer::from_slice(&ma_periods)?;
        let d_channel_rate_fractions = DeviceBuffer::from_slice(&channel_rate_fractions)?;
        let d_linear_regression_periods = DeviceBuffer::from_slice(&linear_regression_periods)?;
        let d_ma_type_ids = DeviceBuffer::from_slice(&ma_type_ids)?;
        let d_out_values = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_base_ma_history = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_ma_history = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("trend_follower_batch_f64")
            .map_err(|_| CudaTrendFollowerError::MissingKernelSymbol {
                name: "trend_follower_batch_f64",
            })?;
        let grid_x = ((rows as u32) + TREND_FOLLOWER_BLOCK_X - 1) / TREND_FOLLOWER_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(TREND_FOLLOWER_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols_i32,
                d_trend_periods.as_device_ptr(),
                d_ma_periods.as_device_ptr(),
                d_channel_rate_fractions.as_device_ptr(),
                d_linear_regression_periods.as_device_ptr(),
                d_ma_type_ids.as_device_ptr(),
                if sweep.use_linear_regression { 1i32 } else { 0i32 },
                rows_i32,
                d_out_values.as_device_ptr(),
                d_base_ma_history.as_device_ptr(),
                d_ma_history.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaTrendFollowerBatchResult {
            outputs: TrendFollowerDeviceArrayF64 {
                buf: d_out_values,
                rows,
                cols,
            },
            combos,
        })
    }
}
