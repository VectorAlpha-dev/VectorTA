#![cfg(feature = "cuda")]

use crate::indicators::bulls_v_bears::{
    bulls_v_bears_expand_grid, BullsVBearsBatchRange, BullsVBearsCalculationMethod,
    BullsVBearsMaType, BullsVBearsParams,
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

const BULLS_V_BEARS_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;

const MA_EMA: i32 = 0;
const MA_SMA: i32 = 1;
const MA_WMA: i32 = 2;

const METHOD_NORMALIZED: i32 = 0;
const METHOD_RAW: i32 = 1;

#[derive(Debug, Error)]
pub enum CudaBullsVBearsError {
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

pub struct BullsVBearsDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl BullsVBearsDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct BullsVBearsDeviceArrayF64Decet {
    pub value: BullsVBearsDeviceArrayF64,
    pub bull: BullsVBearsDeviceArrayF64,
    pub bear: BullsVBearsDeviceArrayF64,
    pub ma: BullsVBearsDeviceArrayF64,
    pub upper: BullsVBearsDeviceArrayF64,
    pub lower: BullsVBearsDeviceArrayF64,
    pub bullish_signal: BullsVBearsDeviceArrayF64,
    pub bearish_signal: BullsVBearsDeviceArrayF64,
    pub zero_cross_up: BullsVBearsDeviceArrayF64,
    pub zero_cross_down: BullsVBearsDeviceArrayF64,
}

impl BullsVBearsDeviceArrayF64Decet {
    #[inline]
    pub fn rows(&self) -> usize {
        self.value.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.value.cols
    }
}

pub struct CudaBullsVBearsBatchResult {
    pub outputs: BullsVBearsDeviceArrayF64Decet,
    pub combos: Vec<BullsVBearsParams>,
}

pub struct CudaBullsVBears {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaBullsVBears {
    pub fn new(device_id: usize) -> Result<Self, CudaBullsVBearsError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("bulls_v_bears_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaBullsVBearsError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaBullsVBearsError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaBullsVBearsError::OutOfMemory {
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
    ) -> Result<(), CudaBullsVBearsError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaBullsVBearsError::LaunchConfigTooLarge {
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

    fn ma_type_id(value: BullsVBearsMaType) -> i32 {
        match value {
            BullsVBearsMaType::Ema => MA_EMA,
            BullsVBearsMaType::Sma => MA_SMA,
            BullsVBearsMaType::Wma => MA_WMA,
        }
    }

    fn calculation_method_id(value: BullsVBearsCalculationMethod) -> i32 {
        match value {
            BullsVBearsCalculationMethod::Normalized => METHOD_NORMALIZED,
            BullsVBearsCalculationMethod::Raw => METHOD_RAW,
        }
    }

    pub fn batch_dev(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        sweep: &BullsVBearsBatchRange,
    ) -> Result<CudaBullsVBearsBatchResult, CudaBullsVBearsError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaBullsVBearsError::InvalidInput("empty input".into()));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaBullsVBearsError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }
        if !high
            .iter()
            .zip(low.iter())
            .zip(close.iter())
            .any(|((&h, &l), &c)| h.is_finite() && l.is_finite() && c.is_finite())
        {
            return Err(CudaBullsVBearsError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = bulls_v_bears_expand_grid(sweep)
            .map_err(|err| CudaBullsVBearsError::InvalidInput(err.to_string()))?;
        if combos.is_empty() {
            return Err(CudaBullsVBearsError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut periods = Vec::with_capacity(rows);
        let mut normalized_bars_backs = Vec::with_capacity(rows);
        let mut raw_rolling_periods = Vec::with_capacity(rows);
        let mut raw_threshold_percentiles = Vec::with_capacity(rows);
        let mut threshold_levels = Vec::with_capacity(rows);

        for combo in &combos {
            let period = combo
                .period
                .ok_or_else(|| CudaBullsVBearsError::InvalidInput("missing period".to_string()))?;
            let normalized_bars_back = combo.normalized_bars_back.ok_or_else(|| {
                CudaBullsVBearsError::InvalidInput("missing normalized_bars_back".to_string())
            })?;
            let raw_rolling_period = combo.raw_rolling_period.ok_or_else(|| {
                CudaBullsVBearsError::InvalidInput("missing raw_rolling_period".to_string())
            })?;
            let raw_threshold_percentile = combo.raw_threshold_percentile.ok_or_else(|| {
                CudaBullsVBearsError::InvalidInput("missing raw_threshold_percentile".to_string())
            })?;
            let threshold_level = combo.threshold_level.ok_or_else(|| {
                CudaBullsVBearsError::InvalidInput("missing threshold_level".to_string())
            })?;

            if period == 0 {
                return Err(CudaBullsVBearsError::InvalidInput(format!(
                    "invalid period: {period}"
                )));
            }
            if normalized_bars_back == 0 {
                return Err(CudaBullsVBearsError::InvalidInput(format!(
                    "invalid normalized_bars_back: {normalized_bars_back}"
                )));
            }
            if raw_rolling_period == 0 {
                return Err(CudaBullsVBearsError::InvalidInput(format!(
                    "invalid raw_rolling_period: {raw_rolling_period}"
                )));
            }
            if !raw_threshold_percentile.is_finite()
                || !(80.0..=99.0).contains(&raw_threshold_percentile)
            {
                return Err(CudaBullsVBearsError::InvalidInput(format!(
                    "invalid raw_threshold_percentile: {raw_threshold_percentile}"
                )));
            }
            if !threshold_level.is_finite() || !(0.0..=100.0).contains(&threshold_level) {
                return Err(CudaBullsVBearsError::InvalidInput(format!(
                    "invalid threshold_level: {threshold_level}"
                )));
            }

            periods.push(i32::try_from(period).map_err(|_| {
                CudaBullsVBearsError::InvalidInput(format!("period out of range: {period}"))
            })?);
            normalized_bars_backs.push(i32::try_from(normalized_bars_back).map_err(|_| {
                CudaBullsVBearsError::InvalidInput(format!(
                    "normalized_bars_back out of range: {normalized_bars_back}"
                ))
            })?);
            raw_rolling_periods.push(i32::try_from(raw_rolling_period).map_err(|_| {
                CudaBullsVBearsError::InvalidInput(format!(
                    "raw_rolling_period out of range: {raw_rolling_period}"
                ))
            })?);
            raw_threshold_percentiles.push(raw_threshold_percentile);
            threshold_levels.push(threshold_level);
        }

        let rows_i32 = i32::try_from(rows)
            .map_err(|_| CudaBullsVBearsError::InvalidInput("rows out of range".into()))?;
        let cols_i32 = i32::try_from(cols)
            .map_err(|_| CudaBullsVBearsError::InvalidInput("cols out of range".into()))?;

        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| CudaBullsVBearsError::InvalidInput("input bytes overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() * 3 + std::mem::size_of::<f64>() * 2)
            .ok_or_else(|| CudaBullsVBearsError::InvalidInput("param bytes overflow".into()))?;
        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaBullsVBearsError::InvalidInput("rows*cols overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(10))
            .ok_or_else(|| CudaBullsVBearsError::InvalidInput("output bytes overflow".into()))?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .ok_or_else(|| CudaBullsVBearsError::InvalidInput("required bytes overflow".into()))?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_normalized_bars_backs = DeviceBuffer::from_slice(&normalized_bars_backs)?;
        let d_raw_rolling_periods = DeviceBuffer::from_slice(&raw_rolling_periods)?;
        let d_raw_threshold_percentiles = DeviceBuffer::from_slice(&raw_threshold_percentiles)?;
        let d_threshold_levels = DeviceBuffer::from_slice(&threshold_levels)?;
        let d_out_value = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bull = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bear = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_ma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_upper = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_lower = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bullish_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_bearish_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_zero_cross_up = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_out_zero_cross_down = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("bulls_v_bears_batch_f64")
            .map_err(|_| CudaBullsVBearsError::MissingKernelSymbol {
                name: "bulls_v_bears_batch_f64",
            })?;
        let grid_x = ((rows as u32) + BULLS_V_BEARS_BLOCK_X - 1) / BULLS_V_BEARS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(BULLS_V_BEARS_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols_i32,
                d_periods.as_device_ptr(),
                d_normalized_bars_backs.as_device_ptr(),
                d_raw_rolling_periods.as_device_ptr(),
                d_raw_threshold_percentiles.as_device_ptr(),
                d_threshold_levels.as_device_ptr(),
                Self::ma_type_id(sweep.ma_type),
                Self::calculation_method_id(sweep.calculation_method),
                rows_i32,
                d_out_value.as_device_ptr(),
                d_out_bull.as_device_ptr(),
                d_out_bear.as_device_ptr(),
                d_out_ma.as_device_ptr(),
                d_out_upper.as_device_ptr(),
                d_out_lower.as_device_ptr(),
                d_out_bullish_signal.as_device_ptr(),
                d_out_bearish_signal.as_device_ptr(),
                d_out_zero_cross_up.as_device_ptr(),
                d_out_zero_cross_down.as_device_ptr()
            ))?;
        }

        self.stream.synchronize()?;

        Ok(CudaBullsVBearsBatchResult {
            outputs: BullsVBearsDeviceArrayF64Decet {
                value: BullsVBearsDeviceArrayF64 {
                    buf: d_out_value,
                    rows,
                    cols,
                },
                bull: BullsVBearsDeviceArrayF64 {
                    buf: d_out_bull,
                    rows,
                    cols,
                },
                bear: BullsVBearsDeviceArrayF64 {
                    buf: d_out_bear,
                    rows,
                    cols,
                },
                ma: BullsVBearsDeviceArrayF64 {
                    buf: d_out_ma,
                    rows,
                    cols,
                },
                upper: BullsVBearsDeviceArrayF64 {
                    buf: d_out_upper,
                    rows,
                    cols,
                },
                lower: BullsVBearsDeviceArrayF64 {
                    buf: d_out_lower,
                    rows,
                    cols,
                },
                bullish_signal: BullsVBearsDeviceArrayF64 {
                    buf: d_out_bullish_signal,
                    rows,
                    cols,
                },
                bearish_signal: BullsVBearsDeviceArrayF64 {
                    buf: d_out_bearish_signal,
                    rows,
                    cols,
                },
                zero_cross_up: BullsVBearsDeviceArrayF64 {
                    buf: d_out_zero_cross_up,
                    rows,
                    cols,
                },
                zero_cross_down: BullsVBearsDeviceArrayF64 {
                    buf: d_out_zero_cross_down,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
