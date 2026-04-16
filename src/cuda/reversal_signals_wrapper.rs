#![cfg(feature = "cuda")]

use crate::indicators::reversal_signals::{ReversalSignalsBatchRange, ReversalSignalsParams};
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

const REVERSAL_SIGNALS_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_LOOKBACK_PERIOD: usize = 12;
const DEFAULT_CONFIRMATION_PERIOD: usize = 3;
const DEFAULT_USE_VOLUME_CONFIRMATION: bool = true;
const DEFAULT_TREND_MA_PERIOD: usize = 50;
const DEFAULT_TREND_MA_TYPE: &str = "EMA";
const DEFAULT_MA_STEP_PERIOD: usize = 33;
const VOLUME_SMA_PERIOD: usize = 20;

const TREND_MA_SMA: i32 = 0;
const TREND_MA_EMA: i32 = 1;
const TREND_MA_WMA: i32 = 2;
const TREND_MA_VWMA: i32 = 3;

#[derive(Debug, Error)]
pub enum CudaReversalSignalsError {
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

pub struct ReversalSignalsDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl ReversalSignalsDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct ReversalSignalsDeviceOutputs {
    pub buy_signal: ReversalSignalsDeviceArrayF64,
    pub sell_signal: ReversalSignalsDeviceArrayF64,
    pub stepped_ma: ReversalSignalsDeviceArrayF64,
    pub state: ReversalSignalsDeviceArrayF64,
}

impl ReversalSignalsDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.buy_signal.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.buy_signal.cols
    }
}

pub struct CudaReversalSignalsBatchResult {
    pub outputs: ReversalSignalsDeviceOutputs,
    pub combos: Vec<ReversalSignalsParams>,
}

pub struct CudaReversalSignals {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn parse_trend_ma_kind(value: &str) -> Result<i32, CudaReversalSignalsError> {
    if value.eq_ignore_ascii_case("SMA") {
        Ok(TREND_MA_SMA)
    } else if value.eq_ignore_ascii_case("EMA") {
        Ok(TREND_MA_EMA)
    } else if value.eq_ignore_ascii_case("WMA") {
        Ok(TREND_MA_WMA)
    } else if value.eq_ignore_ascii_case("VWMA") {
        Ok(TREND_MA_VWMA)
    } else {
        Err(CudaReversalSignalsError::InvalidInput(format!(
            "invalid trend_ma_type: {value}"
        )))
    }
}

#[inline]
fn is_valid_ohlcv(open: f64, high: f64, low: f64, close: f64, volume: f64) -> bool {
    open.is_finite()
        && high.is_finite()
        && low.is_finite()
        && close.is_finite()
        && volume.is_finite()
}

#[inline]
fn longest_valid_run(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for i in 0..close.len() {
        if is_valid_ohlcv(open[i], high[i], low[i], close[i], volume[i]) {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

#[inline]
fn required_run(
    lookback_period: usize,
    trend_ma_period: usize,
    trend_ma_kind: i32,
    use_volume_confirmation: bool,
) -> usize {
    let ma_needed = if trend_ma_kind == TREND_MA_EMA {
        1
    } else {
        trend_ma_period
    };
    let volume_needed = if use_volume_confirmation {
        VOLUME_SMA_PERIOD
    } else {
        1
    };
    lookback_period.max(ma_needed).max(volume_needed)
}

#[inline]
fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CudaReversalSignalsError> {
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(CudaReversalSignalsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end {
            break;
        }
        let next = current
            .checked_add(step)
            .ok_or_else(|| CudaReversalSignalsError::InvalidInput("range step overflow".into()))?;
        if next <= current {
            break;
        }
        current = next.min(end);
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    range: &ReversalSignalsBatchRange,
) -> Result<Vec<ReversalSignalsParams>, CudaReversalSignalsError> {
    let _ = parse_trend_ma_kind(range.trend_ma_type.as_str())?;
    let lookbacks = expand_usize_range(
        range.lookback_period.0,
        range.lookback_period.1,
        range.lookback_period.2,
    )?;
    let confirmations = expand_usize_range(
        range.confirmation_period.0,
        range.confirmation_period.1,
        range.confirmation_period.2,
    )?;
    let trend_ma_periods = expand_usize_range(
        range.trend_ma_period.0,
        range.trend_ma_period.1,
        range.trend_ma_period.2,
    )?;
    let ma_steps = expand_usize_range(
        range.ma_step_period.0,
        range.ma_step_period.1,
        range.ma_step_period.2,
    )?;

    let mut combos = Vec::new();
    for &lookback_period in &lookbacks {
        for &confirmation_period in &confirmations {
            for &trend_ma_period in &trend_ma_periods {
                for &ma_step_period in &ma_steps {
                    if lookback_period == 0 {
                        return Err(CudaReversalSignalsError::InvalidInput(format!(
                            "invalid lookback_period: {lookback_period}"
                        )));
                    }
                    if trend_ma_period == 0 {
                        return Err(CudaReversalSignalsError::InvalidInput(format!(
                            "invalid trend_ma_period: {trend_ma_period}"
                        )));
                    }
                    if ma_step_period == 0 {
                        return Err(CudaReversalSignalsError::InvalidInput(format!(
                            "invalid ma_step_period: {ma_step_period}"
                        )));
                    }
                    combos.push(ReversalSignalsParams {
                        lookback_period: Some(lookback_period),
                        confirmation_period: Some(confirmation_period),
                        use_volume_confirmation: Some(range.use_volume_confirmation),
                        trend_ma_period: Some(trend_ma_period),
                        trend_ma_type: Some(range.trend_ma_type.clone()),
                        ma_step_period: Some(ma_step_period),
                    });
                }
            }
        }
    }
    Ok(combos)
}

impl CudaReversalSignals {
    pub fn new(device_id: usize) -> Result<Self, CudaReversalSignalsError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("reversal_signals_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaReversalSignalsError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(required: usize, headroom: usize) -> Result<(), CudaReversalSignalsError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaReversalSignalsError::OutOfMemory {
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
    ) -> Result<(), CudaReversalSignalsError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaReversalSignalsError::LaunchConfigTooLarge {
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
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        sweep: &ReversalSignalsBatchRange,
    ) -> Result<CudaReversalSignalsBatchResult, CudaReversalSignalsError> {
        if open.is_empty()
            || high.is_empty()
            || low.is_empty()
            || close.is_empty()
            || volume.is_empty()
        {
            return Err(CudaReversalSignalsError::InvalidInput("empty input".into()));
        }
        if open.len() != high.len()
            || open.len() != low.len()
            || open.len() != close.len()
            || open.len() != volume.len()
        {
            return Err(CudaReversalSignalsError::InvalidInput(format!(
                "input length mismatch: open={}, high={}, low={}, close={}, volume={}",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len()
            )));
        }

        let trend_ma_kind = parse_trend_ma_kind(if sweep.trend_ma_type.is_empty() {
            DEFAULT_TREND_MA_TYPE
        } else {
            sweep.trend_ma_type.as_str()
        })?;
        let use_volume_confirmation = sweep.use_volume_confirmation;
        let combos = expand_grid_checked(sweep)?;
        if combos.is_empty() {
            return Err(CudaReversalSignalsError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let max_run = longest_valid_run(open, high, low, close, volume);
        if max_run == 0 {
            return Err(CudaReversalSignalsError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut lookbacks = Vec::with_capacity(rows);
        let mut confirmations = Vec::with_capacity(rows);
        let mut trend_ma_periods = Vec::with_capacity(rows);
        let mut ma_steps = Vec::with_capacity(rows);
        let mut max_lookback = 1usize;
        let mut max_trend_ma_period = 1usize;

        for combo in &combos {
            let lookback_period = combo.lookback_period.unwrap_or(DEFAULT_LOOKBACK_PERIOD);
            let confirmation_period = combo
                .confirmation_period
                .unwrap_or(DEFAULT_CONFIRMATION_PERIOD);
            let trend_ma_period = combo.trend_ma_period.unwrap_or(DEFAULT_TREND_MA_PERIOD);
            let ma_step_period = combo.ma_step_period.unwrap_or(DEFAULT_MA_STEP_PERIOD);
            let needed = required_run(
                lookback_period,
                trend_ma_period,
                trend_ma_kind,
                use_volume_confirmation,
            );
            if max_run < needed {
                return Err(CudaReversalSignalsError::InvalidInput(format!(
                    "not enough valid data: needed={needed}, valid={max_run}"
                )));
            }

            max_lookback = max_lookback.max(lookback_period);
            max_trend_ma_period = max_trend_ma_period.max(trend_ma_period);
            lookbacks.push(i32::try_from(lookback_period).map_err(|_| {
                CudaReversalSignalsError::InvalidInput("lookback_period out of range".into())
            })?);
            confirmations.push(i32::try_from(confirmation_period).map_err(|_| {
                CudaReversalSignalsError::InvalidInput("confirmation_period out of range".into())
            })?);
            trend_ma_periods.push(i32::try_from(trend_ma_period).map_err(|_| {
                CudaReversalSignalsError::InvalidInput("trend_ma_period out of range".into())
            })?);
            ma_steps.push(i32::try_from(ma_step_period).map_err(|_| {
                CudaReversalSignalsError::InvalidInput("ma_step_period out of range".into())
            })?);
        }

        let output_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaReversalSignalsError::InvalidInput("rows*cols overflow".into()))?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(5))
            .ok_or_else(|| CudaReversalSignalsError::InvalidInput("input bytes overflow".into()))?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<i32>() * 4)
            .ok_or_else(|| CudaReversalSignalsError::InvalidInput("param bytes overflow".into()))?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                CudaReversalSignalsError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_elems = rows
            .checked_mul(max_trend_ma_period)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(rows * VOLUME_SMA_PERIOD))
            .and_then(|value| value.checked_add(rows * max_lookback * 2))
            .ok_or_else(|| {
                CudaReversalSignalsError::InvalidInput("scratch elements overflow".into())
            })?;
        let scratch_bytes = scratch_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| {
                value.checked_add(
                    rows.checked_mul(max_lookback)
                        .and_then(|x| x.checked_mul(std::mem::size_of::<i32>() * 2))
                        .unwrap_or(usize::MAX),
                )
            })
            .ok_or_else(|| {
                CudaReversalSignalsError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaReversalSignalsError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_open = DeviceBuffer::from_slice(open)?;
        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_volume = DeviceBuffer::from_slice(volume)?;
        let d_lookbacks = DeviceBuffer::from_slice(&lookbacks)?;
        let d_confirmations = DeviceBuffer::from_slice(&confirmations)?;
        let d_trend_ma_periods = DeviceBuffer::from_slice(&trend_ma_periods)?;
        let d_ma_steps = DeviceBuffer::from_slice(&ma_steps)?;
        let d_ma_price = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_trend_ma_period)? };
        let d_ma_volume =
            unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_trend_ma_period)? };
        let d_volume_sma = unsafe { DeviceBuffer::<f64>::uninitialized(rows * VOLUME_SMA_PERIOD)? };
        let d_low_idx = unsafe { DeviceBuffer::<i32>::uninitialized(rows * max_lookback)? };
        let d_low_val = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_lookback)? };
        let d_high_idx = unsafe { DeviceBuffer::<i32>::uninitialized(rows * max_lookback)? };
        let d_high_val = unsafe { DeviceBuffer::<f64>::uninitialized(rows * max_lookback)? };

        let d_buy_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_sell_signal = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_stepped_ma = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_state = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("reversal_signals_batch_f64")
            .map_err(|_| CudaReversalSignalsError::MissingKernelSymbol {
                name: "reversal_signals_batch_f64",
            })?;
        let grid_x = ((rows as u32) + REVERSAL_SIGNALS_BLOCK_X - 1) / REVERSAL_SIGNALS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(REVERSAL_SIGNALS_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_open.as_device_ptr(),
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                d_volume.as_device_ptr(),
                cols as i32,
                d_lookbacks.as_device_ptr(),
                d_confirmations.as_device_ptr(),
                d_trend_ma_periods.as_device_ptr(),
                d_ma_steps.as_device_ptr(),
                if use_volume_confirmation { 1 } else { 0 },
                trend_ma_kind,
                rows as i32,
                max_lookback as i32,
                max_trend_ma_period as i32,
                d_ma_price.as_device_ptr(),
                d_ma_volume.as_device_ptr(),
                d_volume_sma.as_device_ptr(),
                d_low_idx.as_device_ptr(),
                d_low_val.as_device_ptr(),
                d_high_idx.as_device_ptr(),
                d_high_val.as_device_ptr(),
                d_buy_signal.as_device_ptr(),
                d_sell_signal.as_device_ptr(),
                d_stepped_ma.as_device_ptr(),
                d_state.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaReversalSignalsBatchResult {
            outputs: ReversalSignalsDeviceOutputs {
                buy_signal: ReversalSignalsDeviceArrayF64 {
                    buf: d_buy_signal,
                    rows,
                    cols,
                },
                sell_signal: ReversalSignalsDeviceArrayF64 {
                    buf: d_sell_signal,
                    rows,
                    cols,
                },
                stepped_ma: ReversalSignalsDeviceArrayF64 {
                    buf: d_stepped_ma,
                    rows,
                    cols,
                },
                state: ReversalSignalsDeviceArrayF64 {
                    buf: d_state,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
