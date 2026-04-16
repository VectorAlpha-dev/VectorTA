#![cfg(feature = "cuda")]

use crate::indicators::range_filtered_trend_signals::{
    RangeFilteredTrendSignalsBatchRange, RangeFilteredTrendSignalsParams,
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

const RANGE_FILTERED_TREND_SIGNALS_BLOCK_X: u32 = 64;
const DEFAULT_HEADROOM: usize = 64 * 1024 * 1024;
const DEFAULT_KALMAN_ALPHA: f64 = 0.01;
const DEFAULT_KALMAN_BETA: f64 = 0.1;
const DEFAULT_KALMAN_PERIOD: usize = 77;
const DEFAULT_DEV: f64 = 1.2;
const DEFAULT_SUPERTREND_FACTOR: f64 = 0.7;
const DEFAULT_SUPERTREND_ATR_PERIOD: usize = 7;
const WMA_PERIOD: usize = 200;

#[derive(Debug, Error)]
pub enum CudaRangeFilteredTrendSignalsError {
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

pub struct RangeFilteredTrendSignalsDeviceArrayF64 {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl RangeFilteredTrendSignalsDeviceArrayF64 {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct RangeFilteredTrendSignalsDeviceOutputs {
    pub kalman: RangeFilteredTrendSignalsDeviceArrayF64,
    pub supertrend: RangeFilteredTrendSignalsDeviceArrayF64,
    pub upper_band: RangeFilteredTrendSignalsDeviceArrayF64,
    pub lower_band: RangeFilteredTrendSignalsDeviceArrayF64,
    pub trend: RangeFilteredTrendSignalsDeviceArrayF64,
    pub kalman_trend: RangeFilteredTrendSignalsDeviceArrayF64,
    pub state: RangeFilteredTrendSignalsDeviceArrayF64,
    pub market_trending: RangeFilteredTrendSignalsDeviceArrayF64,
    pub market_ranging: RangeFilteredTrendSignalsDeviceArrayF64,
    pub short_term_bullish: RangeFilteredTrendSignalsDeviceArrayF64,
    pub short_term_bearish: RangeFilteredTrendSignalsDeviceArrayF64,
    pub long_term_bullish: RangeFilteredTrendSignalsDeviceArrayF64,
    pub long_term_bearish: RangeFilteredTrendSignalsDeviceArrayF64,
}

impl RangeFilteredTrendSignalsDeviceOutputs {
    #[inline]
    pub fn rows(&self) -> usize {
        self.kalman.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.kalman.cols
    }
}

pub struct CudaRangeFilteredTrendSignalsBatchResult {
    pub outputs: RangeFilteredTrendSignalsDeviceOutputs,
    pub combos: Vec<RangeFilteredTrendSignalsParams>,
}

pub struct CudaRangeFilteredTrendSignals {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

#[inline]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, CudaRangeFilteredTrendSignalsError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start <= end {
        let mut current = start;
        while current <= end {
            out.push(current);
            current = current.checked_add(step).ok_or_else(|| {
                CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                ))
            })?;
        }
    } else {
        let mut current = start;
        while current >= end {
            out.push(current);
            current = current.checked_sub(step).ok_or_else(|| {
                CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
                    "invalid range: start={start}, end={end}, step={step}"
                ))
            })?;
            if current < end {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, CudaRangeFilteredTrendSignalsError> {
    const EPS: f64 = 1e-12;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    if step.abs() < EPS || (start - end).abs() < EPS {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    let direction = if end >= start { 1.0 } else { -1.0 };
    let step_eff = direction * step.abs();
    let mut current = start;
    if direction > 0.0 {
        while current <= end + EPS {
            out.push(current);
            current += step_eff;
        }
    } else {
        while current >= end - EPS {
            out.push(current);
            current += step_eff;
        }
    }
    if out.is_empty() {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "invalid range: start={start}, end={end}, step={step}"
        )));
    }
    Ok(out)
}

#[inline]
fn expand_grid_checked(
    range: &RangeFilteredTrendSignalsBatchRange,
) -> Result<Vec<RangeFilteredTrendSignalsParams>, CudaRangeFilteredTrendSignalsError> {
    let kalman_alphas = axis_f64(range.kalman_alpha)?;
    let kalman_betas = axis_f64(range.kalman_beta)?;
    let kalman_periods = axis_usize(range.kalman_period)?;
    let devs = axis_f64(range.dev)?;
    let supertrend_factors = axis_f64(range.supertrend_factor)?;
    let supertrend_atr_periods = axis_usize(range.supertrend_atr_period)?;

    let total = kalman_alphas
        .len()
        .checked_mul(kalman_betas.len())
        .and_then(|value| value.checked_mul(kalman_periods.len()))
        .and_then(|value| value.checked_mul(devs.len()))
        .and_then(|value| value.checked_mul(supertrend_factors.len()))
        .and_then(|value| value.checked_mul(supertrend_atr_periods.len()))
        .ok_or_else(|| {
            CudaRangeFilteredTrendSignalsError::InvalidInput("parameter grid overflow".into())
        })?;

    let mut out = Vec::with_capacity(total);
    for &kalman_alpha in &kalman_alphas {
        for &kalman_beta in &kalman_betas {
            for &kalman_period in &kalman_periods {
                for &dev in &devs {
                    for &supertrend_factor in &supertrend_factors {
                        for &supertrend_atr_period in &supertrend_atr_periods {
                            out.push(RangeFilteredTrendSignalsParams {
                                kalman_alpha: Some(kalman_alpha),
                                kalman_beta: Some(kalman_beta),
                                kalman_period: Some(kalman_period),
                                dev: Some(dev),
                                supertrend_factor: Some(supertrend_factor),
                                supertrend_atr_period: Some(supertrend_atr_period),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[inline]
fn longest_valid_run(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for ((&h, &l), &c) in high.iter().zip(low.iter()).zip(close.iter()) {
        if h.is_finite() && l.is_finite() && c.is_finite() {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

#[inline]
fn validate_combo(
    params: &RangeFilteredTrendSignalsParams,
    data_len: usize,
    max_run: usize,
) -> Result<(), CudaRangeFilteredTrendSignalsError> {
    let kalman_alpha = params.kalman_alpha.unwrap_or(DEFAULT_KALMAN_ALPHA);
    let kalman_beta = params.kalman_beta.unwrap_or(DEFAULT_KALMAN_BETA);
    let kalman_period = params.kalman_period.unwrap_or(DEFAULT_KALMAN_PERIOD);
    let dev = params.dev.unwrap_or(DEFAULT_DEV);
    let supertrend_factor = params
        .supertrend_factor
        .unwrap_or(DEFAULT_SUPERTREND_FACTOR);
    let supertrend_atr_period = params
        .supertrend_atr_period
        .unwrap_or(DEFAULT_SUPERTREND_ATR_PERIOD);

    if !kalman_alpha.is_finite() || kalman_alpha <= 0.0 {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "invalid kalman_alpha: {kalman_alpha}"
        )));
    }
    if !kalman_beta.is_finite() || kalman_beta < 0.0 {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "invalid kalman_beta: {kalman_beta}"
        )));
    }
    if kalman_period == 0 || kalman_period > data_len {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "invalid kalman_period: kalman_period={kalman_period}, data_len={data_len}"
        )));
    }
    if !dev.is_finite() || dev < 0.0 {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "invalid dev: {dev}"
        )));
    }
    if !supertrend_factor.is_finite() || supertrend_factor < 0.0 {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "invalid supertrend_factor: {supertrend_factor}"
        )));
    }
    if supertrend_atr_period == 0 || supertrend_atr_period > data_len {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "invalid supertrend_atr_period: supertrend_atr_period={supertrend_atr_period}, data_len={data_len}"
        )));
    }
    let needed = WMA_PERIOD.max(supertrend_atr_period);
    if max_run < needed {
        return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
            "not enough valid data: needed={needed}, valid={max_run}"
        )));
    }
    Ok(())
}

impl CudaRangeFilteredTrendSignals {
    pub fn new(device_id: usize) -> Result<Self, CudaRangeFilteredTrendSignalsError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        let module = crate::load_cuda_embedded_module!("range_filtered_trend_signals_kernel")?;
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

    pub fn synchronize(&self) -> Result<(), CudaRangeFilteredTrendSignalsError> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn will_fit(
        required: usize,
        headroom: usize,
    ) -> Result<(), CudaRangeFilteredTrendSignalsError> {
        if let Ok((free, _)) = mem_get_info() {
            if required.saturating_add(headroom) > free {
                return Err(CudaRangeFilteredTrendSignalsError::OutOfMemory {
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
    ) -> Result<(), CudaRangeFilteredTrendSignalsError> {
        let device = Device::get_device(self.device_id)?;
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .unwrap_or(i32::MAX) as u32;
        let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
        if threads > max_threads || grid.x > max_grid_x {
            return Err(CudaRangeFilteredTrendSignalsError::LaunchConfigTooLarge {
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
        range: &RangeFilteredTrendSignalsBatchRange,
    ) -> Result<CudaRangeFilteredTrendSignalsBatchResult, CudaRangeFilteredTrendSignalsError> {
        if high.is_empty() || low.is_empty() || close.is_empty() {
            return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(
                "empty input".into(),
            ));
        }
        if high.len() != low.len() || high.len() != close.len() {
            return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(format!(
                "input length mismatch: high={}, low={}, close={}",
                high.len(),
                low.len(),
                close.len()
            )));
        }

        let max_run = longest_valid_run(high, low, close);
        if max_run == 0 {
            return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(
                "all values are NaN".into(),
            ));
        }

        let combos = expand_grid_checked(range)?;
        if combos.is_empty() {
            return Err(CudaRangeFilteredTrendSignalsError::InvalidInput(
                "empty parameter grid".into(),
            ));
        }

        let rows = combos.len();
        let cols = close.len();
        let mut kalman_alphas = Vec::with_capacity(rows);
        let mut kalman_betas = Vec::with_capacity(rows);
        let mut kalman_periods = Vec::with_capacity(rows);
        let mut devs = Vec::with_capacity(rows);
        let mut supertrend_factors = Vec::with_capacity(rows);
        let mut supertrend_atr_periods = Vec::with_capacity(rows);

        for combo in &combos {
            validate_combo(combo, cols, max_run)?;
            kalman_alphas.push(combo.kalman_alpha.unwrap_or(DEFAULT_KALMAN_ALPHA));
            kalman_betas.push(combo.kalman_beta.unwrap_or(DEFAULT_KALMAN_BETA));
            kalman_periods.push(
                i32::try_from(combo.kalman_period.unwrap_or(DEFAULT_KALMAN_PERIOD)).map_err(
                    |_| {
                        CudaRangeFilteredTrendSignalsError::InvalidInput(
                            "kalman_period out of range".into(),
                        )
                    },
                )?,
            );
            devs.push(combo.dev.unwrap_or(DEFAULT_DEV));
            supertrend_factors.push(combo.supertrend_factor.unwrap_or(DEFAULT_SUPERTREND_FACTOR));
            supertrend_atr_periods.push(
                i32::try_from(
                    combo
                        .supertrend_atr_period
                        .unwrap_or(DEFAULT_SUPERTREND_ATR_PERIOD),
                )
                .map_err(|_| {
                    CudaRangeFilteredTrendSignalsError::InvalidInput(
                        "supertrend_atr_period out of range".into(),
                    )
                })?,
            );
        }

        let output_elems = rows.checked_mul(cols).ok_or_else(|| {
            CudaRangeFilteredTrendSignalsError::InvalidInput("rows*cols overflow".into())
        })?;
        let input_bytes = cols
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| {
                CudaRangeFilteredTrendSignalsError::InvalidInput("input bytes overflow".into())
            })?;
        let param_bytes = rows
            .checked_mul(std::mem::size_of::<f64>() * 4 + std::mem::size_of::<i32>() * 2)
            .ok_or_else(|| {
                CudaRangeFilteredTrendSignalsError::InvalidInput("param bytes overflow".into())
            })?;
        let output_bytes = output_elems
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|value| value.checked_mul(13))
            .ok_or_else(|| {
                CudaRangeFilteredTrendSignalsError::InvalidInput("output bytes overflow".into())
            })?;
        let scratch_bytes = rows
            .checked_mul(WMA_PERIOD)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                CudaRangeFilteredTrendSignalsError::InvalidInput("scratch bytes overflow".into())
            })?;
        let required = input_bytes
            .checked_add(param_bytes)
            .and_then(|value| value.checked_add(output_bytes))
            .and_then(|value| value.checked_add(scratch_bytes))
            .ok_or_else(|| {
                CudaRangeFilteredTrendSignalsError::InvalidInput("required bytes overflow".into())
            })?;
        Self::will_fit(required, DEFAULT_HEADROOM)?;

        let d_high = DeviceBuffer::from_slice(high)?;
        let d_low = DeviceBuffer::from_slice(low)?;
        let d_close = DeviceBuffer::from_slice(close)?;
        let d_kalman_alphas = DeviceBuffer::from_slice(&kalman_alphas)?;
        let d_kalman_betas = DeviceBuffer::from_slice(&kalman_betas)?;
        let d_kalman_periods = DeviceBuffer::from_slice(&kalman_periods)?;
        let d_devs = DeviceBuffer::from_slice(&devs)?;
        let d_supertrend_factors = DeviceBuffer::from_slice(&supertrend_factors)?;
        let d_supertrend_atr_periods = DeviceBuffer::from_slice(&supertrend_atr_periods)?;
        let d_wma_scratch = unsafe { DeviceBuffer::<f64>::uninitialized(rows * WMA_PERIOD)? };
        let d_kalman = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_supertrend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_upper_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_lower_band = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_trend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_kalman_trend = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_state = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_market_trending = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_market_ranging = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_short_term_bullish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_short_term_bearish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_long_term_bullish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };
        let d_long_term_bearish = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        let func = self
            .module
            .get_function("range_filtered_trend_signals_batch_f64")
            .map_err(
                |_| CudaRangeFilteredTrendSignalsError::MissingKernelSymbol {
                    name: "range_filtered_trend_signals_batch_f64",
                },
            )?;
        let grid_x = ((rows as u32) + RANGE_FILTERED_TREND_SIGNALS_BLOCK_X - 1)
            / RANGE_FILTERED_TREND_SIGNALS_BLOCK_X;
        let grid = GridSize::x(grid_x.max(1));
        let block = BlockSize::x(RANGE_FILTERED_TREND_SIGNALS_BLOCK_X);
        self.validate_launch(grid, block)?;
        let stream = &self.stream;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_high.as_device_ptr(),
                d_low.as_device_ptr(),
                d_close.as_device_ptr(),
                cols as i32,
                d_kalman_alphas.as_device_ptr(),
                d_kalman_betas.as_device_ptr(),
                d_kalman_periods.as_device_ptr(),
                d_devs.as_device_ptr(),
                d_supertrend_factors.as_device_ptr(),
                d_supertrend_atr_periods.as_device_ptr(),
                rows as i32,
                d_wma_scratch.as_device_ptr(),
                d_kalman.as_device_ptr(),
                d_supertrend.as_device_ptr(),
                d_upper_band.as_device_ptr(),
                d_lower_band.as_device_ptr(),
                d_trend.as_device_ptr(),
                d_kalman_trend.as_device_ptr(),
                d_state.as_device_ptr(),
                d_market_trending.as_device_ptr(),
                d_market_ranging.as_device_ptr(),
                d_short_term_bullish.as_device_ptr(),
                d_short_term_bearish.as_device_ptr(),
                d_long_term_bullish.as_device_ptr(),
                d_long_term_bearish.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        Ok(CudaRangeFilteredTrendSignalsBatchResult {
            outputs: RangeFilteredTrendSignalsDeviceOutputs {
                kalman: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_kalman,
                    rows,
                    cols,
                },
                supertrend: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_supertrend,
                    rows,
                    cols,
                },
                upper_band: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_upper_band,
                    rows,
                    cols,
                },
                lower_band: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_lower_band,
                    rows,
                    cols,
                },
                trend: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_trend,
                    rows,
                    cols,
                },
                kalman_trend: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_kalman_trend,
                    rows,
                    cols,
                },
                state: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_state,
                    rows,
                    cols,
                },
                market_trending: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_market_trending,
                    rows,
                    cols,
                },
                market_ranging: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_market_ranging,
                    rows,
                    cols,
                },
                short_term_bullish: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_short_term_bullish,
                    rows,
                    cols,
                },
                short_term_bearish: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_short_term_bearish,
                    rows,
                    cols,
                },
                long_term_bullish: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_long_term_bullish,
                    rows,
                    cols,
                },
                long_term_bearish: RangeFilteredTrendSignalsDeviceArrayF64 {
                    buf: d_long_term_bearish,
                    rows,
                    cols,
                },
            },
            combos,
        })
    }
}
