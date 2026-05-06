#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::error::Error;
use thiserror::Error;

const LOOKBACK_TYPE_BAR_COUNT: &str = "Bar Count";
const LOOKBACK_TYPE_FVG_COUNT: &str = "FVG Count";
const ATR_PERIOD: usize = 200;

#[derive(Debug, Clone)]
pub enum FvgPositioningAverageData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct FvgPositioningAverageOutput {
    pub bull_average: Vec<f64>,
    pub bear_average: Vec<f64>,
    pub bull_mid: Vec<f64>,
    pub bear_mid: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct FvgPositioningAverageParams {
    pub lookback: Option<usize>,
    pub lookback_type: Option<String>,
    pub atr_multiplier: Option<f64>,
}

impl Default for FvgPositioningAverageParams {
    fn default() -> Self {
        Self {
            lookback: Some(30),
            lookback_type: Some(LOOKBACK_TYPE_BAR_COUNT.to_string()),
            atr_multiplier: Some(0.25),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FvgPositioningAverageInput<'a> {
    pub data: FvgPositioningAverageData<'a>,
    pub params: FvgPositioningAverageParams,
}

impl<'a> FvgPositioningAverageInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: FvgPositioningAverageParams) -> Self {
        Self {
            data: FvgPositioningAverageData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: FvgPositioningAverageParams,
    ) -> Self {
        Self {
            data: FvgPositioningAverageData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FvgPositioningAverageParams::default())
    }

    #[inline]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(30)
    }

    #[inline]
    pub fn get_lookback_type(&self) -> &str {
        self.params
            .lookback_type
            .as_deref()
            .unwrap_or(LOOKBACK_TYPE_BAR_COUNT)
    }

    #[inline]
    pub fn get_atr_multiplier(&self) -> f64 {
        self.params.atr_multiplier.unwrap_or(0.25)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FvgPositioningAverageBuilder {
    lookback: Option<usize>,
    lookback_type: Option<&'static str>,
    atr_multiplier: Option<f64>,
    kernel: Kernel,
}

impl Default for FvgPositioningAverageBuilder {
    fn default() -> Self {
        Self {
            lookback: None,
            lookback_type: None,
            atr_multiplier: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FvgPositioningAverageBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn lookback(mut self, value: usize) -> Self {
        self.lookback = Some(value);
        self
    }

    #[inline(always)]
    pub fn lookback_type(mut self, value: &str) -> Result<Self, FvgPositioningAverageError> {
        self.lookback_type = Some(canonical_lookback_type(value)?);
        Ok(self)
    }

    #[inline(always)]
    pub fn atr_multiplier(mut self, value: f64) -> Self {
        self.atr_multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<FvgPositioningAverageOutput, FvgPositioningAverageError> {
        fvg_positioning_average_with_kernel(
            &FvgPositioningAverageInput::from_candles(
                candles,
                FvgPositioningAverageParams {
                    lookback: self.lookback,
                    lookback_type: self.lookback_type.map(str::to_string),
                    atr_multiplier: self.atr_multiplier,
                },
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FvgPositioningAverageOutput, FvgPositioningAverageError> {
        fvg_positioning_average_with_kernel(
            &FvgPositioningAverageInput::from_slices(
                open,
                high,
                low,
                close,
                FvgPositioningAverageParams {
                    lookback: self.lookback,
                    lookback_type: self.lookback_type.map(str::to_string),
                    atr_multiplier: self.atr_multiplier,
                },
            ),
            self.kernel,
        )
    }
}

#[derive(Debug, Error)]
pub enum FvgPositioningAverageError {
    #[error("fvg_positioning_average: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "fvg_positioning_average: Input length mismatch: open = {open_len}, high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    InputLengthMismatch {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("fvg_positioning_average: All values are NaN.")]
    AllValuesNaN,
    #[error("fvg_positioning_average: Invalid lookback: {lookback}")]
    InvalidLookback { lookback: usize },
    #[error("fvg_positioning_average: Invalid lookback_type: {value}")]
    InvalidLookbackType { value: String },
    #[error("fvg_positioning_average: Invalid atr_multiplier: {atr_multiplier}")]
    InvalidAtrMultiplier { atr_multiplier: f64 },
    #[error("fvg_positioning_average: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fvg_positioning_average: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "fvg_positioning_average: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("fvg_positioning_average: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("fvg_positioning_average: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("fvg_positioning_average: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LookbackMode {
    BarCount,
    FvgCount,
}

#[derive(Debug, Clone, Copy)]
struct FvgLevel {
    left: usize,
    value: f64,
}

#[inline(always)]
fn is_valid_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline(always)]
fn longest_valid_run(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> (usize, bool) {
    let mut best = 0usize;
    let mut cur = 0usize;
    let mut all_valid = true;
    for (((&o, &h), &l), &c) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
    {
        if is_valid_ohlc(o, h, l, c) {
            cur += 1;
            best = best.max(cur);
        } else {
            all_valid = false;
            cur = 0;
        }
    }
    (best, all_valid)
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a FvgPositioningAverageInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), FvgPositioningAverageError> {
    match &input.data {
        FvgPositioningAverageData::Candles { candles } => Ok((
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        FvgPositioningAverageData::Slices {
            open,
            high,
            low,
            close,
        } => Ok((open, high, low, close)),
    }
}

#[inline(always)]
fn canonical_lookback_type(value: &str) -> Result<&'static str, FvgPositioningAverageError> {
    if value.eq_ignore_ascii_case(LOOKBACK_TYPE_BAR_COUNT) {
        return Ok(LOOKBACK_TYPE_BAR_COUNT);
    }
    if value.eq_ignore_ascii_case(LOOKBACK_TYPE_FVG_COUNT) {
        return Ok(LOOKBACK_TYPE_FVG_COUNT);
    }
    Err(FvgPositioningAverageError::InvalidLookbackType {
        value: value.to_string(),
    })
}

#[inline(always)]
fn parse_lookback_type(value: &str) -> Result<LookbackMode, FvgPositioningAverageError> {
    match canonical_lookback_type(value)? {
        LOOKBACK_TYPE_BAR_COUNT => Ok(LookbackMode::BarCount),
        LOOKBACK_TYPE_FVG_COUNT => Ok(LookbackMode::FvgCount),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn validate_params_only(
    lookback: usize,
    lookback_type: &str,
    atr_multiplier: f64,
) -> Result<LookbackMode, FvgPositioningAverageError> {
    if lookback == 0 {
        return Err(FvgPositioningAverageError::InvalidLookback { lookback });
    }
    if !atr_multiplier.is_finite() || atr_multiplier < 0.0 {
        return Err(FvgPositioningAverageError::InvalidAtrMultiplier { atr_multiplier });
    }
    parse_lookback_type(lookback_type)
}

#[inline(always)]
fn validate_common(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    lookback_type: &str,
    atr_multiplier: f64,
) -> Result<(LookbackMode, bool), FvgPositioningAverageError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FvgPositioningAverageError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(FvgPositioningAverageError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let lookback_mode = validate_params_only(lookback, lookback_type, atr_multiplier)?;
    let (longest, all_valid) = longest_valid_run(open, high, low, close);
    if longest == 0 {
        return Err(FvgPositioningAverageError::AllValuesNaN);
    }
    if longest < 3 {
        return Err(FvgPositioningAverageError::NotEnoughValidData {
            needed: 3,
            valid: longest,
        });
    }
    Ok((lookback_mode, all_valid))
}

#[inline(always)]
fn clear_levels(levels: &mut VecDeque<FvgLevel>, sum: &mut f64) {
    levels.clear();
    *sum = 0.0;
}

#[inline(always)]
fn push_level_count_mode(
    levels: &mut VecDeque<FvgLevel>,
    sum: &mut f64,
    level: FvgLevel,
    lookback: usize,
) {
    levels.push_back(level);
    *sum += level.value;
    while levels.len() > lookback {
        if let Some(old) = levels.pop_front() {
            *sum -= old.value;
        }
    }
}

#[inline(always)]
fn prune_levels_bar_count(
    levels: &mut VecDeque<FvgLevel>,
    sum: &mut f64,
    current_idx: usize,
    lookback: usize,
) {
    let cutoff = current_idx.saturating_sub(lookback);
    while let Some(front) = levels.front() {
        if front.left >= cutoff {
            break;
        }
        if let Some(old) = levels.pop_front() {
            *sum -= old.value;
        }
    }
}

#[inline(always)]
fn current_average(levels: &VecDeque<FvgLevel>, sum: f64) -> f64 {
    if levels.is_empty() {
        f64::NAN
    } else {
        sum / levels.len() as f64
    }
}

#[inline(always)]
fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    lookback_mode: LookbackMode,
    atr_multiplier: f64,
    out_bull_average: &mut [f64],
    out_bear_average: &mut [f64],
    out_bull_mid: &mut [f64],
    out_bear_mid: &mut [f64],
) {
    let mut bull_levels = VecDeque::<FvgLevel>::new();
    let mut bear_levels = VecDeque::<FvgLevel>::new();
    let mut bull_sum = 0.0;
    let mut bear_sum = 0.0;
    let mut valid_count = 0usize;
    let mut cumulative_range = 0.0;
    let mut tr_sum = 0.0;
    let mut atr = None::<f64>;
    let mut prev_close = 0.0;

    for i in 0..close.len() {
        if !is_valid_ohlc(open[i], high[i], low[i], close[i]) {
            clear_levels(&mut bull_levels, &mut bull_sum);
            clear_levels(&mut bear_levels, &mut bear_sum);
            valid_count = 0;
            cumulative_range = 0.0;
            tr_sum = 0.0;
            atr = None;
            out_bull_average[i] = f64::NAN;
            out_bear_average[i] = f64::NAN;
            out_bull_mid[i] = f64::NAN;
            out_bear_mid[i] = f64::NAN;
            continue;
        }

        valid_count += 1;
        let high_low = high[i] - low[i];
        cumulative_range += high_low;
        let tr = if valid_count == 1 {
            high_low
        } else {
            high_low
                .max((high[i] - prev_close).abs())
                .max((low[i] - prev_close).abs())
        };

        let threshold = if valid_count < ATR_PERIOD {
            tr_sum += tr;
            cumulative_range / valid_count as f64
        } else if valid_count == ATR_PERIOD {
            tr_sum += tr;
            let seed = tr_sum / ATR_PERIOD as f64;
            atr = Some(seed);
            seed * atr_multiplier
        } else {
            let next = (atr.unwrap_or(tr) * (ATR_PERIOD as f64 - 1.0) + tr) / ATR_PERIOD as f64;
            atr = Some(next);
            next * atr_multiplier
        };

        if valid_count >= 3 {
            let idx1 = i - 1;
            let idx2 = i - 2;

            if low[i] > high[idx2] && close[idx1] > high[idx2] && (low[i] - high[idx2]) > threshold
            {
                let level = FvgLevel {
                    left: idx2,
                    value: high[idx2],
                };
                match lookback_mode {
                    LookbackMode::BarCount => {
                        bull_levels.push_back(level);
                        bull_sum += level.value;
                    }
                    LookbackMode::FvgCount => {
                        push_level_count_mode(&mut bull_levels, &mut bull_sum, level, lookback);
                    }
                }
            }

            if high[i] < low[idx2] && close[idx1] < low[idx2] && (low[idx2] - high[i]) > threshold {
                let level = FvgLevel {
                    left: idx2,
                    value: low[idx2],
                };
                match lookback_mode {
                    LookbackMode::BarCount => {
                        bear_levels.push_back(level);
                        bear_sum += level.value;
                    }
                    LookbackMode::FvgCount => {
                        push_level_count_mode(&mut bear_levels, &mut bear_sum, level, lookback);
                    }
                }
            }
        }

        if lookback_mode == LookbackMode::BarCount {
            prune_levels_bar_count(&mut bull_levels, &mut bull_sum, i, lookback);
            prune_levels_bar_count(&mut bear_levels, &mut bear_sum, i, lookback);
        }

        let bull_average = current_average(&bull_levels, bull_sum);
        let bear_average = current_average(&bear_levels, bear_sum);
        let body_mid = 0.5 * (open[i] + close[i]);
        out_bull_average[i] = bull_average;
        out_bear_average[i] = bear_average;
        out_bull_mid[i] = if bull_average.is_nan() {
            f64::NAN
        } else {
            body_mid.max(bull_average)
        };
        out_bear_mid[i] = if bear_average.is_nan() {
            f64::NAN
        } else {
            body_mid.min(bear_average)
        };
        prev_close = close[i];
    }
}

#[inline(always)]
fn compute_row_bar_count_all_valid(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    atr_multiplier: f64,
    out_bull_average: &mut [f64],
    out_bear_average: &mut [f64],
    out_bull_mid: &mut [f64],
    out_bear_mid: &mut [f64],
) {
    let mut bull_levels = VecDeque::<FvgLevel>::new();
    let mut bear_levels = VecDeque::<FvgLevel>::new();
    let mut bull_sum = 0.0;
    let mut bear_sum = 0.0;
    let mut valid_count = 0usize;
    let mut cumulative_range = 0.0;
    let mut tr_sum = 0.0;
    let mut atr = None::<f64>;
    let mut prev_close = 0.0;

    for i in 0..close.len() {
        valid_count += 1;
        let high_low = high[i] - low[i];
        cumulative_range += high_low;
        let tr = if valid_count == 1 {
            high_low
        } else {
            high_low
                .max((high[i] - prev_close).abs())
                .max((low[i] - prev_close).abs())
        };

        let threshold = if valid_count < ATR_PERIOD {
            tr_sum += tr;
            cumulative_range / valid_count as f64
        } else if valid_count == ATR_PERIOD {
            tr_sum += tr;
            let seed = tr_sum / ATR_PERIOD as f64;
            atr = Some(seed);
            seed * atr_multiplier
        } else {
            let next = (atr.unwrap_or(tr) * (ATR_PERIOD as f64 - 1.0) + tr) / ATR_PERIOD as f64;
            atr = Some(next);
            next * atr_multiplier
        };

        if valid_count >= 3 {
            let idx1 = i - 1;
            let idx2 = i - 2;

            if low[i] > high[idx2] && close[idx1] > high[idx2] && (low[i] - high[idx2]) > threshold
            {
                let level = FvgLevel {
                    left: idx2,
                    value: high[idx2],
                };
                bull_levels.push_back(level);
                bull_sum += level.value;
            }

            if high[i] < low[idx2] && close[idx1] < low[idx2] && (low[idx2] - high[i]) > threshold {
                let level = FvgLevel {
                    left: idx2,
                    value: low[idx2],
                };
                bear_levels.push_back(level);
                bear_sum += level.value;
            }
        }

        prune_levels_bar_count(&mut bull_levels, &mut bull_sum, i, lookback);
        prune_levels_bar_count(&mut bear_levels, &mut bear_sum, i, lookback);

        let bull_average = current_average(&bull_levels, bull_sum);
        let bear_average = current_average(&bear_levels, bear_sum);
        let body_mid = 0.5 * (open[i] + close[i]);
        out_bull_average[i] = bull_average;
        out_bear_average[i] = bear_average;
        out_bull_mid[i] = if bull_average.is_nan() {
            f64::NAN
        } else {
            body_mid.max(bull_average)
        };
        out_bear_mid[i] = if bear_average.is_nan() {
            f64::NAN
        } else {
            body_mid.min(bear_average)
        };
        prev_close = close[i];
    }
}

#[inline]
pub fn fvg_positioning_average(
    input: &FvgPositioningAverageInput,
) -> Result<FvgPositioningAverageOutput, FvgPositioningAverageError> {
    fvg_positioning_average_with_kernel(input, Kernel::Auto)
}

pub fn fvg_positioning_average_with_kernel(
    input: &FvgPositioningAverageInput,
    kernel: Kernel,
) -> Result<FvgPositioningAverageOutput, FvgPositioningAverageError> {
    let (open, high, low, close) = input_slices(input)?;
    let lookback = input.get_lookback();
    let atr_multiplier = input.get_atr_multiplier();
    let (lookback_mode, all_valid) = validate_common(
        open,
        high,
        low,
        close,
        lookback,
        input.get_lookback_type(),
        atr_multiplier,
    )?;

    let mut bull_average = alloc_uninit_f64(close.len());
    let mut bear_average = alloc_uninit_f64(close.len());
    let mut bull_mid = alloc_uninit_f64(close.len());
    let mut bear_mid = alloc_uninit_f64(close.len());
    let _ = kernel;

    if all_valid && lookback_mode == LookbackMode::BarCount {
        compute_row_bar_count_all_valid(
            open,
            high,
            low,
            close,
            lookback,
            atr_multiplier,
            &mut bull_average,
            &mut bear_average,
            &mut bull_mid,
            &mut bear_mid,
        );
    } else {
        compute_row(
            open,
            high,
            low,
            close,
            lookback,
            lookback_mode,
            atr_multiplier,
            &mut bull_average,
            &mut bear_average,
            &mut bull_mid,
            &mut bear_mid,
        );
    }

    Ok(FvgPositioningAverageOutput {
        bull_average,
        bear_average,
        bull_mid,
        bear_mid,
    })
}

pub fn fvg_positioning_average_into_slice(
    out_bull_average: &mut [f64],
    out_bear_average: &mut [f64],
    out_bull_mid: &mut [f64],
    out_bear_mid: &mut [f64],
    input: &FvgPositioningAverageInput,
    kernel: Kernel,
) -> Result<(), FvgPositioningAverageError> {
    let (open, high, low, close) = input_slices(input)?;
    let lookback = input.get_lookback();
    let atr_multiplier = input.get_atr_multiplier();
    let (lookback_mode, all_valid) = validate_common(
        open,
        high,
        low,
        close,
        lookback,
        input.get_lookback_type(),
        atr_multiplier,
    )?;

    if out_bull_average.len() != close.len() {
        return Err(FvgPositioningAverageError::OutputLengthMismatch {
            expected: close.len(),
            got: out_bull_average.len(),
        });
    }
    if out_bear_average.len() != close.len() {
        return Err(FvgPositioningAverageError::OutputLengthMismatch {
            expected: close.len(),
            got: out_bear_average.len(),
        });
    }
    if out_bull_mid.len() != close.len() {
        return Err(FvgPositioningAverageError::OutputLengthMismatch {
            expected: close.len(),
            got: out_bull_mid.len(),
        });
    }
    if out_bear_mid.len() != close.len() {
        return Err(FvgPositioningAverageError::OutputLengthMismatch {
            expected: close.len(),
            got: out_bear_mid.len(),
        });
    }

    let _ = kernel;
    if all_valid && lookback_mode == LookbackMode::BarCount {
        compute_row_bar_count_all_valid(
            open,
            high,
            low,
            close,
            lookback,
            atr_multiplier,
            out_bull_average,
            out_bear_average,
            out_bull_mid,
            out_bear_mid,
        );
    } else {
        compute_row(
            open,
            high,
            low,
            close,
            lookback,
            lookback_mode,
            atr_multiplier,
            out_bull_average,
            out_bear_average,
            out_bull_mid,
            out_bear_mid,
        );
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn fvg_positioning_average_into(
    input: &FvgPositioningAverageInput,
    out_bull_average: &mut [f64],
    out_bear_average: &mut [f64],
    out_bull_mid: &mut [f64],
    out_bear_mid: &mut [f64],
) -> Result<(), FvgPositioningAverageError> {
    fvg_positioning_average_into_slice(
        out_bull_average,
        out_bear_average,
        out_bull_mid,
        out_bear_mid,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, Copy)]
pub struct FvgPositioningAverageBatchRange {
    pub lookback: (usize, usize, usize),
    pub atr_multiplier: (f64, f64, f64),
}

impl Default for FvgPositioningAverageBatchRange {
    fn default() -> Self {
        Self {
            lookback: (30, 30, 0),
            atr_multiplier: (0.25, 0.25, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FvgPositioningAverageBatchOutput {
    pub bull_average: Vec<f64>,
    pub bear_average: Vec<f64>,
    pub bull_mid: Vec<f64>,
    pub bear_mid: Vec<f64>,
    pub combos: Vec<FvgPositioningAverageParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FvgPositioningAverageBatchBuilder {
    range: FvgPositioningAverageBatchRange,
    lookback_type: Option<&'static str>,
    kernel: Kernel,
}

impl Default for FvgPositioningAverageBatchBuilder {
    fn default() -> Self {
        Self {
            range: FvgPositioningAverageBatchRange::default(),
            lookback_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl FvgPositioningAverageBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn lookback_static(mut self, value: usize) -> Self {
        self.range.lookback = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn atr_multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.atr_multiplier = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn atr_multiplier_static(mut self, value: f64) -> Self {
        self.range.atr_multiplier = (value, value, 0.0);
        self
    }

    #[inline(always)]
    pub fn lookback_type(mut self, value: &str) -> Result<Self, FvgPositioningAverageError> {
        self.lookback_type = Some(canonical_lookback_type(value)?);
        Ok(self)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FvgPositioningAverageBatchOutput, FvgPositioningAverageError> {
        fvg_positioning_average_batch_with_kernel(
            open,
            high,
            low,
            close,
            &self.range,
            self.lookback_type.unwrap_or(LOOKBACK_TYPE_BAR_COUNT),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<FvgPositioningAverageBatchOutput, FvgPositioningAverageError> {
        fvg_positioning_average_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.lookback_type.unwrap_or(LOOKBACK_TYPE_BAR_COUNT),
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_usize_range(
    field: &'static str,
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, FvgPositioningAverageError> {
    if start == 0 || end == 0 {
        return Err(FvgPositioningAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(FvgPositioningAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end {
            break;
        }
        let next = current.saturating_add(step);
        if next <= current {
            return Err(FvgPositioningAverageError::InvalidRange {
                start: field.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current = next.min(end);
        if current == *out.last().unwrap() {
            break;
        }
    }
    Ok(out)
}

#[inline(always)]
fn expand_f64_range(
    field: &'static str,
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, FvgPositioningAverageError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(FvgPositioningAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(FvgPositioningAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end || (end - current).abs() <= 1e-12 {
            break;
        }
        let next = current + step;
        if next <= current {
            return Err(FvgPositioningAverageError::InvalidRange {
                start: field.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current = if next > end { end } else { next };
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_checked(
    range: &FvgPositioningAverageBatchRange,
    lookback_type: &str,
) -> Result<Vec<FvgPositioningAverageParams>, FvgPositioningAverageError> {
    let lookbacks = expand_usize_range(
        "lookback",
        range.lookback.0,
        range.lookback.1,
        range.lookback.2,
    )?;
    let atr_multipliers = expand_f64_range(
        "atr_multiplier",
        range.atr_multiplier.0,
        range.atr_multiplier.1,
        range.atr_multiplier.2,
    )?;
    let lookback_type = canonical_lookback_type(lookback_type)?;

    let mut out = Vec::new();
    for &lookback in &lookbacks {
        for &atr_multiplier in &atr_multipliers {
            out.push(FvgPositioningAverageParams {
                lookback: Some(lookback),
                lookback_type: Some(lookback_type.to_string()),
                atr_multiplier: Some(atr_multiplier),
            });
        }
    }
    Ok(out)
}

pub fn expand_grid_fvg_positioning_average(
    range: &FvgPositioningAverageBatchRange,
    lookback_type: &str,
) -> Vec<FvgPositioningAverageParams> {
    expand_grid_checked(range, lookback_type).unwrap_or_default()
}

pub fn fvg_positioning_average_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FvgPositioningAverageBatchRange,
    lookback_type: &str,
    kernel: Kernel,
) -> Result<FvgPositioningAverageBatchOutput, FvgPositioningAverageError> {
    fvg_positioning_average_batch_inner(open, high, low, close, sweep, lookback_type, kernel, true)
}

pub fn fvg_positioning_average_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FvgPositioningAverageBatchRange,
    lookback_type: &str,
    kernel: Kernel,
) -> Result<FvgPositioningAverageBatchOutput, FvgPositioningAverageError> {
    fvg_positioning_average_batch_inner(open, high, low, close, sweep, lookback_type, kernel, false)
}

pub fn fvg_positioning_average_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FvgPositioningAverageBatchRange,
    lookback_type: &str,
    kernel: Kernel,
) -> Result<FvgPositioningAverageBatchOutput, FvgPositioningAverageError> {
    fvg_positioning_average_batch_inner(open, high, low, close, sweep, lookback_type, kernel, true)
}

fn fvg_positioning_average_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FvgPositioningAverageBatchRange,
    lookback_type: &str,
    kernel: Kernel,
    parallel: bool,
) -> Result<FvgPositioningAverageBatchOutput, FvgPositioningAverageError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(FvgPositioningAverageError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep, lookback_type)?;
    let max_lookback = combos
        .iter()
        .map(|params| params.lookback.unwrap_or(30))
        .max()
        .unwrap_or(0);
    let max_atr_multiplier = combos
        .iter()
        .map(|params| params.atr_multiplier.unwrap_or(0.25))
        .fold(0.0_f64, f64::max);
    validate_common(
        open,
        high,
        low,
        close,
        max_lookback,
        lookback_type,
        max_atr_multiplier,
    )?;

    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| FvgPositioningAverageError::InvalidInput {
            msg: "fvg_positioning_average: rows*cols overflow in batch".to_string(),
        })?;

    let mut bull_average = vec![f64::NAN; total];
    let mut bear_average = vec![f64::NAN; total];
    let mut bull_mid = vec![f64::NAN; total];
    let mut bear_mid = vec![f64::NAN; total];
    fvg_positioning_average_batch_inner_into(
        open,
        high,
        low,
        close,
        sweep,
        lookback_type,
        kernel,
        parallel,
        &mut bull_average,
        &mut bear_average,
        &mut bull_mid,
        &mut bear_mid,
    )?;

    Ok(FvgPositioningAverageBatchOutput {
        bull_average,
        bear_average,
        bull_mid,
        bear_mid,
        combos,
        rows,
        cols,
    })
}

fn fvg_positioning_average_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FvgPositioningAverageBatchRange,
    lookback_type: &str,
    kernel: Kernel,
    parallel: bool,
    out_bull_average: &mut [f64],
    out_bear_average: &mut [f64],
    out_bull_mid: &mut [f64],
    out_bear_mid: &mut [f64],
) -> Result<Vec<FvgPositioningAverageParams>, FvgPositioningAverageError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(FvgPositioningAverageError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep, lookback_type)?;
    let max_lookback = combos
        .iter()
        .map(|params| params.lookback.unwrap_or(30))
        .max()
        .unwrap_or(0);
    let max_atr_multiplier = combos
        .iter()
        .map(|params| params.atr_multiplier.unwrap_or(0.25))
        .fold(0.0_f64, f64::max);
    validate_common(
        open,
        high,
        low,
        close,
        max_lookback,
        lookback_type,
        max_atr_multiplier,
    )?;

    let cols = close.len();
    let total =
        combos
            .len()
            .checked_mul(cols)
            .ok_or_else(|| FvgPositioningAverageError::InvalidInput {
                msg: "fvg_positioning_average: rows*cols overflow in batch_into".to_string(),
            })?;
    if out_bull_average.len() != total {
        return Err(FvgPositioningAverageError::MismatchedOutputLen {
            dst_len: out_bull_average.len(),
            expected_len: total,
        });
    }
    if out_bear_average.len() != total {
        return Err(FvgPositioningAverageError::MismatchedOutputLen {
            dst_len: out_bear_average.len(),
            expected_len: total,
        });
    }
    if out_bull_mid.len() != total {
        return Err(FvgPositioningAverageError::MismatchedOutputLen {
            dst_len: out_bull_mid.len(),
            expected_len: total,
        });
    }
    if out_bear_mid.len() != total {
        return Err(FvgPositioningAverageError::MismatchedOutputLen {
            dst_len: out_bear_mid.len(),
            expected_len: total,
        });
    }

    out_bull_average.fill(f64::NAN);
    out_bear_average.fill(f64::NAN);
    out_bull_mid.fill(f64::NAN);
    out_bear_mid.fill(f64::NAN);

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize,
                  bull_average_row: &mut [f64],
                  bear_average_row: &mut [f64],
                  bull_mid_row: &mut [f64],
                  bear_mid_row: &mut [f64]| {
        let params = &combos[row];
        let lookback = params.lookback.unwrap_or(30);
        let atr_multiplier = params.atr_multiplier.unwrap_or(0.25);
        let lookback_mode = parse_lookback_type(
            params
                .lookback_type
                .as_deref()
                .unwrap_or(LOOKBACK_TYPE_BAR_COUNT),
        )
        .expect("validated lookback_type");
        compute_row(
            open,
            high,
            low,
            close,
            lookback,
            lookback_mode,
            atr_multiplier,
            bull_average_row,
            bear_average_row,
            bull_mid_row,
            bear_mid_row,
        );
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel && combos.len() > 1 {
        out_bull_average
            .par_chunks_mut(cols)
            .zip(out_bear_average.par_chunks_mut(cols))
            .zip(out_bull_mid.par_chunks_mut(cols))
            .zip(out_bear_mid.par_chunks_mut(cols))
            .enumerate()
            .for_each(
                |(row, (((bull_average_row, bear_average_row), bull_mid_row), bear_mid_row))| {
                    worker(
                        row,
                        bull_average_row,
                        bear_average_row,
                        bull_mid_row,
                        bear_mid_row,
                    );
                },
            );
    } else {
        for (row, (((bull_average_row, bear_average_row), bull_mid_row), bear_mid_row)) in
            out_bull_average
                .chunks_mut(cols)
                .zip(out_bear_average.chunks_mut(cols))
                .zip(out_bull_mid.chunks_mut(cols))
                .zip(out_bear_mid.chunks_mut(cols))
                .enumerate()
        {
            worker(
                row,
                bull_average_row,
                bear_average_row,
                bull_mid_row,
                bear_mid_row,
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, (((bull_average_row, bear_average_row), bull_mid_row), bear_mid_row)) in
            out_bull_average
                .chunks_mut(cols)
                .zip(out_bear_average.chunks_mut(cols))
                .zip(out_bull_mid.chunks_mut(cols))
                .zip(out_bear_mid.chunks_mut(cols))
                .enumerate()
        {
            worker(
                row,
                bull_average_row,
                bear_average_row,
                bull_mid_row,
                bear_mid_row,
            );
        }
    }

    Ok(combos)
}

#[derive(Debug, Clone)]
pub struct FvgPositioningAverageStream {
    lookback: usize,
    lookback_mode: LookbackMode,
    atr_multiplier: f64,
    bar_index: usize,
    valid_count: usize,
    bull_levels: VecDeque<FvgLevel>,
    bear_levels: VecDeque<FvgLevel>,
    bull_sum: f64,
    bear_sum: f64,
    cumulative_range: f64,
    tr_sum: f64,
    atr: Option<f64>,
    prev_close: Option<f64>,
    recent: VecDeque<(usize, f64, f64, f64, f64)>,
}

impl FvgPositioningAverageStream {
    pub fn try_new(
        params: FvgPositioningAverageParams,
    ) -> Result<Self, FvgPositioningAverageError> {
        let lookback = params.lookback.unwrap_or(30);
        let atr_multiplier = params.atr_multiplier.unwrap_or(0.25);
        let lookback_mode = validate_params_only(
            lookback,
            params
                .lookback_type
                .as_deref()
                .unwrap_or(LOOKBACK_TYPE_BAR_COUNT),
            atr_multiplier,
        )?;
        Ok(Self {
            lookback,
            lookback_mode,
            atr_multiplier,
            bar_index: 0,
            valid_count: 0,
            bull_levels: VecDeque::new(),
            bear_levels: VecDeque::new(),
            bull_sum: 0.0,
            bear_sum: 0.0,
            cumulative_range: 0.0,
            tr_sum: 0.0,
            atr: None,
            prev_close: None,
            recent: VecDeque::with_capacity(3),
        })
    }

    fn reset_segment(&mut self) {
        self.valid_count = 0;
        self.bull_levels.clear();
        self.bear_levels.clear();
        self.bull_sum = 0.0;
        self.bear_sum = 0.0;
        self.cumulative_range = 0.0;
        self.tr_sum = 0.0;
        self.atr = None;
        self.prev_close = None;
        self.recent.clear();
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        let idx = self.bar_index;
        self.bar_index = self.bar_index.saturating_add(1);

        if !is_valid_ohlc(open, high, low, close) {
            self.reset_segment();
            return None;
        }

        self.valid_count += 1;
        let high_low = high - low;
        self.cumulative_range += high_low;
        let tr = match self.prev_close {
            Some(prev_close) => high_low
                .max((high - prev_close).abs())
                .max((low - prev_close).abs()),
            None => high_low,
        };
        let threshold = if self.valid_count < ATR_PERIOD {
            self.tr_sum += tr;
            self.cumulative_range / self.valid_count as f64
        } else if self.valid_count == ATR_PERIOD {
            self.tr_sum += tr;
            let seed = self.tr_sum / ATR_PERIOD as f64;
            self.atr = Some(seed);
            seed * self.atr_multiplier
        } else {
            let next =
                (self.atr.unwrap_or(tr) * (ATR_PERIOD as f64 - 1.0) + tr) / ATR_PERIOD as f64;
            self.atr = Some(next);
            next * self.atr_multiplier
        };

        self.recent.push_back((idx, open, high, low, close));
        if self.recent.len() > 3 {
            self.recent.pop_front();
        }

        if self.recent.len() == 3 {
            let (left_idx, _, high2, low2, _) = self.recent[0];
            let (_, _, _, _, close1) = self.recent[1];

            if low > high2 && close1 > high2 && (low - high2) > threshold {
                let level = FvgLevel {
                    left: left_idx,
                    value: high2,
                };
                match self.lookback_mode {
                    LookbackMode::BarCount => {
                        self.bull_levels.push_back(level);
                        self.bull_sum += level.value;
                    }
                    LookbackMode::FvgCount => {
                        push_level_count_mode(
                            &mut self.bull_levels,
                            &mut self.bull_sum,
                            level,
                            self.lookback,
                        );
                    }
                }
            }

            if high < low2 && close1 < low2 && (low2 - high) > threshold {
                let level = FvgLevel {
                    left: left_idx,
                    value: low2,
                };
                match self.lookback_mode {
                    LookbackMode::BarCount => {
                        self.bear_levels.push_back(level);
                        self.bear_sum += level.value;
                    }
                    LookbackMode::FvgCount => {
                        push_level_count_mode(
                            &mut self.bear_levels,
                            &mut self.bear_sum,
                            level,
                            self.lookback,
                        );
                    }
                }
            }
        }

        if self.lookback_mode == LookbackMode::BarCount {
            prune_levels_bar_count(
                &mut self.bull_levels,
                &mut self.bull_sum,
                idx,
                self.lookback,
            );
            prune_levels_bar_count(
                &mut self.bear_levels,
                &mut self.bear_sum,
                idx,
                self.lookback,
            );
        }

        self.prev_close = Some(close);

        if self.valid_count < 3 {
            return None;
        }

        let bull_average = current_average(&self.bull_levels, self.bull_sum);
        let bear_average = current_average(&self.bear_levels, self.bear_sum);
        let body_mid = 0.5 * (open + close);
        let bull_mid = if bull_average.is_nan() {
            f64::NAN
        } else {
            body_mid.max(bull_average)
        };
        let bear_mid = if bear_average.is_nan() {
            f64::NAN
        } else {
            body_mid.min(bear_average)
        };
        Some((bull_average, bear_average, bull_mid, bear_mid))
    }
}

impl FvgPositioningAverageBuilder {
    #[inline(always)]
    pub fn into_stream(self) -> Result<FvgPositioningAverageStream, FvgPositioningAverageError> {
        FvgPositioningAverageStream::try_new(FvgPositioningAverageParams {
            lookback: self.lookback,
            lookback_type: self.lookback_type.map(str::to_string),
            atr_multiplier: self.atr_multiplier,
        })
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "fvg_positioning_average",
    signature = (open, high, low, close, lookback=30, lookback_type=LOOKBACK_TYPE_BAR_COUNT, atr_multiplier=0.25, kernel=None)
)]
pub fn fvg_positioning_average_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    lookback: usize,
    lookback_type: &str,
    atr_multiplier: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = FvgPositioningAverageInput::from_slices(
        open,
        high,
        low,
        close,
        FvgPositioningAverageParams {
            lookback: Some(lookback),
            lookback_type: Some(lookback_type.to_string()),
            atr_multiplier: Some(atr_multiplier),
        },
    );
    let out = py
        .allow_threads(|| fvg_positioning_average_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.bull_average.into_pyarray(py),
        out.bear_average.into_pyarray(py),
        out.bull_mid.into_pyarray(py),
        out.bear_mid.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "fvg_positioning_average_batch",
    signature = (open, high, low, close, lookback_range=(30, 30, 0), atr_multiplier_range=(0.25, 0.25, 0.0), lookback_type=LOOKBACK_TYPE_BAR_COUNT, kernel=None)
)]
pub fn fvg_positioning_average_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    lookback_range: (usize, usize, usize),
    atr_multiplier_range: (f64, f64, f64),
    lookback_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let output = py
        .allow_threads(|| {
            fvg_positioning_average_batch_with_kernel(
                open,
                high,
                low,
                close,
                &FvgPositioningAverageBatchRange {
                    lookback: lookback_range,
                    atr_multiplier: atr_multiplier_range,
                },
                lookback_type,
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "bull_average",
        output
            .bull_average
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bear_average",
        output
            .bear_average
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bull_mid",
        output
            .bull_mid
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bear_mid",
        output
            .bear_mid
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lookbacks",
        output
            .combos
            .iter()
            .map(|params| params.lookback.unwrap_or(30) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "lookback_types",
        output
            .combos
            .iter()
            .map(|params| {
                params
                    .lookback_type
                    .clone()
                    .unwrap_or_else(|| LOOKBACK_TYPE_BAR_COUNT.to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "atr_multipliers",
        output
            .combos
            .iter()
            .map(|params| params.atr_multiplier.unwrap_or(0.25))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "FvgPositioningAverageStream")]
pub struct FvgPositioningAverageStreamPy {
    stream: FvgPositioningAverageStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl FvgPositioningAverageStreamPy {
    #[new]
    #[pyo3(signature = (lookback=30, lookback_type=LOOKBACK_TYPE_BAR_COUNT, atr_multiplier=0.25))]
    fn new(lookback: usize, lookback_type: &str, atr_multiplier: f64) -> PyResult<Self> {
        let stream = FvgPositioningAverageStream::try_new(FvgPositioningAverageParams {
            lookback: Some(lookback),
            lookback_type: Some(lookback_type.to_string()),
            atr_multiplier: Some(atr_multiplier),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
pub fn register_fvg_positioning_average_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fvg_positioning_average_py, m)?)?;
    m.add_function(wrap_pyfunction!(fvg_positioning_average_batch_py, m)?)?;
    m.add_class::<FvgPositioningAverageStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FvgPositioningAverageBatchConfig {
    pub lookback_range: Vec<usize>,
    pub atr_multiplier_range: Vec<f64>,
    pub lookback_type: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = fvg_positioning_average_js)]
pub fn fvg_positioning_average_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    lookback_type: &str,
    atr_multiplier: f64,
) -> Result<JsValue, JsValue> {
    let input = FvgPositioningAverageInput::from_slices(
        open,
        high,
        low,
        close,
        FvgPositioningAverageParams {
            lookback: Some(lookback),
            lookback_type: Some(lookback_type.to_string()),
            atr_multiplier: Some(atr_multiplier),
        },
    );
    let out = fvg_positioning_average_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bull_average"),
        &serde_wasm_bindgen::to_value(&out.bull_average).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bear_average"),
        &serde_wasm_bindgen::to_value(&out.bear_average).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bull_mid"),
        &serde_wasm_bindgen::to_value(&out.bull_mid).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bear_mid"),
        &serde_wasm_bindgen::to_value(&out.bear_mid).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = fvg_positioning_average_batch_js)]
pub fn fvg_positioning_average_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: FvgPositioningAverageBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.lookback_range.len() != 3 || config.atr_multiplier_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = fvg_positioning_average_batch_with_kernel(
        open,
        high,
        low,
        close,
        &FvgPositioningAverageBatchRange {
            lookback: (
                config.lookback_range[0],
                config.lookback_range[1],
                config.lookback_range[2],
            ),
            atr_multiplier: (
                config.atr_multiplier_range[0],
                config.atr_multiplier_range[1],
                config.atr_multiplier_range[2],
            ),
        },
        config
            .lookback_type
            .as_deref()
            .unwrap_or(LOOKBACK_TYPE_BAR_COUNT),
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bull_average"),
        &serde_wasm_bindgen::to_value(&out.bull_average).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bear_average"),
        &serde_wasm_bindgen::to_value(&out.bear_average).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bull_mid"),
        &serde_wasm_bindgen::to_value(&out.bull_mid).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bear_mid"),
        &serde_wasm_bindgen::to_value(&out.bear_mid).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(out.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&out.combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fvg_positioning_average_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(4 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fvg_positioning_average_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 4 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fvg_positioning_average_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback: usize,
    lookback_type: &str,
    atr_multiplier: f64,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to fvg_positioning_average_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 4 * len);
        let (bull_average, tail) = out.split_at_mut(len);
        let (bear_average, tail) = tail.split_at_mut(len);
        let (bull_mid, bear_mid) = tail.split_at_mut(len);
        let input = FvgPositioningAverageInput::from_slices(
            open,
            high,
            low,
            close,
            FvgPositioningAverageParams {
                lookback: Some(lookback),
                lookback_type: Some(lookback_type.to_string()),
                atr_multiplier: Some(atr_multiplier),
            },
        );
        fvg_positioning_average_into_slice(
            bull_average,
            bear_average,
            bull_mid,
            bear_mid,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fvg_positioning_average_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
    atr_multiplier_start: f64,
    atr_multiplier_end: f64,
    atr_multiplier_step: f64,
    lookback_type: &str,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to fvg_positioning_average_batch_into",
        ));
    }

    let sweep = FvgPositioningAverageBatchRange {
        lookback: (lookback_start, lookback_end, lookback_step),
        atr_multiplier: (
            atr_multiplier_start,
            atr_multiplier_end,
            atr_multiplier_step,
        ),
    };
    let combos = expand_grid_checked(&sweep, lookback_type)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let split = rows.checked_mul(len).ok_or_else(|| {
        JsValue::from_str("rows*cols overflow in fvg_positioning_average_batch_into")
    })?;
    let total = split.checked_mul(4).ok_or_else(|| {
        JsValue::from_str("4*rows*cols overflow in fvg_positioning_average_batch_into")
    })?;

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (bull_average, tail) = out.split_at_mut(split);
        let (bear_average, tail) = tail.split_at_mut(split);
        let (bull_mid, bear_mid) = tail.split_at_mut(split);
        fvg_positioning_average_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            lookback_type,
            Kernel::Auto,
            false,
            bull_average,
            bear_average,
            bull_mid,
            bear_mid,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fvg_positioning_average_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    lookback_type: &str,
    atr_multiplier: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = fvg_positioning_average_js(
        open,
        high,
        low,
        close,
        lookback,
        lookback_type,
        atr_multiplier,
    )?;
    crate::write_wasm_object_f64_outputs("fvg_positioning_average_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fvg_positioning_average_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = fvg_positioning_average_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "fvg_positioning_average_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, ParamKV,
        ParamValue,
    };

    fn sample_ohlc() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open = vec![10.0, 12.0, 15.0, 18.0, 13.0, 9.0, 6.0, 7.0, 8.0];
        let high = vec![11.0, 13.0, 16.0, 19.0, 13.5, 9.5, 6.5, 8.0, 9.0];
        let low = vec![9.0, 12.0, 14.0, 17.0, 12.5, 8.5, 5.5, 6.5, 7.5];
        let close = vec![10.0, 12.5, 15.0, 18.0, 13.0, 9.0, 6.0, 7.5, 8.5];
        (open, high, low, close)
    }

    fn sample_ohlc_long(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let x = i as f64;
            let base = 100.0 + x * 0.08 + (x * 0.11).sin() * 3.0;
            let o = base + (x * 0.17).cos() * 0.8;
            let c = base + (x * 0.13).sin() * 0.9;
            let hi = o.max(c) + 0.9 + (x * 0.07).cos().abs() * 0.5;
            let lo = o.min(c) - 0.9 - (x * 0.05).sin().abs() * 0.4;
            open.push(o);
            high.push(hi);
            low.push(lo);
            close.push(c);
        }
        if len > 12 {
            low[6] = high[4] + 4.0;
            close[5] = high[4] + 2.0;
            low[9] = high[7] + 3.5;
            close[8] = high[7] + 1.5;
            high[12] = low[10] - 4.0;
            close[11] = low[10] - 1.0;
        }
        (open, high, low, close)
    }

    fn assert_vec_eq_nan(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (l, r) in lhs.iter().zip(rhs.iter()) {
            if l.is_nan() && r.is_nan() {
                continue;
            }
            assert_eq!(l, r);
        }
    }

    #[test]
    fn fvg_positioning_average_output_contract() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc_long(240);
        let out = fvg_positioning_average_with_kernel(
            &FvgPositioningAverageInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                FvgPositioningAverageParams::default(),
            ),
            Kernel::Scalar,
        )?;
        assert_eq!(out.bull_average.len(), close.len());
        assert_eq!(out.bear_average.len(), close.len());
        assert_eq!(out.bull_mid.len(), close.len());
        assert_eq!(out.bear_mid.len(), close.len());
        assert!(out.bull_average.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn fvg_positioning_average_exact_small_case() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc();
        let out = fvg_positioning_average_with_kernel(
            &FvgPositioningAverageInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                FvgPositioningAverageParams {
                    lookback: Some(10),
                    lookback_type: Some(LOOKBACK_TYPE_BAR_COUNT.to_string()),
                    atr_multiplier: Some(0.25),
                },
            ),
            Kernel::Scalar,
        )?;

        assert!(out.bull_average[0].is_nan());
        assert_eq!(out.bull_average[2], 11.0);
        assert_eq!(out.bull_average[3], 12.0);
        assert_eq!(out.bear_average[5], 17.0);
        assert_eq!(out.bear_average[6], 14.75);
        assert_eq!(out.bull_mid[3], 18.0);
        assert_eq!(out.bear_mid[6], 6.0);
        Ok(())
    }

    #[test]
    fn fvg_positioning_average_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc_long(180);
        let input = FvgPositioningAverageInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            FvgPositioningAverageParams::default(),
        );
        let base = fvg_positioning_average(&input)?;
        let mut bull_average = vec![f64::NAN; close.len()];
        let mut bear_average = vec![f64::NAN; close.len()];
        let mut bull_mid = vec![f64::NAN; close.len()];
        let mut bear_mid = vec![f64::NAN; close.len()];
        fvg_positioning_average_into(
            &input,
            &mut bull_average,
            &mut bear_average,
            &mut bull_mid,
            &mut bear_mid,
        )?;
        assert_vec_eq_nan(&bull_average, &base.bull_average);
        assert_vec_eq_nan(&bear_average, &base.bear_average);
        assert_vec_eq_nan(&bull_mid, &base.bull_mid);
        assert_vec_eq_nan(&bear_mid, &base.bear_mid);
        Ok(())
    }

    #[test]
    fn fvg_positioning_average_lookback_modes_differ() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc_long(220);
        let bar_count = fvg_positioning_average_with_kernel(
            &FvgPositioningAverageInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                FvgPositioningAverageParams {
                    lookback: Some(5),
                    lookback_type: Some(LOOKBACK_TYPE_BAR_COUNT.to_string()),
                    atr_multiplier: Some(0.25),
                },
            ),
            Kernel::Scalar,
        )?;
        let fvg_count = fvg_positioning_average_with_kernel(
            &FvgPositioningAverageInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                FvgPositioningAverageParams {
                    lookback: Some(1),
                    lookback_type: Some(LOOKBACK_TYPE_FVG_COUNT.to_string()),
                    atr_multiplier: Some(0.25),
                },
            ),
            Kernel::Scalar,
        )?;
        assert_ne!(bar_count.bull_average, fvg_count.bull_average);
        Ok(())
    }

    #[test]
    fn fvg_positioning_average_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc_long(160);
        let batch = fvg_positioning_average_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &FvgPositioningAverageBatchRange {
                lookback: (30, 30, 0),
                atr_multiplier: (0.25, 0.25, 0.0),
            },
            LOOKBACK_TYPE_BAR_COUNT,
            Kernel::Scalar,
        )?;
        let single = fvg_positioning_average_with_kernel(
            &FvgPositioningAverageInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                FvgPositioningAverageParams::default(),
            ),
            Kernel::Scalar,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_vec_eq_nan(&batch.bull_average, &single.bull_average);
        assert_vec_eq_nan(&batch.bear_average, &single.bear_average);
        assert_vec_eq_nan(&batch.bull_mid, &single.bull_mid);
        assert_vec_eq_nan(&batch.bear_mid, &single.bear_mid);
        Ok(())
    }

    #[test]
    fn fvg_positioning_average_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc_long(180);
        let batch = fvg_positioning_average_with_kernel(
            &FvgPositioningAverageInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                FvgPositioningAverageParams::default(),
            ),
            Kernel::Scalar,
        )?;
        let mut stream = FvgPositioningAverageBuilder::new().into_stream()?;
        let mut bull_average = Vec::with_capacity(close.len());
        let mut bear_average = Vec::with_capacity(close.len());
        let mut bull_mid = Vec::with_capacity(close.len());
        let mut bear_mid = Vec::with_capacity(close.len());

        for i in 0..close.len() {
            match stream.update(open[i], high[i], low[i], close[i]) {
                Some((ba, da, bm, dm)) => {
                    bull_average.push(ba);
                    bear_average.push(da);
                    bull_mid.push(bm);
                    bear_mid.push(dm);
                }
                None => {
                    bull_average.push(f64::NAN);
                    bear_average.push(f64::NAN);
                    bull_mid.push(f64::NAN);
                    bear_mid.push(f64::NAN);
                }
            }
        }

        assert_vec_eq_nan(&bull_average, &batch.bull_average);
        assert_vec_eq_nan(&bear_average, &batch.bear_average);
        assert_vec_eq_nan(&bull_mid, &batch.bull_mid);
        assert_vec_eq_nan(&bear_mid, &batch.bear_mid);
        Ok(())
    }

    #[test]
    fn fvg_positioning_average_rejects_invalid_params() {
        let (open, high, low, close) = sample_ohlc_long(64);
        let err = fvg_positioning_average_with_kernel(
            &FvgPositioningAverageInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                FvgPositioningAverageParams {
                    lookback: Some(0),
                    lookback_type: Some(LOOKBACK_TYPE_BAR_COUNT.to_string()),
                    atr_multiplier: Some(0.25),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FvgPositioningAverageError::InvalidLookback { .. }
        ));
    }

    #[test]
    fn fvg_positioning_average_dispatch_compute_returns_expected_outputs(
    ) -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc_long(220);
        let params = [
            ParamKV {
                key: "lookback",
                value: ParamValue::Int(30),
            },
            ParamKV {
                key: "lookback_type",
                value: ParamValue::EnumString(LOOKBACK_TYPE_BAR_COUNT),
            },
            ParamKV {
                key: "atr_multiplier",
                value: ParamValue::Float(0.25),
            },
        ];

        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "fvg_positioning_average",
            output_id: Some("bull_average"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &params,
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(out.output_id, "bull_average");
        match out.series {
            IndicatorSeries::F64(values) => assert_eq!(values.len(), close.len()),
            other => panic!("expected f64 series, got {:?}", other),
        }

        let bear_out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "fvg_positioning_average",
            output_id: Some("bear_mid"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &params,
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(bear_out.output_id, "bear_mid");
        Ok(())
    }
}
