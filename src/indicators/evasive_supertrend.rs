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
use crate::utilities::helpers::alloc_with_nan_prefix;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

const DEFAULT_ATR_LENGTH: usize = 10;
const DEFAULT_BASE_MULTIPLIER: f64 = 3.0;
const DEFAULT_NOISE_THRESHOLD: f64 = 1.0;
const DEFAULT_EXPANSION_ALPHA: f64 = 0.5;

#[derive(Debug, Clone)]
pub enum EvasiveSuperTrendData<'a> {
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
pub struct EvasiveSuperTrendOutput {
    pub band: Vec<f64>,
    pub state: Vec<f64>,
    pub noisy: Vec<f64>,
    pub changed: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EvasiveSuperTrendParams {
    pub atr_length: Option<usize>,
    pub base_multiplier: Option<f64>,
    pub noise_threshold: Option<f64>,
    pub expansion_alpha: Option<f64>,
}

impl Default for EvasiveSuperTrendParams {
    fn default() -> Self {
        Self {
            atr_length: Some(DEFAULT_ATR_LENGTH),
            base_multiplier: Some(DEFAULT_BASE_MULTIPLIER),
            noise_threshold: Some(DEFAULT_NOISE_THRESHOLD),
            expansion_alpha: Some(DEFAULT_EXPANSION_ALPHA),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvasiveSuperTrendInput<'a> {
    pub data: EvasiveSuperTrendData<'a>,
    pub params: EvasiveSuperTrendParams,
}

impl<'a> EvasiveSuperTrendInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: EvasiveSuperTrendParams) -> Self {
        Self {
            data: EvasiveSuperTrendData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: EvasiveSuperTrendParams,
    ) -> Self {
        Self {
            data: EvasiveSuperTrendData::Slices {
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
        Self::from_candles(candles, EvasiveSuperTrendParams::default())
    }

    #[inline]
    pub fn get_atr_length(&self) -> usize {
        self.params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH)
    }

    #[inline]
    pub fn get_base_multiplier(&self) -> f64 {
        self.params
            .base_multiplier
            .unwrap_or(DEFAULT_BASE_MULTIPLIER)
    }

    #[inline]
    pub fn get_noise_threshold(&self) -> f64 {
        self.params
            .noise_threshold
            .unwrap_or(DEFAULT_NOISE_THRESHOLD)
    }

    #[inline]
    pub fn get_expansion_alpha(&self) -> f64 {
        self.params
            .expansion_alpha
            .unwrap_or(DEFAULT_EXPANSION_ALPHA)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EvasiveSuperTrendBuilder {
    atr_length: Option<usize>,
    base_multiplier: Option<f64>,
    noise_threshold: Option<f64>,
    expansion_alpha: Option<f64>,
    kernel: Kernel,
}

impl Default for EvasiveSuperTrendBuilder {
    fn default() -> Self {
        Self {
            atr_length: None,
            base_multiplier: None,
            noise_threshold: None,
            expansion_alpha: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EvasiveSuperTrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn atr_length(mut self, value: usize) -> Self {
        self.atr_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn base_multiplier(mut self, value: f64) -> Self {
        self.base_multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn noise_threshold(mut self, value: f64) -> Self {
        self.noise_threshold = Some(value);
        self
    }

    #[inline(always)]
    pub fn expansion_alpha(mut self, value: f64) -> Self {
        self.expansion_alpha = Some(value);
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
    ) -> Result<EvasiveSuperTrendOutput, EvasiveSuperTrendError> {
        evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_candles(
                candles,
                EvasiveSuperTrendParams {
                    atr_length: self.atr_length,
                    base_multiplier: self.base_multiplier,
                    noise_threshold: self.noise_threshold,
                    expansion_alpha: self.expansion_alpha,
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
    ) -> Result<EvasiveSuperTrendOutput, EvasiveSuperTrendError> {
        evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_slices(
                open,
                high,
                low,
                close,
                EvasiveSuperTrendParams {
                    atr_length: self.atr_length,
                    base_multiplier: self.base_multiplier,
                    noise_threshold: self.noise_threshold,
                    expansion_alpha: self.expansion_alpha,
                },
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EvasiveSuperTrendStream, EvasiveSuperTrendError> {
        EvasiveSuperTrendStream::try_new(EvasiveSuperTrendParams {
            atr_length: self.atr_length,
            base_multiplier: self.base_multiplier,
            noise_threshold: self.noise_threshold,
            expansion_alpha: self.expansion_alpha,
        })
    }
}

#[derive(Debug, Error)]
pub enum EvasiveSuperTrendError {
    #[error("evasive_supertrend: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "evasive_supertrend: Input length mismatch: open = {open_len}, high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    InputLengthMismatch {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("evasive_supertrend: All values are NaN.")]
    AllValuesNaN,
    #[error("evasive_supertrend: Invalid atr_length: {atr_length}")]
    InvalidAtrLength { atr_length: usize },
    #[error("evasive_supertrend: Invalid base_multiplier: {base_multiplier}")]
    InvalidBaseMultiplier { base_multiplier: f64 },
    #[error("evasive_supertrend: Invalid noise_threshold: {noise_threshold}")]
    InvalidNoiseThreshold { noise_threshold: f64 },
    #[error("evasive_supertrend: Invalid expansion_alpha: {expansion_alpha}")]
    InvalidExpansionAlpha { expansion_alpha: f64 },
    #[error("evasive_supertrend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("evasive_supertrend: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "evasive_supertrend: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("evasive_supertrend: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("evasive_supertrend: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("evasive_supertrend: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone, Copy)]
struct AtrTracker {
    period: usize,
    count: usize,
    tr_sum: f64,
    prev_close: Option<f64>,
    atr: f64,
}

impl AtrTracker {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            tr_sum: 0.0,
            prev_close: None,
            atr: f64::NAN,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.tr_sum = 0.0;
        self.prev_close = None;
        self.atr = f64::NAN;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = match self.prev_close {
            Some(prev_close) => {
                let hl = high - low;
                let hc = (high - prev_close).abs();
                let lc = (low - prev_close).abs();
                hl.max(hc).max(lc)
            }
            None => high - low,
        };
        self.prev_close = Some(close);

        if self.count < self.period {
            self.count += 1;
            self.tr_sum += tr;
            if self.count == self.period {
                self.atr = self.tr_sum / self.period as f64;
                Some(self.atr)
            } else {
                None
            }
        } else {
            self.atr = ((self.atr * (self.period as f64 - 1.0)) + tr) / self.period as f64;
            Some(self.atr)
        }
    }
}

#[inline(always)]
fn is_valid_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline(always)]
fn longest_valid_run(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
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
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a EvasiveSuperTrendInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), EvasiveSuperTrendError> {
    match &input.data {
        EvasiveSuperTrendData::Candles { candles } => Ok((
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        EvasiveSuperTrendData::Slices {
            open,
            high,
            low,
            close,
        } => Ok((open, high, low, close)),
    }
}

#[inline(always)]
fn validate_params_only(
    atr_length: usize,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
) -> Result<(), EvasiveSuperTrendError> {
    if atr_length == 0 {
        return Err(EvasiveSuperTrendError::InvalidAtrLength { atr_length });
    }
    if !base_multiplier.is_finite() || base_multiplier < 0.1 {
        return Err(EvasiveSuperTrendError::InvalidBaseMultiplier { base_multiplier });
    }
    if !noise_threshold.is_finite() || noise_threshold < 0.1 {
        return Err(EvasiveSuperTrendError::InvalidNoiseThreshold { noise_threshold });
    }
    if !expansion_alpha.is_finite() || expansion_alpha < 0.0 {
        return Err(EvasiveSuperTrendError::InvalidExpansionAlpha { expansion_alpha });
    }
    Ok(())
}

#[inline(always)]
fn validate_common(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
) -> Result<bool, EvasiveSuperTrendError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(EvasiveSuperTrendError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(EvasiveSuperTrendError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    validate_params_only(
        atr_length,
        base_multiplier,
        noise_threshold,
        expansion_alpha,
    )?;
    let longest = longest_valid_run(open, high, low, close);
    if longest == 0 {
        return Err(EvasiveSuperTrendError::AllValuesNaN);
    }
    if longest < atr_length {
        return Err(EvasiveSuperTrendError::NotEnoughValidData {
            needed: atr_length,
            valid: longest,
        });
    }
    Ok(longest == open.len())
}

#[inline(always)]
fn compute_point(
    tracker: &mut AtrTracker,
    trend: &mut i8,
    band: &mut f64,
    high: f64,
    low: f64,
    close: f64,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
) -> Option<(f64, f64, f64, f64)> {
    let atr = tracker.update(high, low, close)?;
    let src = (high + low) * 0.5;
    let upper_base = src + base_multiplier * atr;
    let lower_base = src - base_multiplier * atr;
    let prev_band = if band.is_nan() {
        if *trend == 1 {
            lower_base
        } else {
            upper_base
        }
    } else {
        *band
    };
    let is_noisy = (close - prev_band).abs() < atr * noise_threshold;
    let prev_trend = *trend;
    let mut next_band;

    if prev_trend == 1 {
        next_band = if is_noisy {
            prev_band - atr * expansion_alpha
        } else {
            lower_base.max(prev_band)
        };
        if close < next_band {
            *trend = -1;
            next_band = upper_base;
        }
    } else {
        next_band = if is_noisy {
            prev_band + atr * expansion_alpha
        } else {
            upper_base.min(prev_band)
        };
        if close > next_band {
            *trend = 1;
            next_band = lower_base;
        }
    }

    *band = next_band;
    Some((
        next_band,
        *trend as f64,
        if is_noisy { 1.0 } else { 0.0 },
        if *trend != prev_trend { 1.0 } else { 0.0 },
    ))
}

fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
    band_out: &mut [f64],
    state_out: &mut [f64],
    noisy_out: &mut [f64],
    changed_out: &mut [f64],
) {
    let mut tracker = AtrTracker::new(atr_length);
    let mut trend = 1i8;
    let mut band = f64::NAN;

    for i in 0..close.len() {
        if !is_valid_ohlc(open[i], high[i], low[i], close[i]) {
            tracker.reset();
            trend = 1;
            band = f64::NAN;
            continue;
        }

        if let Some((band_value, state_value, noisy_value, changed_value)) = compute_point(
            &mut tracker,
            &mut trend,
            &mut band,
            high[i],
            low[i],
            close[i],
            base_multiplier,
            noise_threshold,
            expansion_alpha,
        ) {
            band_out[i] = band_value;
            state_out[i] = state_value;
            noisy_out[i] = noisy_value;
            changed_out[i] = changed_value;
        }
    }
}

fn compute_row_all_valid(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
    band_out: &mut [f64],
    state_out: &mut [f64],
    noisy_out: &mut [f64],
    changed_out: &mut [f64],
) {
    let mut tracker = AtrTracker::new(atr_length);
    let mut trend = 1i8;
    let mut band = f64::NAN;

    for i in 0..close.len() {
        if let Some((band_value, state_value, noisy_value, changed_value)) = compute_point(
            &mut tracker,
            &mut trend,
            &mut band,
            high[i],
            low[i],
            close[i],
            base_multiplier,
            noise_threshold,
            expansion_alpha,
        ) {
            band_out[i] = band_value;
            state_out[i] = state_value;
            noisy_out[i] = noisy_value;
            changed_out[i] = changed_value;
        }
    }
}

#[inline]
pub fn evasive_supertrend(
    input: &EvasiveSuperTrendInput,
) -> Result<EvasiveSuperTrendOutput, EvasiveSuperTrendError> {
    evasive_supertrend_with_kernel(input, Kernel::Auto)
}

pub fn evasive_supertrend_with_kernel(
    input: &EvasiveSuperTrendInput,
    kernel: Kernel,
) -> Result<EvasiveSuperTrendOutput, EvasiveSuperTrendError> {
    let (open, high, low, close) = input_slices(input)?;
    let atr_length = input.get_atr_length();
    let base_multiplier = input.get_base_multiplier();
    let noise_threshold = input.get_noise_threshold();
    let expansion_alpha = input.get_expansion_alpha();
    let all_valid = validate_common(
        open,
        high,
        low,
        close,
        atr_length,
        base_multiplier,
        noise_threshold,
        expansion_alpha,
    )?;

    let mut band = alloc_with_nan_prefix(close.len(), close.len());
    let mut state = alloc_with_nan_prefix(close.len(), close.len());
    let mut noisy = alloc_with_nan_prefix(close.len(), close.len());
    let mut changed = alloc_with_nan_prefix(close.len(), close.len());

    let _ = kernel;

    if all_valid {
        compute_row_all_valid(
            high,
            low,
            close,
            atr_length,
            base_multiplier,
            noise_threshold,
            expansion_alpha,
            &mut band,
            &mut state,
            &mut noisy,
            &mut changed,
        );
    } else {
        compute_row(
            open,
            high,
            low,
            close,
            atr_length,
            base_multiplier,
            noise_threshold,
            expansion_alpha,
            &mut band,
            &mut state,
            &mut noisy,
            &mut changed,
        );
    }

    Ok(EvasiveSuperTrendOutput {
        band,
        state,
        noisy,
        changed,
    })
}

pub fn evasive_supertrend_into_slice(
    out_band: &mut [f64],
    out_state: &mut [f64],
    out_noisy: &mut [f64],
    out_changed: &mut [f64],
    input: &EvasiveSuperTrendInput,
    kernel: Kernel,
) -> Result<(), EvasiveSuperTrendError> {
    let (open, high, low, close) = input_slices(input)?;
    let atr_length = input.get_atr_length();
    let base_multiplier = input.get_base_multiplier();
    let noise_threshold = input.get_noise_threshold();
    let expansion_alpha = input.get_expansion_alpha();
    let all_valid = validate_common(
        open,
        high,
        low,
        close,
        atr_length,
        base_multiplier,
        noise_threshold,
        expansion_alpha,
    )?;

    if out_band.len() != close.len() {
        return Err(EvasiveSuperTrendError::OutputLengthMismatch {
            expected: close.len(),
            got: out_band.len(),
        });
    }
    if out_state.len() != close.len() {
        return Err(EvasiveSuperTrendError::OutputLengthMismatch {
            expected: close.len(),
            got: out_state.len(),
        });
    }
    if out_noisy.len() != close.len() {
        return Err(EvasiveSuperTrendError::OutputLengthMismatch {
            expected: close.len(),
            got: out_noisy.len(),
        });
    }
    if out_changed.len() != close.len() {
        return Err(EvasiveSuperTrendError::OutputLengthMismatch {
            expected: close.len(),
            got: out_changed.len(),
        });
    }

    let _ = kernel;

    out_band.fill(f64::NAN);
    out_state.fill(f64::NAN);
    out_noisy.fill(f64::NAN);
    out_changed.fill(f64::NAN);
    if all_valid {
        compute_row_all_valid(
            high,
            low,
            close,
            atr_length,
            base_multiplier,
            noise_threshold,
            expansion_alpha,
            out_band,
            out_state,
            out_noisy,
            out_changed,
        );
    } else {
        compute_row(
            open,
            high,
            low,
            close,
            atr_length,
            base_multiplier,
            noise_threshold,
            expansion_alpha,
            out_band,
            out_state,
            out_noisy,
            out_changed,
        );
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn evasive_supertrend_into(
    input: &EvasiveSuperTrendInput,
    out_band: &mut [f64],
    out_state: &mut [f64],
    out_noisy: &mut [f64],
    out_changed: &mut [f64],
) -> Result<(), EvasiveSuperTrendError> {
    evasive_supertrend_into_slice(
        out_band,
        out_state,
        out_noisy,
        out_changed,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, Copy)]
pub struct EvasiveSuperTrendBatchRange {
    pub atr_length: (usize, usize, usize),
    pub base_multiplier: (f64, f64, f64),
    pub noise_threshold: (f64, f64, f64),
    pub expansion_alpha: (f64, f64, f64),
}

impl Default for EvasiveSuperTrendBatchRange {
    fn default() -> Self {
        Self {
            atr_length: (DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0),
            base_multiplier: (DEFAULT_BASE_MULTIPLIER, DEFAULT_BASE_MULTIPLIER, 0.0),
            noise_threshold: (DEFAULT_NOISE_THRESHOLD, DEFAULT_NOISE_THRESHOLD, 0.0),
            expansion_alpha: (DEFAULT_EXPANSION_ALPHA, DEFAULT_EXPANSION_ALPHA, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvasiveSuperTrendBatchOutput {
    pub band: Vec<f64>,
    pub state: Vec<f64>,
    pub noisy: Vec<f64>,
    pub changed: Vec<f64>,
    pub combos: Vec<EvasiveSuperTrendParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EvasiveSuperTrendBatchBuilder {
    range: EvasiveSuperTrendBatchRange,
    kernel: Kernel,
}

impl Default for EvasiveSuperTrendBatchBuilder {
    fn default() -> Self {
        Self {
            range: EvasiveSuperTrendBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EvasiveSuperTrendBatchBuilder {
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
    pub fn atr_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.atr_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn atr_length_static(mut self, value: usize) -> Self {
        self.range.atr_length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn base_multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.base_multiplier = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn base_multiplier_static(mut self, value: f64) -> Self {
        self.range.base_multiplier = (value, value, 0.0);
        self
    }

    #[inline(always)]
    pub fn noise_threshold_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.noise_threshold = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn noise_threshold_static(mut self, value: f64) -> Self {
        self.range.noise_threshold = (value, value, 0.0);
        self
    }

    #[inline(always)]
    pub fn expansion_alpha_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.expansion_alpha = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn expansion_alpha_static(mut self, value: f64) -> Self {
        self.range.expansion_alpha = (value, value, 0.0);
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<EvasiveSuperTrendBatchOutput, EvasiveSuperTrendError> {
        evasive_supertrend_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<EvasiveSuperTrendBatchOutput, EvasiveSuperTrendError> {
        evasive_supertrend_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
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
) -> Result<Vec<usize>, EvasiveSuperTrendError> {
    if start == 0 || end == 0 {
        return Err(EvasiveSuperTrendError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(EvasiveSuperTrendError::InvalidRange {
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
            return Err(EvasiveSuperTrendError::InvalidRange {
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
) -> Result<Vec<f64>, EvasiveSuperTrendError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(EvasiveSuperTrendError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(EvasiveSuperTrendError::InvalidRange {
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
            return Err(EvasiveSuperTrendError::InvalidRange {
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
    range: &EvasiveSuperTrendBatchRange,
) -> Result<Vec<EvasiveSuperTrendParams>, EvasiveSuperTrendError> {
    let atr_lengths = expand_usize_range(
        "atr_length",
        range.atr_length.0,
        range.atr_length.1,
        range.atr_length.2,
    )?;
    let base_multipliers = expand_f64_range(
        "base_multiplier",
        range.base_multiplier.0,
        range.base_multiplier.1,
        range.base_multiplier.2,
    )?;
    let noise_thresholds = expand_f64_range(
        "noise_threshold",
        range.noise_threshold.0,
        range.noise_threshold.1,
        range.noise_threshold.2,
    )?;
    let expansion_alphas = expand_f64_range(
        "expansion_alpha",
        range.expansion_alpha.0,
        range.expansion_alpha.1,
        range.expansion_alpha.2,
    )?;

    let mut out = Vec::new();
    for &atr_length in &atr_lengths {
        for &base_multiplier in &base_multipliers {
            for &noise_threshold in &noise_thresholds {
                for &expansion_alpha in &expansion_alphas {
                    out.push(EvasiveSuperTrendParams {
                        atr_length: Some(atr_length),
                        base_multiplier: Some(base_multiplier),
                        noise_threshold: Some(noise_threshold),
                        expansion_alpha: Some(expansion_alpha),
                    });
                }
            }
        }
    }
    Ok(out)
}

pub fn expand_grid_evasive_supertrend(
    range: &EvasiveSuperTrendBatchRange,
) -> Vec<EvasiveSuperTrendParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn evasive_supertrend_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &EvasiveSuperTrendBatchRange,
    kernel: Kernel,
) -> Result<EvasiveSuperTrendBatchOutput, EvasiveSuperTrendError> {
    evasive_supertrend_batch_inner(open, high, low, close, sweep, kernel, true)
}

pub fn evasive_supertrend_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &EvasiveSuperTrendBatchRange,
    kernel: Kernel,
) -> Result<EvasiveSuperTrendBatchOutput, EvasiveSuperTrendError> {
    evasive_supertrend_batch_inner(open, high, low, close, sweep, kernel, false)
}

pub fn evasive_supertrend_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &EvasiveSuperTrendBatchRange,
    kernel: Kernel,
) -> Result<EvasiveSuperTrendBatchOutput, EvasiveSuperTrendError> {
    evasive_supertrend_batch_inner(open, high, low, close, sweep, kernel, true)
}

fn evasive_supertrend_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &EvasiveSuperTrendBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EvasiveSuperTrendBatchOutput, EvasiveSuperTrendError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(EvasiveSuperTrendError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let max_atr_length = combos
        .iter()
        .map(|params| params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH))
        .max()
        .unwrap_or(0);
    let max_base_multiplier = combos
        .iter()
        .map(|params| params.base_multiplier.unwrap_or(DEFAULT_BASE_MULTIPLIER))
        .fold(0.0_f64, f64::max);
    let max_noise_threshold = combos
        .iter()
        .map(|params| params.noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD))
        .fold(0.0_f64, f64::max);
    let max_expansion_alpha = combos
        .iter()
        .map(|params| params.expansion_alpha.unwrap_or(DEFAULT_EXPANSION_ALPHA))
        .fold(0.0_f64, f64::max);
    validate_common(
        open,
        high,
        low,
        close,
        max_atr_length,
        max_base_multiplier,
        max_noise_threshold,
        max_expansion_alpha,
    )?;

    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| EvasiveSuperTrendError::InvalidInput {
            msg: "evasive_supertrend: rows*cols overflow in batch".to_string(),
        })?;

    let mut band = vec![f64::NAN; total];
    let mut state = vec![f64::NAN; total];
    let mut noisy = vec![f64::NAN; total];
    let mut changed = vec![f64::NAN; total];
    evasive_supertrend_batch_inner_into(
        open,
        high,
        low,
        close,
        sweep,
        kernel,
        parallel,
        &mut band,
        &mut state,
        &mut noisy,
        &mut changed,
    )?;

    Ok(EvasiveSuperTrendBatchOutput {
        band,
        state,
        noisy,
        changed,
        combos,
        rows,
        cols,
    })
}

pub fn evasive_supertrend_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &EvasiveSuperTrendBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_band: &mut [f64],
    out_state: &mut [f64],
    out_noisy: &mut [f64],
    out_changed: &mut [f64],
) -> Result<Vec<EvasiveSuperTrendParams>, EvasiveSuperTrendError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(EvasiveSuperTrendError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let max_atr_length = combos
        .iter()
        .map(|params| params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH))
        .max()
        .unwrap_or(0);
    let max_base_multiplier = combos
        .iter()
        .map(|params| params.base_multiplier.unwrap_or(DEFAULT_BASE_MULTIPLIER))
        .fold(0.0_f64, f64::max);
    let max_noise_threshold = combos
        .iter()
        .map(|params| params.noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD))
        .fold(0.0_f64, f64::max);
    let max_expansion_alpha = combos
        .iter()
        .map(|params| params.expansion_alpha.unwrap_or(DEFAULT_EXPANSION_ALPHA))
        .fold(0.0_f64, f64::max);
    let all_valid = validate_common(
        open,
        high,
        low,
        close,
        max_atr_length,
        max_base_multiplier,
        max_noise_threshold,
        max_expansion_alpha,
    )?;

    let cols = close.len();
    let total =
        combos
            .len()
            .checked_mul(cols)
            .ok_or_else(|| EvasiveSuperTrendError::InvalidInput {
                msg: "evasive_supertrend: rows*cols overflow in batch_into".to_string(),
            })?;
    if out_band.len() != total {
        return Err(EvasiveSuperTrendError::MismatchedOutputLen {
            dst_len: out_band.len(),
            expected_len: total,
        });
    }
    if out_state.len() != total {
        return Err(EvasiveSuperTrendError::MismatchedOutputLen {
            dst_len: out_state.len(),
            expected_len: total,
        });
    }
    if out_noisy.len() != total {
        return Err(EvasiveSuperTrendError::MismatchedOutputLen {
            dst_len: out_noisy.len(),
            expected_len: total,
        });
    }
    if out_changed.len() != total {
        return Err(EvasiveSuperTrendError::MismatchedOutputLen {
            dst_len: out_changed.len(),
            expected_len: total,
        });
    }

    let _ = kernel;

    let worker = |row: usize,
                  band_row: &mut [f64],
                  state_row: &mut [f64],
                  noisy_row: &mut [f64],
                  changed_row: &mut [f64]| {
        band_row.fill(f64::NAN);
        state_row.fill(f64::NAN);
        noisy_row.fill(f64::NAN);
        changed_row.fill(f64::NAN);
        let params = &combos[row];
        if all_valid {
            compute_row_all_valid(
                high,
                low,
                close,
                params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH),
                params.base_multiplier.unwrap_or(DEFAULT_BASE_MULTIPLIER),
                params.noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD),
                params.expansion_alpha.unwrap_or(DEFAULT_EXPANSION_ALPHA),
                band_row,
                state_row,
                noisy_row,
                changed_row,
            );
        } else {
            compute_row(
                open,
                high,
                low,
                close,
                params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH),
                params.base_multiplier.unwrap_or(DEFAULT_BASE_MULTIPLIER),
                params.noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD),
                params.expansion_alpha.unwrap_or(DEFAULT_EXPANSION_ALPHA),
                band_row,
                state_row,
                noisy_row,
                changed_row,
            );
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel && combos.len() > 1 {
        out_band
            .par_chunks_mut(cols)
            .zip(out_state.par_chunks_mut(cols))
            .zip(out_noisy.par_chunks_mut(cols))
            .zip(out_changed.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (((band_row, state_row), noisy_row), changed_row))| {
                worker(row, band_row, state_row, noisy_row, changed_row);
            });
    } else {
        for (row, (((band_row, state_row), noisy_row), changed_row)) in out_band
            .chunks_mut(cols)
            .zip(out_state.chunks_mut(cols))
            .zip(out_noisy.chunks_mut(cols))
            .zip(out_changed.chunks_mut(cols))
            .enumerate()
        {
            worker(row, band_row, state_row, noisy_row, changed_row);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, (((band_row, state_row), noisy_row), changed_row)) in out_band
            .chunks_mut(cols)
            .zip(out_state.chunks_mut(cols))
            .zip(out_noisy.chunks_mut(cols))
            .zip(out_changed.chunks_mut(cols))
            .enumerate()
        {
            worker(row, band_row, state_row, noisy_row, changed_row);
        }
    }

    Ok(combos)
}

#[derive(Debug, Clone)]
pub struct EvasiveSuperTrendStream {
    atr_length: usize,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
    tracker: AtrTracker,
    trend: i8,
    band: f64,
}

impl EvasiveSuperTrendStream {
    pub fn try_new(params: EvasiveSuperTrendParams) -> Result<Self, EvasiveSuperTrendError> {
        let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
        let base_multiplier = params.base_multiplier.unwrap_or(DEFAULT_BASE_MULTIPLIER);
        let noise_threshold = params.noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD);
        let expansion_alpha = params.expansion_alpha.unwrap_or(DEFAULT_EXPANSION_ALPHA);
        validate_params_only(
            atr_length,
            base_multiplier,
            noise_threshold,
            expansion_alpha,
        )?;
        Ok(Self {
            atr_length,
            base_multiplier,
            noise_threshold,
            expansion_alpha,
            tracker: AtrTracker::new(atr_length),
            trend: 1,
            band: f64::NAN,
        })
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.tracker.reset();
        self.trend = 1;
        self.band = f64::NAN;
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        if !is_valid_ohlc(open, high, low, close) {
            self.reset();
            return None;
        }
        compute_point(
            &mut self.tracker,
            &mut self.trend,
            &mut self.band,
            high,
            low,
            close,
            self.base_multiplier,
            self.noise_threshold,
            self.expansion_alpha,
        )
    }

    #[inline(always)]
    pub fn atr_length(&self) -> usize {
        self.atr_length
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "evasive_supertrend",
    signature = (open, high, low, close, atr_length=DEFAULT_ATR_LENGTH, base_multiplier=DEFAULT_BASE_MULTIPLIER, noise_threshold=DEFAULT_NOISE_THRESHOLD, expansion_alpha=DEFAULT_EXPANSION_ALPHA, kernel=None)
)]
pub fn evasive_supertrend_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    atr_length: usize,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
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
    let input = EvasiveSuperTrendInput::from_slices(
        open,
        high,
        low,
        close,
        EvasiveSuperTrendParams {
            atr_length: Some(atr_length),
            base_multiplier: Some(base_multiplier),
            noise_threshold: Some(noise_threshold),
            expansion_alpha: Some(expansion_alpha),
        },
    );
    let out = py
        .allow_threads(|| evasive_supertrend_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.band.into_pyarray(py),
        out.state.into_pyarray(py),
        out.noisy.into_pyarray(py),
        out.changed.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "evasive_supertrend_batch",
    signature = (open, high, low, close, atr_length_range=(DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0), base_multiplier_range=(DEFAULT_BASE_MULTIPLIER, DEFAULT_BASE_MULTIPLIER, 0.0), noise_threshold_range=(DEFAULT_NOISE_THRESHOLD, DEFAULT_NOISE_THRESHOLD, 0.0), expansion_alpha_range=(DEFAULT_EXPANSION_ALPHA, DEFAULT_EXPANSION_ALPHA, 0.0), kernel=None)
)]
pub fn evasive_supertrend_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    atr_length_range: (usize, usize, usize),
    base_multiplier_range: (f64, f64, f64),
    noise_threshold_range: (f64, f64, f64),
    expansion_alpha_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let output = py
        .allow_threads(|| {
            evasive_supertrend_batch_with_kernel(
                open,
                high,
                low,
                close,
                &EvasiveSuperTrendBatchRange {
                    atr_length: atr_length_range,
                    base_multiplier: base_multiplier_range,
                    noise_threshold: noise_threshold_range,
                    expansion_alpha: expansion_alpha_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "band",
        output
            .band
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "state",
        output
            .state
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "noisy",
        output
            .noisy
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "changed",
        output
            .changed
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "atr_lengths",
        output
            .combos
            .iter()
            .map(|params| params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "base_multipliers",
        output
            .combos
            .iter()
            .map(|params| params.base_multiplier.unwrap_or(DEFAULT_BASE_MULTIPLIER))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "noise_thresholds",
        output
            .combos
            .iter()
            .map(|params| params.noise_threshold.unwrap_or(DEFAULT_NOISE_THRESHOLD))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "expansion_alphas",
        output
            .combos
            .iter()
            .map(|params| params.expansion_alpha.unwrap_or(DEFAULT_EXPANSION_ALPHA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "EvasiveSuperTrendStream")]
pub struct EvasiveSuperTrendStreamPy {
    stream: EvasiveSuperTrendStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EvasiveSuperTrendStreamPy {
    #[new]
    #[pyo3(signature = (atr_length=DEFAULT_ATR_LENGTH, base_multiplier=DEFAULT_BASE_MULTIPLIER, noise_threshold=DEFAULT_NOISE_THRESHOLD, expansion_alpha=DEFAULT_EXPANSION_ALPHA))]
    fn new(
        atr_length: usize,
        base_multiplier: f64,
        noise_threshold: f64,
        expansion_alpha: f64,
    ) -> PyResult<Self> {
        let stream = EvasiveSuperTrendStream::try_new(EvasiveSuperTrendParams {
            atr_length: Some(atr_length),
            base_multiplier: Some(base_multiplier),
            noise_threshold: Some(noise_threshold),
            expansion_alpha: Some(expansion_alpha),
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
pub fn register_evasive_supertrend_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evasive_supertrend_py, m)?)?;
    m.add_function(wrap_pyfunction!(evasive_supertrend_batch_py, m)?)?;
    m.add_class::<EvasiveSuperTrendStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvasiveSuperTrendBatchConfig {
    pub atr_length_range: Vec<usize>,
    pub base_multiplier_range: Vec<f64>,
    pub noise_threshold_range: Vec<f64>,
    pub expansion_alpha_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = evasive_supertrend_js)]
pub fn evasive_supertrend_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
) -> Result<JsValue, JsValue> {
    let input = EvasiveSuperTrendInput::from_slices(
        open,
        high,
        low,
        close,
        EvasiveSuperTrendParams {
            atr_length: Some(atr_length),
            base_multiplier: Some(base_multiplier),
            noise_threshold: Some(noise_threshold),
            expansion_alpha: Some(expansion_alpha),
        },
    );
    let out = evasive_supertrend_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("band"),
        &serde_wasm_bindgen::to_value(&out.band).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("state"),
        &serde_wasm_bindgen::to_value(&out.state).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("noisy"),
        &serde_wasm_bindgen::to_value(&out.noisy).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("changed"),
        &serde_wasm_bindgen::to_value(&out.changed).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = evasive_supertrend_batch_js)]
pub fn evasive_supertrend_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EvasiveSuperTrendBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.atr_length_range.len() != 3
        || config.base_multiplier_range.len() != 3
        || config.noise_threshold_range.len() != 3
        || config.expansion_alpha_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = evasive_supertrend_batch_with_kernel(
        open,
        high,
        low,
        close,
        &EvasiveSuperTrendBatchRange {
            atr_length: (
                config.atr_length_range[0],
                config.atr_length_range[1],
                config.atr_length_range[2],
            ),
            base_multiplier: (
                config.base_multiplier_range[0],
                config.base_multiplier_range[1],
                config.base_multiplier_range[2],
            ),
            noise_threshold: (
                config.noise_threshold_range[0],
                config.noise_threshold_range[1],
                config.noise_threshold_range[2],
            ),
            expansion_alpha: (
                config.expansion_alpha_range[0],
                config.expansion_alpha_range[1],
                config.expansion_alpha_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("band"),
        &serde_wasm_bindgen::to_value(&out.band).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("state"),
        &serde_wasm_bindgen::to_value(&out.state).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("noisy"),
        &serde_wasm_bindgen::to_value(&out.noisy).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("changed"),
        &serde_wasm_bindgen::to_value(&out.changed).unwrap(),
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
pub fn evasive_supertrend_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(4 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn evasive_supertrend_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 4 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn evasive_supertrend_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    atr_length: usize,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to evasive_supertrend_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 4 * len);
        let (band, tail) = out.split_at_mut(len);
        let (state, tail) = tail.split_at_mut(len);
        let (noisy, changed) = tail.split_at_mut(len);
        let input = EvasiveSuperTrendInput::from_slices(
            open,
            high,
            low,
            close,
            EvasiveSuperTrendParams {
                atr_length: Some(atr_length),
                base_multiplier: Some(base_multiplier),
                noise_threshold: Some(noise_threshold),
                expansion_alpha: Some(expansion_alpha),
            },
        );
        evasive_supertrend_into_slice(band, state, noisy, changed, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn evasive_supertrend_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    atr_length_start: usize,
    atr_length_end: usize,
    atr_length_step: usize,
    base_multiplier_start: f64,
    base_multiplier_end: f64,
    base_multiplier_step: f64,
    noise_threshold_start: f64,
    noise_threshold_end: f64,
    noise_threshold_step: f64,
    expansion_alpha_start: f64,
    expansion_alpha_end: f64,
    expansion_alpha_step: f64,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to evasive_supertrend_batch_into",
        ));
    }

    let sweep = EvasiveSuperTrendBatchRange {
        atr_length: (atr_length_start, atr_length_end, atr_length_step),
        base_multiplier: (
            base_multiplier_start,
            base_multiplier_end,
            base_multiplier_step,
        ),
        noise_threshold: (
            noise_threshold_start,
            noise_threshold_end,
            noise_threshold_step,
        ),
        expansion_alpha: (
            expansion_alpha_start,
            expansion_alpha_end,
            expansion_alpha_step,
        ),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let split = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in evasive_supertrend_batch_into"))?;
    let total = split.checked_mul(4).ok_or_else(|| {
        JsValue::from_str("4*rows*cols overflow in evasive_supertrend_batch_into")
    })?;

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (band, tail) = out.split_at_mut(split);
        let (state, tail) = tail.split_at_mut(split);
        let (noisy, changed) = tail.split_at_mut(split);
        evasive_supertrend_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            Kernel::Auto,
            false,
            band,
            state,
            noisy,
            changed,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn evasive_supertrend_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    base_multiplier: f64,
    noise_threshold: f64,
    expansion_alpha: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = evasive_supertrend_js(
        open,
        high,
        low,
        close,
        atr_length,
        base_multiplier,
        noise_threshold,
        expansion_alpha,
    )?;
    crate::write_wasm_object_f64_outputs("evasive_supertrend_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn evasive_supertrend_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = evasive_supertrend_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "evasive_supertrend_batch_output_into_js",
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

    fn assert_vec_eq_nan(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            assert!(
                (a - e).abs() <= 1e-12,
                "mismatch at {idx}: actual={a}, expected={e}"
            );
        }
    }

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + ((i as f64) * 0.19).sin() * 2.5 + (i as f64) * 0.03;
            let body = ((i as f64) * 0.13).cos() * 0.4;
            let o = base - body;
            let c = base + body;
            let h = o.max(c) + 0.8 + ((i as f64) * 0.07).sin().abs();
            let l = o.min(c) - 0.8 - ((i as f64) * 0.11).cos().abs() * 0.6;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
        }
        (open, high, low, close)
    }

    #[test]
    fn evasive_supertrend_output_contract() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(256);
        let out = evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                EvasiveSuperTrendParams::default(),
            ),
            Kernel::Scalar,
        )?;
        assert_eq!(out.band.len(), close.len());
        assert_eq!(out.state.len(), close.len());
        assert_eq!(out.noisy.len(), close.len());
        assert_eq!(out.changed.len(), close.len());
        assert!(out.band.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn evasive_supertrend_exact_small_case() -> Result<(), Box<dyn Error>> {
        let open = vec![10.0, 11.0, 12.0, 13.0];
        let high = vec![11.0, 12.0, 13.0, 14.0];
        let low = vec![9.0, 10.0, 11.0, 12.0];
        let close = vec![10.0, 11.0, 12.0, 13.0];
        let out = evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                EvasiveSuperTrendParams {
                    atr_length: Some(2),
                    base_multiplier: Some(1.0),
                    noise_threshold: Some(1.0),
                    expansion_alpha: Some(0.5),
                },
            ),
            Kernel::Scalar,
        )?;
        assert_vec_eq_nan(&out.band, &[f64::NAN, 9.0, 10.0, 11.0]);
        assert_vec_eq_nan(&out.state, &[f64::NAN, 1.0, 1.0, 1.0]);
        assert_vec_eq_nan(&out.noisy, &[f64::NAN, 0.0, 0.0, 0.0]);
        assert_vec_eq_nan(&out.changed, &[f64::NAN, 0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn evasive_supertrend_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(200);
        let input = EvasiveSuperTrendInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            EvasiveSuperTrendParams::default(),
        );
        let baseline = evasive_supertrend_with_kernel(&input, Kernel::Scalar)?;
        let mut band = vec![0.0; close.len()];
        let mut state = vec![0.0; close.len()];
        let mut noisy = vec![0.0; close.len()];
        let mut changed = vec![0.0; close.len()];
        evasive_supertrend_into(&input, &mut band, &mut state, &mut noisy, &mut changed)?;
        assert_vec_eq_nan(&band, &baseline.band);
        assert_vec_eq_nan(&state, &baseline.state);
        assert_vec_eq_nan(&noisy, &baseline.noisy);
        assert_vec_eq_nan(&changed, &baseline.changed);
        Ok(())
    }

    #[test]
    fn evasive_supertrend_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(180);
        let single = evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                EvasiveSuperTrendParams::default(),
            ),
            Kernel::Scalar,
        )?;
        let batch = evasive_supertrend_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &EvasiveSuperTrendBatchRange::default(),
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_vec_eq_nan(&batch.band[..close.len()], &single.band);
        assert_vec_eq_nan(&batch.state[..close.len()], &single.state);
        assert_vec_eq_nan(&batch.noisy[..close.len()], &single.noisy);
        assert_vec_eq_nan(&batch.changed[..close.len()], &single.changed);
        Ok(())
    }

    #[test]
    fn evasive_supertrend_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(220);
        let batch = evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                EvasiveSuperTrendParams::default(),
            ),
            Kernel::Scalar,
        )?;
        let mut stream = EvasiveSuperTrendBuilder::new().into_stream()?;
        let mut band = Vec::with_capacity(close.len());
        let mut state = Vec::with_capacity(close.len());
        let mut noisy = Vec::with_capacity(close.len());
        let mut changed = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            match stream.update(open[i], high[i], low[i], close[i]) {
                Some((b, s, n, c)) => {
                    band.push(b);
                    state.push(s);
                    noisy.push(n);
                    changed.push(c);
                }
                None => {
                    band.push(f64::NAN);
                    state.push(f64::NAN);
                    noisy.push(f64::NAN);
                    changed.push(f64::NAN);
                }
            }
        }
        assert_vec_eq_nan(&band, &batch.band);
        assert_vec_eq_nan(&state, &batch.state);
        assert_vec_eq_nan(&noisy, &batch.noisy);
        assert_vec_eq_nan(&changed, &batch.changed);
        Ok(())
    }

    #[test]
    fn evasive_supertrend_rejects_invalid_params() {
        let (open, high, low, close) = sample_ohlc(32);
        let err = evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                EvasiveSuperTrendParams {
                    atr_length: Some(0),
                    base_multiplier: Some(3.0),
                    noise_threshold: Some(1.0),
                    expansion_alpha: Some(0.5),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvasiveSuperTrendError::InvalidAtrLength { .. }
        ));

        let err = evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                EvasiveSuperTrendParams {
                    atr_length: Some(10),
                    base_multiplier: Some(0.0),
                    noise_threshold: Some(1.0),
                    expansion_alpha: Some(0.5),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvasiveSuperTrendError::InvalidBaseMultiplier { .. }
        ));

        let err = evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                EvasiveSuperTrendParams {
                    atr_length: Some(10),
                    base_multiplier: Some(3.0),
                    noise_threshold: Some(0.0),
                    expansion_alpha: Some(0.5),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvasiveSuperTrendError::InvalidNoiseThreshold { .. }
        ));

        let err = evasive_supertrend_with_kernel(
            &EvasiveSuperTrendInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                EvasiveSuperTrendParams {
                    atr_length: Some(10),
                    base_multiplier: Some(3.0),
                    noise_threshold: Some(1.0),
                    expansion_alpha: Some(-0.1),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvasiveSuperTrendError::InvalidExpansionAlpha { .. }
        ));
    }

    #[test]
    fn evasive_supertrend_dispatch_compute_returns_expected_outputs() -> Result<(), Box<dyn Error>>
    {
        let (open, high, low, close) = sample_ohlc(192);
        let params = [
            ParamKV {
                key: "atr_length",
                value: ParamValue::Int(10),
            },
            ParamKV {
                key: "base_multiplier",
                value: ParamValue::Float(3.0),
            },
            ParamKV {
                key: "noise_threshold",
                value: ParamValue::Float(1.0),
            },
            ParamKV {
                key: "expansion_alpha",
                value: ParamValue::Float(0.5),
            },
        ];

        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "evasive_supertrend",
            output_id: Some("band"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &params,
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(out.output_id, "band");
        match out.series {
            IndicatorSeries::F64(values) => assert_eq!(values.len(), close.len()),
            other => panic!("expected f64 series, got {:?}", other),
        }

        let state_out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "evasive_supertrend",
            output_id: Some("state"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &params,
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(state_out.output_id, "state");
        Ok(())
    }
}
