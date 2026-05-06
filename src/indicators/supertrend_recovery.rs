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
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_recovery_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    multiplier: f64,
    alpha_percent: f64,
    threshold_atr: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = supertrend_recovery_js(
        high,
        low,
        close,
        atr_length,
        multiplier,
        alpha_percent,
        threshold_atr,
    )?;
    crate::write_wasm_object_f64_outputs("supertrend_recovery_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_recovery_batch_unified_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = supertrend_recovery_batch_unified_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "supertrend_recovery_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_ATR_LENGTH: usize = 10;
const DEFAULT_MULTIPLIER: f64 = 3.0;
const DEFAULT_ALPHA_PERCENT: f64 = 5.0;
const DEFAULT_THRESHOLD_ATR: f64 = 1.0;
const DEFAULT_TREND: i8 = 1;
const MIN_ALPHA_PERCENT: f64 = 0.1;
const MAX_ALPHA_PERCENT: f64 = 100.0;
const MIN_MULTIPLIER: f64 = 0.1;

#[inline(always)]
fn high_source(candles: &Candles) -> &[f64] {
    &candles.high
}

#[inline(always)]
fn low_source(candles: &Candles) -> &[f64] {
    &candles.low
}

#[inline(always)]
fn close_source(candles: &Candles) -> &[f64] {
    &candles.close
}

#[inline(always)]
fn hl2(high: f64, low: f64) -> f64 {
    0.5 * (high + low)
}

#[inline(always)]
fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
    (high - low)
        .max((high - prev_close).abs())
        .max((low - prev_close).abs())
}

#[derive(Debug, Clone)]
pub enum SuperTrendRecoveryData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct SuperTrendRecoveryOutput {
    pub band: Vec<f64>,
    pub switch_price: Vec<f64>,
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct SuperTrendRecoveryParams {
    pub atr_length: Option<usize>,
    pub multiplier: Option<f64>,
    pub alpha_percent: Option<f64>,
    pub threshold_atr: Option<f64>,
}

impl Default for SuperTrendRecoveryParams {
    fn default() -> Self {
        Self {
            atr_length: Some(DEFAULT_ATR_LENGTH),
            multiplier: Some(DEFAULT_MULTIPLIER),
            alpha_percent: Some(DEFAULT_ALPHA_PERCENT),
            threshold_atr: Some(DEFAULT_THRESHOLD_ATR),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendRecoveryInput<'a> {
    pub data: SuperTrendRecoveryData<'a>,
    pub params: SuperTrendRecoveryParams,
}

impl<'a> SuperTrendRecoveryInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles, params: SuperTrendRecoveryParams) -> Self {
        Self {
            data: SuperTrendRecoveryData::Candles { candles },
            params,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: SuperTrendRecoveryParams,
    ) -> Self {
        Self {
            data: SuperTrendRecoveryData::Slices { high, low, close },
            params,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, SuperTrendRecoveryParams::default())
    }

    #[inline(always)]
    pub fn get_atr_length(&self) -> usize {
        self.params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH)
    }

    #[inline(always)]
    pub fn get_multiplier(&self) -> f64 {
        self.params.multiplier.unwrap_or(DEFAULT_MULTIPLIER)
    }

    #[inline(always)]
    pub fn get_alpha_percent(&self) -> f64 {
        self.params.alpha_percent.unwrap_or(DEFAULT_ALPHA_PERCENT)
    }

    #[inline(always)]
    pub fn get_threshold_atr(&self) -> f64 {
        self.params.threshold_atr.unwrap_or(DEFAULT_THRESHOLD_ATR)
    }

    #[inline(always)]
    fn as_hlc(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            SuperTrendRecoveryData::Candles { candles } => (
                high_source(candles),
                low_source(candles),
                close_source(candles),
            ),
            SuperTrendRecoveryData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

impl<'a> AsRef<[f64]> for SuperTrendRecoveryInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        self.as_hlc().2
    }
}

#[derive(Clone, Debug)]
pub struct SuperTrendRecoveryBuilder {
    atr_length: Option<usize>,
    multiplier: Option<f64>,
    alpha_percent: Option<f64>,
    threshold_atr: Option<f64>,
    kernel: Kernel,
}

impl Default for SuperTrendRecoveryBuilder {
    fn default() -> Self {
        Self {
            atr_length: None,
            multiplier: None,
            alpha_percent: None,
            threshold_atr: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SuperTrendRecoveryBuilder {
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
    pub fn multiplier(mut self, value: f64) -> Self {
        self.multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn alpha_percent(mut self, value: f64) -> Self {
        self.alpha_percent = Some(value);
        self
    }

    #[inline(always)]
    pub fn threshold_atr(mut self, value: f64) -> Self {
        self.threshold_atr = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> SuperTrendRecoveryParams {
        SuperTrendRecoveryParams {
            atr_length: self.atr_length,
            multiplier: self.multiplier,
            alpha_percent: self.alpha_percent,
            threshold_atr: self.threshold_atr,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<SuperTrendRecoveryOutput, SuperTrendRecoveryError> {
        let kernel = self.kernel;
        let params = self.params();
        supertrend_recovery_with_kernel(
            &SuperTrendRecoveryInput::from_candles(candles, params),
            kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<SuperTrendRecoveryOutput, SuperTrendRecoveryError> {
        let kernel = self.kernel;
        let params = self.params();
        supertrend_recovery_with_kernel(
            &SuperTrendRecoveryInput::from_slices(high, low, close, params),
            kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SuperTrendRecoveryStream, SuperTrendRecoveryError> {
        SuperTrendRecoveryStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum SuperTrendRecoveryError {
    #[error("supertrend_recovery: input data slice is empty.")]
    EmptyInputData,
    #[error("supertrend_recovery: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "supertrend_recovery: inconsistent data lengths - high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    DataLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "supertrend_recovery: invalid period: atr_length = {atr_length}, data length = {data_len}"
    )]
    InvalidPeriod { atr_length: usize, data_len: usize },
    #[error("supertrend_recovery: invalid multiplier: {multiplier}")]
    InvalidMultiplier { multiplier: f64 },
    #[error("supertrend_recovery: invalid alpha_percent: {alpha_percent}")]
    InvalidAlphaPercent { alpha_percent: f64 },
    #[error("supertrend_recovery: invalid threshold_atr: {threshold_atr}")]
    InvalidThresholdAtr { threshold_atr: f64 },
    #[error("supertrend_recovery: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("supertrend_recovery: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "supertrend_recovery: invalid range for {axis}: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        axis: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("supertrend_recovery: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    atr_length: usize,
    multiplier: f64,
    alpha: f64,
    threshold_atr: f64,
    warmup: usize,
}

#[inline(always)]
fn normalize_single_kernel(_kernel: Kernel) -> Kernel {
    Kernel::Scalar
}

#[inline(always)]
fn validate_params(
    atr_length: usize,
    multiplier: f64,
    alpha_percent: f64,
    threshold_atr: f64,
    data_len: usize,
) -> Result<(), SuperTrendRecoveryError> {
    if atr_length == 0 || atr_length > data_len {
        return Err(SuperTrendRecoveryError::InvalidPeriod {
            atr_length,
            data_len,
        });
    }
    if !multiplier.is_finite() || multiplier < MIN_MULTIPLIER {
        return Err(SuperTrendRecoveryError::InvalidMultiplier { multiplier });
    }
    if !alpha_percent.is_finite()
        || !(MIN_ALPHA_PERCENT..=MAX_ALPHA_PERCENT).contains(&alpha_percent)
    {
        return Err(SuperTrendRecoveryError::InvalidAlphaPercent { alpha_percent });
    }
    if !threshold_atr.is_finite() || threshold_atr < 0.0 {
        return Err(SuperTrendRecoveryError::InvalidThresholdAtr { threshold_atr });
    }
    Ok(())
}

#[inline(always)]
fn analyze_valid_segments(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(usize, usize), SuperTrendRecoveryError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(SuperTrendRecoveryError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(SuperTrendRecoveryError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let mut first_valid = None;
    let mut max_run = 0usize;
    let mut run = 0usize;

    for i in 0..close.len() {
        let valid = high[i].is_finite() && low[i].is_finite() && close[i].is_finite();
        if valid {
            if first_valid.is_none() {
                first_valid = Some(i);
            }
            run += 1;
            if run > max_run {
                max_run = run;
            }
        } else {
            run = 0;
        }
    }

    match first_valid {
        Some(idx) => Ok((idx, max_run)),
        None => Err(SuperTrendRecoveryError::AllValuesNaN),
    }
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a SuperTrendRecoveryInput<'a>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, SuperTrendRecoveryError> {
    let _chosen = normalize_single_kernel(kernel);
    let (high, low, close) = input.as_hlc();
    let atr_length = input.get_atr_length();
    let multiplier = input.get_multiplier();
    let alpha_percent = input.get_alpha_percent();
    let threshold_atr = input.get_threshold_atr();
    validate_params(
        atr_length,
        multiplier,
        alpha_percent,
        threshold_atr,
        close.len(),
    )?;

    let (first_valid, max_run) = analyze_valid_segments(high, low, close)?;
    if max_run < atr_length {
        return Err(SuperTrendRecoveryError::NotEnoughValidData {
            needed: atr_length,
            valid: max_run,
        });
    }

    Ok(PreparedInput {
        high,
        low,
        close,
        atr_length,
        multiplier,
        alpha: alpha_percent * 0.01,
        threshold_atr,
        warmup: first_valid + atr_length - 1,
    })
}

#[derive(Clone, Debug)]
struct AtrState {
    length: usize,
    count: usize,
    sum: f64,
    value: f64,
}

impl AtrState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            length,
            count: 0,
            sum: 0.0,
            value: f64::NAN,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = f64::NAN;
    }

    #[inline(always)]
    fn update(&mut self, tr: f64) -> Option<f64> {
        if self.count < self.length {
            self.count += 1;
            self.sum += tr;
            if self.count == self.length {
                self.value = self.sum / self.length as f64;
                Some(self.value)
            } else {
                None
            }
        } else {
            self.value = ((self.value * (self.length as f64 - 1.0)) + tr) / self.length as f64;
            Some(self.value)
        }
    }
}

#[derive(Clone, Debug)]
struct SuperTrendRecoveryState {
    atr: AtrState,
    multiplier: f64,
    alpha: f64,
    threshold_atr: f64,
    prev_close: f64,
    band: f64,
    switch_price: f64,
    trend: i8,
}

impl SuperTrendRecoveryState {
    #[inline(always)]
    fn new(atr_length: usize, multiplier: f64, alpha: f64, threshold_atr: f64) -> Self {
        Self {
            atr: AtrState::new(atr_length),
            multiplier,
            alpha,
            threshold_atr,
            prev_close: f64::NAN,
            band: f64::NAN,
            switch_price: f64::NAN,
            trend: DEFAULT_TREND,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.atr.reset();
        self.prev_close = f64::NAN;
        self.band = f64::NAN;
        self.switch_price = f64::NAN;
        self.trend = DEFAULT_TREND;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.reset();
            return None;
        }

        if !self.switch_price.is_finite() {
            self.switch_price = close;
        }

        let tr = if self.prev_close.is_finite() {
            true_range(high, low, self.prev_close)
        } else {
            high - low
        };
        self.prev_close = close;

        let atr = self.atr.update(tr)?;
        let src = hl2(high, low);
        let upper_base = src + self.multiplier * atr;
        let lower_base = src - self.multiplier * atr;
        let deviation = self.threshold_atr * atr;
        let is_at_loss = (self.trend == 1 && (self.switch_price - close) > deviation)
            || (self.trend == -1 && (close - self.switch_price) > deviation);
        let prev_band = if self.band.is_finite() {
            self.band
        } else if self.trend == 1 {
            lower_base
        } else {
            upper_base
        };

        let mut changed = 0.0;

        if self.trend == 1 {
            let target_band = if is_at_loss {
                self.alpha.mul_add(close, (1.0 - self.alpha) * prev_band)
            } else {
                lower_base
            };
            self.band = target_band.max(prev_band);
            if close < self.band {
                self.trend = -1;
                self.band = upper_base;
                self.switch_price = close;
                changed = 1.0;
            }
        } else {
            let target_band = if is_at_loss {
                self.alpha.mul_add(close, (1.0 - self.alpha) * prev_band)
            } else {
                upper_base
            };
            self.band = target_band.min(prev_band);
            if close > self.band {
                self.trend = 1;
                self.band = lower_base;
                self.switch_price = close;
                changed = 1.0;
            }
        }

        Some((self.band, self.switch_price, self.trend as f64, changed))
    }
}

#[derive(Clone, Debug)]
pub struct SuperTrendRecoveryStream {
    params: SuperTrendRecoveryParams,
    state: SuperTrendRecoveryState,
}

impl SuperTrendRecoveryStream {
    #[inline(always)]
    pub fn try_new(params: SuperTrendRecoveryParams) -> Result<Self, SuperTrendRecoveryError> {
        let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
        let multiplier = params.multiplier.unwrap_or(DEFAULT_MULTIPLIER);
        let alpha_percent = params.alpha_percent.unwrap_or(DEFAULT_ALPHA_PERCENT);
        let threshold_atr = params.threshold_atr.unwrap_or(DEFAULT_THRESHOLD_ATR);
        validate_params(
            atr_length,
            multiplier,
            alpha_percent,
            threshold_atr,
            usize::MAX,
        )?;
        Ok(Self {
            state: SuperTrendRecoveryState::new(
                atr_length,
                multiplier,
                alpha_percent * 0.01,
                threshold_atr,
            ),
            params,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        self.state.update(high, low, close)
    }

    #[inline(always)]
    pub fn params(&self) -> &SuperTrendRecoveryParams {
        &self.params
    }
}

#[derive(Clone, Debug)]
pub struct SuperTrendRecoveryBatchRange {
    pub atr_length: (usize, usize, usize),
    pub multiplier: (f64, f64, f64),
    pub alpha_percent: (f64, f64, f64),
    pub threshold_atr: (f64, f64, f64),
}

impl Default for SuperTrendRecoveryBatchRange {
    fn default() -> Self {
        Self {
            atr_length: (DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0),
            multiplier: (DEFAULT_MULTIPLIER, DEFAULT_MULTIPLIER, 0.0),
            alpha_percent: (DEFAULT_ALPHA_PERCENT, DEFAULT_ALPHA_PERCENT, 0.0),
            threshold_atr: (DEFAULT_THRESHOLD_ATR, DEFAULT_THRESHOLD_ATR, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SuperTrendRecoveryBatchBuilder {
    range: SuperTrendRecoveryBatchRange,
    kernel: Kernel,
}

#[derive(Clone, Debug)]
pub struct SuperTrendRecoveryBatchOutput {
    pub band: Vec<f64>,
    pub switch_price: Vec<f64>,
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
    pub combos: Vec<SuperTrendRecoveryParams>,
    pub rows: usize,
    pub cols: usize,
}

impl SuperTrendRecoveryBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn atr_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.atr_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.multiplier = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn alpha_percent_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.alpha_percent = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn threshold_atr_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.threshold_atr = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<SuperTrendRecoveryBatchOutput, SuperTrendRecoveryError> {
        supertrend_recovery_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<SuperTrendRecoveryBatchOutput, SuperTrendRecoveryError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close)
    }
}

#[inline(always)]
fn compute_row(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    multiplier: f64,
    alpha_percent: f64,
    threshold_atr: f64,
    band_out: &mut [f64],
    switch_price_out: &mut [f64],
    trend_out: &mut [f64],
    changed_out: &mut [f64],
) -> Result<(), SuperTrendRecoveryError> {
    let len = close.len();
    if band_out.len() != len
        || switch_price_out.len() != len
        || trend_out.len() != len
        || changed_out.len() != len
    {
        return Err(SuperTrendRecoveryError::OutputLengthMismatch {
            expected: len,
            got: band_out
                .len()
                .max(switch_price_out.len())
                .max(trend_out.len())
                .max(changed_out.len()),
        });
    }

    let mut state =
        SuperTrendRecoveryState::new(atr_length, multiplier, alpha_percent * 0.01, threshold_atr);

    for i in 0..len {
        if let Some((band, switch_price, trend, changed)) = state.update(high[i], low[i], close[i])
        {
            band_out[i] = band;
            switch_price_out[i] = switch_price;
            trend_out[i] = trend;
            changed_out[i] = changed;
        } else {
            band_out[i] = f64::NAN;
            switch_price_out[i] = f64::NAN;
            trend_out[i] = f64::NAN;
            changed_out[i] = f64::NAN;
        }
    }

    Ok(())
}

#[inline]
pub fn supertrend_recovery(
    input: &SuperTrendRecoveryInput,
) -> Result<SuperTrendRecoveryOutput, SuperTrendRecoveryError> {
    supertrend_recovery_with_kernel(input, Kernel::Auto)
}

pub fn supertrend_recovery_with_kernel(
    input: &SuperTrendRecoveryInput,
    kernel: Kernel,
) -> Result<SuperTrendRecoveryOutput, SuperTrendRecoveryError> {
    let prepared = prepare_input(input, kernel)?;
    let len = prepared.close.len();
    let mut band = alloc_with_nan_prefix(len, prepared.warmup);
    let mut switch_price = alloc_with_nan_prefix(len, prepared.warmup);
    let mut trend = alloc_with_nan_prefix(len, prepared.warmup);
    let mut changed = alloc_with_nan_prefix(len, prepared.warmup);
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.atr_length,
        prepared.multiplier,
        prepared.alpha / 0.01,
        prepared.threshold_atr,
        &mut band,
        &mut switch_price,
        &mut trend,
        &mut changed,
    )?;
    Ok(SuperTrendRecoveryOutput {
        band,
        switch_price,
        trend,
        changed,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn supertrend_recovery_into(
    band_out: &mut [f64],
    switch_price_out: &mut [f64],
    trend_out: &mut [f64],
    changed_out: &mut [f64],
    input: &SuperTrendRecoveryInput,
) -> Result<(), SuperTrendRecoveryError> {
    supertrend_recovery_into_slice(
        band_out,
        switch_price_out,
        trend_out,
        changed_out,
        input,
        Kernel::Auto,
    )
}

pub fn supertrend_recovery_into_slice(
    band_out: &mut [f64],
    switch_price_out: &mut [f64],
    trend_out: &mut [f64],
    changed_out: &mut [f64],
    input: &SuperTrendRecoveryInput,
    kernel: Kernel,
) -> Result<(), SuperTrendRecoveryError> {
    let prepared = prepare_input(input, kernel)?;
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.atr_length,
        prepared.multiplier,
        prepared.alpha / 0.01,
        prepared.threshold_atr,
        band_out,
        switch_price_out,
        trend_out,
        changed_out,
    )
}

#[inline(always)]
pub fn expand_grid(
    sweep: &SuperTrendRecoveryBatchRange,
) -> Result<Vec<SuperTrendRecoveryParams>, SuperTrendRecoveryError> {
    fn axis_usize(
        axis: &'static str,
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, SuperTrendRecoveryError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            while value <= end {
                out.push(value);
                match value.checked_add(step) {
                    Some(next) => value = next,
                    None => break,
                }
            }
        } else {
            let mut value = start as isize;
            let stop = end as isize;
            let stride = step as isize;
            while value >= stop {
                out.push(value as usize);
                value -= stride;
            }
        }
        if out.is_empty() {
            return Err(SuperTrendRecoveryError::InvalidRange {
                axis,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    fn axis_float(
        axis: &'static str,
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, SuperTrendRecoveryError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(SuperTrendRecoveryError::InvalidRange {
                axis,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        if step == 0.0 || start == end {
            return Ok(vec![start]);
        }
        if step < 0.0 {
            return Err(SuperTrendRecoveryError::InvalidRange {
                axis,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        let mut out = Vec::new();
        let eps = step.abs() * 1e-9 + 1e-12;
        if start < end {
            let mut value = start;
            while value <= end + eps {
                out.push(value);
                value += step;
            }
        } else {
            let mut value = start;
            while value + eps >= end {
                out.push(value);
                value -= step;
            }
        }
        if out.is_empty() {
            return Err(SuperTrendRecoveryError::InvalidRange {
                axis,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let atr_lengths = axis_usize("atr_length", sweep.atr_length)?;
    let multipliers = axis_float("multiplier", sweep.multiplier)?;
    let alpha_percents = axis_float("alpha_percent", sweep.alpha_percent)?;
    let threshold_atrs = axis_float("threshold_atr", sweep.threshold_atr)?;

    let cap = atr_lengths
        .len()
        .checked_mul(multipliers.len())
        .and_then(|v| v.checked_mul(alpha_percents.len()))
        .and_then(|v| v.checked_mul(threshold_atrs.len()))
        .ok_or(SuperTrendRecoveryError::InvalidRange {
            axis: "grid",
            start: "cap".to_string(),
            end: "overflow".to_string(),
            step: "mul".to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &atr_length in &atr_lengths {
        for &multiplier in &multipliers {
            for &alpha_percent in &alpha_percents {
                for &threshold_atr in &threshold_atrs {
                    out.push(SuperTrendRecoveryParams {
                        atr_length: Some(atr_length),
                        multiplier: Some(multiplier),
                        alpha_percent: Some(alpha_percent),
                        threshold_atr: Some(threshold_atr),
                    });
                }
            }
        }
    }
    Ok(out)
}

fn supertrend_recovery_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendRecoveryBatchRange,
    parallel: bool,
    band_out: &mut [f64],
    switch_price_out: &mut [f64],
    trend_out: &mut [f64],
    changed_out: &mut [f64],
) -> Result<Vec<SuperTrendRecoveryParams>, SuperTrendRecoveryError> {
    let (_, max_run) = analyze_valid_segments(high, low, close)?;
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(SuperTrendRecoveryError::OutputLengthMismatch {
            expected: usize::MAX,
            got: band_out.len(),
        })?;
    if band_out.len() != expected
        || switch_price_out.len() != expected
        || trend_out.len() != expected
        || changed_out.len() != expected
    {
        return Err(SuperTrendRecoveryError::OutputLengthMismatch {
            expected,
            got: band_out
                .len()
                .max(switch_price_out.len())
                .max(trend_out.len())
                .max(changed_out.len()),
        });
    }

    for params in &combos {
        let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
        let multiplier = params.multiplier.unwrap_or(DEFAULT_MULTIPLIER);
        let alpha_percent = params.alpha_percent.unwrap_or(DEFAULT_ALPHA_PERCENT);
        let threshold_atr = params.threshold_atr.unwrap_or(DEFAULT_THRESHOLD_ATR);
        validate_params(atr_length, multiplier, alpha_percent, threshold_atr, cols)?;
        if max_run < atr_length {
            return Err(SuperTrendRecoveryError::NotEnoughValidData {
                needed: atr_length,
                valid: max_run,
            });
        }
    }

    let do_row = |row: usize,
                  band_row: &mut [f64],
                  switch_row: &mut [f64],
                  trend_row: &mut [f64],
                  changed_row: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            high,
            low,
            close,
            params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH),
            params.multiplier.unwrap_or(DEFAULT_MULTIPLIER),
            params.alpha_percent.unwrap_or(DEFAULT_ALPHA_PERCENT),
            params.threshold_atr.unwrap_or(DEFAULT_THRESHOLD_ATR),
            band_row,
            switch_row,
            trend_row,
            changed_row,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            band_out
                .par_chunks_mut(cols)
                .zip(switch_price_out.par_chunks_mut(cols))
                .zip(trend_out.par_chunks_mut(cols))
                .zip(changed_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(
                    |(row, (((band_row, switch_row), trend_row), changed_row))| {
                        do_row(row, band_row, switch_row, trend_row, changed_row)
                    },
                )?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((band_row, switch_row), trend_row), changed_row)) in band_out
                .chunks_mut(cols)
                .zip(switch_price_out.chunks_mut(cols))
                .zip(trend_out.chunks_mut(cols))
                .zip(changed_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, band_row, switch_row, trend_row, changed_row)?;
            }
        }
    } else {
        for (row, (((band_row, switch_row), trend_row), changed_row)) in band_out
            .chunks_mut(cols)
            .zip(switch_price_out.chunks_mut(cols))
            .zip(trend_out.chunks_mut(cols))
            .zip(changed_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, band_row, switch_row, trend_row, changed_row)?;
        }
    }

    Ok(combos)
}

pub fn supertrend_recovery_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendRecoveryBatchRange,
    kernel: Kernel,
) -> Result<SuperTrendRecoveryBatchOutput, SuperTrendRecoveryError> {
    match kernel {
        Kernel::Auto => {
            let _ = detect_best_batch_kernel();
        }
        k if !k.is_batch() => return Err(SuperTrendRecoveryError::InvalidKernelForBatch(k)),
        _ => {}
    }
    supertrend_recovery_batch_par_slice(high, low, close, sweep, Kernel::ScalarBatch)
}

pub fn supertrend_recovery_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendRecoveryBatchRange,
    _kernel: Kernel,
) -> Result<SuperTrendRecoveryBatchOutput, SuperTrendRecoveryError> {
    supertrend_recovery_batch_impl(high, low, close, sweep, false)
}

pub fn supertrend_recovery_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendRecoveryBatchRange,
    _kernel: Kernel,
) -> Result<SuperTrendRecoveryBatchOutput, SuperTrendRecoveryError> {
    supertrend_recovery_batch_impl(high, low, close, sweep, true)
}

fn supertrend_recovery_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendRecoveryBatchRange,
    parallel: bool,
) -> Result<SuperTrendRecoveryBatchOutput, SuperTrendRecoveryError> {
    let rows = expand_grid(sweep)?.len();
    let cols = close.len();

    let band_mu = make_uninit_matrix(rows, cols);
    let switch_mu = make_uninit_matrix(rows, cols);
    let trend_mu = make_uninit_matrix(rows, cols);
    let changed_mu = make_uninit_matrix(rows, cols);

    let mut band_guard = ManuallyDrop::new(band_mu);
    let mut switch_guard = ManuallyDrop::new(switch_mu);
    let mut trend_guard = ManuallyDrop::new(trend_mu);
    let mut changed_guard = ManuallyDrop::new(changed_mu);

    let band_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(band_guard.as_mut_ptr() as *mut f64, band_guard.len())
    };
    let switch_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(switch_guard.as_mut_ptr() as *mut f64, switch_guard.len())
    };
    let trend_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(trend_guard.as_mut_ptr() as *mut f64, trend_guard.len())
    };
    let changed_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(changed_guard.as_mut_ptr() as *mut f64, changed_guard.len())
    };

    let combos = supertrend_recovery_batch_inner_into(
        high,
        low,
        close,
        sweep,
        parallel,
        band_out,
        switch_out,
        trend_out,
        changed_out,
    )?;

    let band = unsafe {
        Vec::from_raw_parts(
            band_guard.as_mut_ptr() as *mut f64,
            band_guard.len(),
            band_guard.capacity(),
        )
    };
    let switch_price = unsafe {
        Vec::from_raw_parts(
            switch_guard.as_mut_ptr() as *mut f64,
            switch_guard.len(),
            switch_guard.capacity(),
        )
    };
    let trend = unsafe {
        Vec::from_raw_parts(
            trend_guard.as_mut_ptr() as *mut f64,
            trend_guard.len(),
            trend_guard.capacity(),
        )
    };
    let changed = unsafe {
        Vec::from_raw_parts(
            changed_guard.as_mut_ptr() as *mut f64,
            changed_guard.len(),
            changed_guard.capacity(),
        )
    };

    Ok(SuperTrendRecoveryBatchOutput {
        band,
        switch_price,
        trend,
        changed,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "supertrend_recovery")]
#[pyo3(signature = (high, low, close, atr_length=DEFAULT_ATR_LENGTH, multiplier=DEFAULT_MULTIPLIER, alpha_percent=DEFAULT_ALPHA_PERCENT, threshold_atr=DEFAULT_THRESHOLD_ATR, kernel=None))]
pub fn supertrend_recovery_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    atr_length: usize,
    multiplier: f64,
    alpha_percent: f64,
    threshold_atr: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = SuperTrendRecoveryInput::from_slices(
        high_slice,
        low_slice,
        close_slice,
        SuperTrendRecoveryParams {
            atr_length: Some(atr_length),
            multiplier: Some(multiplier),
            alpha_percent: Some(alpha_percent),
            threshold_atr: Some(threshold_atr),
        },
    );
    let output = py
        .allow_threads(|| supertrend_recovery_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.band.into_pyarray(py),
        output.switch_price.into_pyarray(py),
        output.trend.into_pyarray(py),
        output.changed.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "supertrend_recovery_batch")]
#[pyo3(signature = (high, low, close, atr_length_range=(DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0), multiplier_range=(DEFAULT_MULTIPLIER, DEFAULT_MULTIPLIER, 0.0), alpha_percent_range=(DEFAULT_ALPHA_PERCENT, DEFAULT_ALPHA_PERCENT, 0.0), threshold_atr_range=(DEFAULT_THRESHOLD_ATR, DEFAULT_THRESHOLD_ATR, 0.0), kernel=None))]
pub fn supertrend_recovery_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    atr_length_range: (usize, usize, usize),
    multiplier_range: (f64, f64, f64),
    alpha_percent_range: (f64, f64, f64),
    threshold_atr_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = SuperTrendRecoveryBatchRange {
        atr_length: atr_length_range,
        multiplier: multiplier_range,
        alpha_percent: alpha_percent_range,
        threshold_atr: threshold_atr_range,
    };

    let rows = expand_grid(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .len();
    let cols = close_slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow in supertrend_recovery_batch"))?;

    let band_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let switch_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let trend_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let changed_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let band_out = unsafe { band_arr.as_slice_mut()? };
    let switch_out = unsafe { switch_arr.as_slice_mut()? };
    let trend_out = unsafe { trend_arr.as_slice_mut()? };
    let changed_out = unsafe { changed_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            supertrend_recovery_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                !matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch),
                band_out,
                switch_out,
                trend_out,
                changed_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("band", band_arr.reshape((rows, cols))?)?;
    dict.set_item("switch_price", switch_arr.reshape((rows, cols))?)?;
    dict.set_item("trend", trend_arr.reshape((rows, cols))?)?;
    dict.set_item("changed", changed_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "atr_lengths",
        combos
            .iter()
            .map(|c| c.atr_length.unwrap_or(DEFAULT_ATR_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "multipliers",
        combos
            .iter()
            .map(|c| c.multiplier.unwrap_or(DEFAULT_MULTIPLIER))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "alpha_percents",
        combos
            .iter()
            .map(|c| c.alpha_percent.unwrap_or(DEFAULT_ALPHA_PERCENT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "threshold_atrs",
        combos
            .iter()
            .map(|c| c.threshold_atr.unwrap_or(DEFAULT_THRESHOLD_ATR))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "SuperTrendRecoveryStream")]
pub struct SuperTrendRecoveryStreamPy {
    stream: SuperTrendRecoveryStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SuperTrendRecoveryStreamPy {
    #[new]
    #[pyo3(signature = (atr_length=DEFAULT_ATR_LENGTH, multiplier=DEFAULT_MULTIPLIER, alpha_percent=DEFAULT_ALPHA_PERCENT, threshold_atr=DEFAULT_THRESHOLD_ATR))]
    fn new(
        atr_length: usize,
        multiplier: f64,
        alpha_percent: f64,
        threshold_atr: f64,
    ) -> PyResult<Self> {
        let stream = SuperTrendRecoveryStream::try_new(SuperTrendRecoveryParams {
            atr_length: Some(atr_length),
            multiplier: Some(multiplier),
            alpha_percent: Some(alpha_percent),
            threshold_atr: Some(threshold_atr),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SuperTrendRecoveryBatchConfig {
    pub atr_length_range: (usize, usize, usize),
    pub multiplier_range: (f64, f64, f64),
    pub alpha_percent_range: (f64, f64, f64),
    pub threshold_atr_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SuperTrendRecoveryBatchJsOutput {
    pub band: Vec<f64>,
    pub switch_price: Vec<f64>,
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
    pub combos: Vec<SuperTrendRecoveryParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_recovery_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr_length: usize,
    multiplier: f64,
    alpha_percent: f64,
    threshold_atr: f64,
) -> Result<JsValue, JsValue> {
    let input = SuperTrendRecoveryInput::from_slices(
        high,
        low,
        close,
        SuperTrendRecoveryParams {
            atr_length: Some(atr_length),
            multiplier: Some(multiplier),
            alpha_percent: Some(alpha_percent),
            threshold_atr: Some(threshold_atr),
        },
    );
    let output = supertrend_recovery_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_recovery_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_recovery_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_recovery_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    band_ptr: *mut f64,
    switch_price_ptr: *mut f64,
    trend_ptr: *mut f64,
    changed_ptr: *mut f64,
    len: usize,
    atr_length: usize,
    multiplier: f64,
    alpha_percent: f64,
    threshold_atr: f64,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || band_ptr.is_null()
        || switch_price_ptr.is_null()
        || trend_ptr.is_null()
        || changed_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = SuperTrendRecoveryInput::from_slices(
            high,
            low,
            close,
            SuperTrendRecoveryParams {
                atr_length: Some(atr_length),
                multiplier: Some(multiplier),
                alpha_percent: Some(alpha_percent),
                threshold_atr: Some(threshold_atr),
            },
        );

        let aliased = [
            high_ptr as *const u8,
            low_ptr as *const u8,
            close_ptr as *const u8,
        ]
        .iter()
        .any(|&inp| {
            [
                band_ptr as *const u8,
                switch_price_ptr as *const u8,
                trend_ptr as *const u8,
                changed_ptr as *const u8,
            ]
            .iter()
            .any(|&out| inp == out)
        }) || band_ptr == switch_price_ptr
            || band_ptr == trend_ptr
            || band_ptr == changed_ptr
            || switch_price_ptr == trend_ptr
            || switch_price_ptr == changed_ptr
            || trend_ptr == changed_ptr;

        if aliased {
            let output = supertrend_recovery_with_kernel(&input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(band_ptr, len).copy_from_slice(&output.band);
            std::slice::from_raw_parts_mut(switch_price_ptr, len)
                .copy_from_slice(&output.switch_price);
            std::slice::from_raw_parts_mut(trend_ptr, len).copy_from_slice(&output.trend);
            std::slice::from_raw_parts_mut(changed_ptr, len).copy_from_slice(&output.changed);
        } else {
            let band_out = std::slice::from_raw_parts_mut(band_ptr, len);
            let switch_out = std::slice::from_raw_parts_mut(switch_price_ptr, len);
            let trend_out = std::slice::from_raw_parts_mut(trend_ptr, len);
            let changed_out = std::slice::from_raw_parts_mut(changed_ptr, len);
            supertrend_recovery_into_slice(
                band_out,
                switch_out,
                trend_out,
                changed_out,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = supertrend_recovery_batch)]
pub fn supertrend_recovery_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: SuperTrendRecoveryBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = SuperTrendRecoveryBatchRange {
        atr_length: config.atr_length_range,
        multiplier: config.multiplier_range,
        alpha_percent: config.alpha_percent_range,
        threshold_atr: config.threshold_atr_range,
    };
    let output = supertrend_recovery_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_output = SuperTrendRecoveryBatchJsOutput {
        band: output.band,
        switch_price: output.switch_price,
        trend: output.trend,
        changed: output.changed,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_recovery_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    band_ptr: *mut f64,
    switch_price_ptr: *mut f64,
    trend_ptr: *mut f64,
    changed_ptr: *mut f64,
    len: usize,
    atr_length_start: usize,
    atr_length_end: usize,
    atr_length_step: usize,
    multiplier_start: f64,
    multiplier_end: f64,
    multiplier_step: f64,
    alpha_percent_start: f64,
    alpha_percent_end: f64,
    alpha_percent_step: f64,
    threshold_atr_start: f64,
    threshold_atr_end: f64,
    threshold_atr_step: f64,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || band_ptr.is_null()
        || switch_price_ptr.is_null()
        || trend_ptr.is_null()
        || changed_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = SuperTrendRecoveryBatchRange {
        atr_length: (atr_length_start, atr_length_end, atr_length_step),
        multiplier: (multiplier_start, multiplier_end, multiplier_step),
        alpha_percent: (alpha_percent_start, alpha_percent_end, alpha_percent_step),
        threshold_atr: (threshold_atr_start, threshold_atr_end, threshold_atr_step),
    };
    let rows = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let band_out = std::slice::from_raw_parts_mut(band_ptr, total);
        let switch_out = std::slice::from_raw_parts_mut(switch_price_ptr, total);
        let trend_out = std::slice::from_raw_parts_mut(trend_ptr, total);
        let changed_out = std::slice::from_raw_parts_mut(changed_ptr, total);
        supertrend_recovery_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            false,
            band_out,
            switch_out,
            trend_out,
            changed_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn trend_data(size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(size);
        let mut low = Vec::with_capacity(size);
        let mut close = Vec::with_capacity(size);
        for i in 0..size {
            let base = 100.0 + i as f64 * 0.8;
            high.push(base + 1.2 + (i % 3) as f64 * 0.1);
            low.push(base - 1.0 - (i % 2) as f64 * 0.1);
            close.push(base + ((i % 5) as f64 - 2.0) * 0.05);
        }
        (high, low, close)
    }

    fn reversal_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 102.0, 99.0, 96.0, 93.0, 91.0, 90.0,
            91.0, 93.0, 96.0, 100.0, 105.0, 109.0, 112.0, 111.0, 109.0, 107.0, 104.0, 101.0, 98.0,
            96.0, 95.0, 96.0, 98.0,
        ];
        let high = close.iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let low = close.iter().map(|v| v - 1.0).collect::<Vec<_>>();
        (high, low, close)
    }

    fn recovery_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 102.0, 97.0, 92.0, 88.0, 90.0, 93.0, 96.0,
            99.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0,
            90.0, 89.0, 88.0,
        ];
        let high = close.iter().map(|v| v + 0.9).collect::<Vec<_>>();
        let low = close.iter().map(|v| v - 0.9).collect::<Vec<_>>();
        (high, low, close)
    }

    fn arrays_eq_nan(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(x, y)| {
                (x.is_nan() && y.is_nan())
                    || (!x.is_nan() && !y.is_nan() && (*x - *y).abs() <= 1e-12)
            })
    }

    #[test]
    fn supertrend_recovery_into_matches_single() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = trend_data(160);
        let input = SuperTrendRecoveryInput::from_slices(
            &high,
            &low,
            &close,
            SuperTrendRecoveryParams::default(),
        );
        let single = supertrend_recovery(&input)?;

        let mut band = vec![0.0; close.len()];
        let mut switch_price = vec![0.0; close.len()];
        let mut trend = vec![0.0; close.len()];
        let mut changed = vec![0.0; close.len()];
        supertrend_recovery_into_slice(
            &mut band,
            &mut switch_price,
            &mut trend,
            &mut changed,
            &input,
            Kernel::Auto,
        )?;

        assert!(arrays_eq_nan(&single.band, &band));
        assert!(arrays_eq_nan(&single.switch_price, &switch_price));
        assert!(arrays_eq_nan(&single.trend, &trend));
        assert!(arrays_eq_nan(&single.changed, &changed));
        Ok(())
    }

    #[test]
    fn supertrend_recovery_stream_matches_batch() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = trend_data(170);
        let params = SuperTrendRecoveryParams::default();
        let input = SuperTrendRecoveryInput::from_slices(&high, &low, &close, params.clone());
        let batch = supertrend_recovery(&input)?;

        let mut stream = SuperTrendRecoveryStream::try_new(params)?;
        let mut band = Vec::with_capacity(close.len());
        let mut switch_price = Vec::with_capacity(close.len());
        let mut trend = Vec::with_capacity(close.len());
        let mut changed = Vec::with_capacity(close.len());

        for i in 0..close.len() {
            if let Some((b, s, t, c)) = stream.update(high[i], low[i], close[i]) {
                band.push(b);
                switch_price.push(s);
                trend.push(t);
                changed.push(c);
            } else {
                band.push(f64::NAN);
                switch_price.push(f64::NAN);
                trend.push(f64::NAN);
                changed.push(f64::NAN);
            }
        }

        assert!(arrays_eq_nan(&batch.band, &band));
        assert!(arrays_eq_nan(&batch.switch_price, &switch_price));
        assert!(arrays_eq_nan(&batch.trend, &trend));
        assert!(arrays_eq_nan(&batch.changed, &changed));
        Ok(())
    }

    #[test]
    fn supertrend_recovery_reversal_behavior() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = reversal_data();
        let output = supertrend_recovery(&SuperTrendRecoveryInput::from_slices(
            &high,
            &low,
            &close,
            SuperTrendRecoveryParams {
                atr_length: Some(4),
                multiplier: Some(1.5),
                alpha_percent: Some(5.0),
                threshold_atr: Some(1.0),
            },
        ))?;

        let changes = output
            .changed
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v == 1.0 { Some(i) } else { None })
            .collect::<Vec<_>>();
        assert!(!changes.is_empty());
        let first = changes[0];
        assert!(output.band[first].is_finite());
        assert!(output.switch_price[first].is_finite());
        assert!(output.trend[first] == 1.0 || output.trend[first] == -1.0);
        Ok(())
    }

    #[test]
    fn supertrend_recovery_recovery_behavior() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = recovery_data();
        let recovered = supertrend_recovery(&SuperTrendRecoveryInput::from_slices(
            &high,
            &low,
            &close,
            SuperTrendRecoveryParams {
                atr_length: Some(4),
                multiplier: Some(3.0),
                alpha_percent: Some(100.0),
                threshold_atr: Some(0.0),
            },
        ))?;
        let baseline = supertrend_recovery(&SuperTrendRecoveryInput::from_slices(
            &high,
            &low,
            &close,
            SuperTrendRecoveryParams {
                atr_length: Some(4),
                multiplier: Some(3.0),
                alpha_percent: Some(0.1),
                threshold_atr: Some(1000.0),
            },
        ))?;

        let mut found = false;
        for i in 0..close.len() {
            if recovered.trend[i] == baseline.trend[i]
                && recovered.band[i].is_finite()
                && baseline.band[i].is_finite()
            {
                if recovered.trend[i] == 1.0 && recovered.band[i] > baseline.band[i] {
                    found = true;
                    break;
                }
                if recovered.trend[i] == -1.0 && recovered.band[i] < baseline.band[i] {
                    found = true;
                    break;
                }
            }
        }
        assert!(found);
        Ok(())
    }

    #[test]
    fn supertrend_recovery_nan_gap_restarts() -> Result<(), Box<dyn StdError>> {
        let (mut high, mut low, mut close) = trend_data(170);
        high[120] = f64::NAN;
        low[120] = f64::NAN;
        close[120] = f64::NAN;
        let output = supertrend_recovery(&SuperTrendRecoveryInput::from_slices(
            &high,
            &low,
            &close,
            SuperTrendRecoveryParams::default(),
        ))?;

        let restart_end = (120 + DEFAULT_ATR_LENGTH).min(output.band.len());
        for i in 120..restart_end {
            assert!(output.band[i].is_nan());
            assert!(output.trend[i].is_nan());
            assert!(output.changed[i].is_nan());
        }
        Ok(())
    }

    #[test]
    fn supertrend_recovery_batch_matches_single() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = trend_data(170);
        let sweep = SuperTrendRecoveryBatchRange {
            atr_length: (4, 5, 1),
            multiplier: (1.5, 2.0, 0.5),
            alpha_percent: (5.0, 10.0, 5.0),
            threshold_atr: (0.5, 1.0, 0.5),
        };
        let batch = supertrend_recovery_batch_with_kernel(
            &high,
            &low,
            &close,
            &sweep,
            Kernel::ScalarBatch,
        )?;

        assert_eq!(batch.rows, 16);
        assert_eq!(batch.cols, close.len());
        for row in 0..batch.rows {
            let combo = &batch.combos[row];
            let single = supertrend_recovery(&SuperTrendRecoveryInput::from_slices(
                &high,
                &low,
                &close,
                combo.clone(),
            ))?;
            let start = row * batch.cols;
            let end = start + batch.cols;
            assert!(arrays_eq_nan(
                &batch.band[start..end],
                single.band.as_slice()
            ));
            assert!(arrays_eq_nan(
                &batch.switch_price[start..end],
                single.switch_price.as_slice()
            ));
            assert!(arrays_eq_nan(
                &batch.trend[start..end],
                single.trend.as_slice()
            ));
            assert!(arrays_eq_nan(
                &batch.changed[start..end],
                single.changed.as_slice()
            ));
        }
        Ok(())
    }

    #[test]
    fn supertrend_recovery_invalid_alpha_errors() {
        let (high, low, close) = trend_data(160);
        let input = SuperTrendRecoveryInput::from_slices(
            &high,
            &low,
            &close,
            SuperTrendRecoveryParams {
                atr_length: Some(10),
                multiplier: Some(3.0),
                alpha_percent: Some(0.0),
                threshold_atr: Some(1.0),
            },
        );
        assert!(matches!(
            supertrend_recovery(&input),
            Err(SuperTrendRecoveryError::InvalidAlphaPercent { .. })
        ));
    }

    #[test]
    fn supertrend_recovery_all_nan_errors() {
        let high = vec![f64::NAN; 160];
        let low = vec![f64::NAN; 160];
        let close = vec![f64::NAN; 160];
        let input = SuperTrendRecoveryInput::from_slices(
            &high,
            &low,
            &close,
            SuperTrendRecoveryParams::default(),
        );
        assert!(matches!(
            supertrend_recovery(&input),
            Err(SuperTrendRecoveryError::AllValuesNaN)
        ));
    }

    #[test]
    fn supertrend_recovery_default_candles_smoke() -> Result<(), Box<dyn StdError>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        let output = supertrend_recovery(&SuperTrendRecoveryInput::with_default_candles(&candles))?;
        assert_eq!(output.band.len(), candles.close.len());
        assert_eq!(output.switch_price.len(), candles.close.len());
        assert_eq!(output.trend.len(), candles.close.len());
        assert_eq!(output.changed.len(), candles.close.len());
        Ok(())
    }
}
