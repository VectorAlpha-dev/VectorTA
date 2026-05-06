#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
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
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn exponential_trend_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    exp_rate: f64,
    initial_distance: f64,
    width_multiplier: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = exponential_trend_js(
        high,
        low,
        close,
        exp_rate,
        initial_distance,
        width_multiplier,
    )?;
    crate::write_wasm_object_f64_outputs("exponential_trend_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn exponential_trend_batch_unified_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = exponential_trend_batch_unified_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "exponential_trend_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_EXP_RATE: f64 = 0.00003;
const DEFAULT_INITIAL_DISTANCE: f64 = 4.0;
const DEFAULT_WIDTH_MULTIPLIER: f64 = 1.0;
const ATR_LENGTH: usize = 14;
const SEED_BAR_INDEX: usize = 100;
const MAX_EXP_RATE: f64 = 0.5;
const MIN_DISTANCE: f64 = 0.1;
const MIN_WIDTH_MULTIPLIER: f64 = 0.1;

#[derive(Debug, Clone)]
pub enum ExponentialTrendData<'a> {
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
pub struct ExponentialTrendOutput {
    pub uptrend_base: Vec<f64>,
    pub downtrend_base: Vec<f64>,
    pub uptrend_extension: Vec<f64>,
    pub downtrend_extension: Vec<f64>,
    pub bullish_change: Vec<f64>,
    pub bearish_change: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct ExponentialTrendParams {
    pub exp_rate: Option<f64>,
    pub initial_distance: Option<f64>,
    pub width_multiplier: Option<f64>,
}

impl Default for ExponentialTrendParams {
    fn default() -> Self {
        Self {
            exp_rate: Some(DEFAULT_EXP_RATE),
            initial_distance: Some(DEFAULT_INITIAL_DISTANCE),
            width_multiplier: Some(DEFAULT_WIDTH_MULTIPLIER),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExponentialTrendInput<'a> {
    pub data: ExponentialTrendData<'a>,
    pub params: ExponentialTrendParams,
}

impl<'a> ExponentialTrendInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles, params: ExponentialTrendParams) -> Self {
        Self {
            data: ExponentialTrendData::Candles { candles },
            params,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: ExponentialTrendParams,
    ) -> Self {
        Self {
            data: ExponentialTrendData::Slices { high, low, close },
            params,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, ExponentialTrendParams::default())
    }

    #[inline(always)]
    pub fn get_exp_rate(&self) -> f64 {
        self.params.exp_rate.unwrap_or(DEFAULT_EXP_RATE)
    }

    #[inline(always)]
    pub fn get_initial_distance(&self) -> f64 {
        self.params
            .initial_distance
            .unwrap_or(DEFAULT_INITIAL_DISTANCE)
    }

    #[inline(always)]
    pub fn get_width_multiplier(&self) -> f64 {
        self.params
            .width_multiplier
            .unwrap_or(DEFAULT_WIDTH_MULTIPLIER)
    }

    #[inline(always)]
    fn as_hlc(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            ExponentialTrendData::Candles { candles } => {
                (&candles.high, &candles.low, &candles.close)
            }
            ExponentialTrendData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

impl<'a> AsRef<[f64]> for ExponentialTrendInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        self.as_hlc().2
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExponentialTrendBuilder {
    exp_rate: Option<f64>,
    initial_distance: Option<f64>,
    width_multiplier: Option<f64>,
    kernel: Kernel,
}

impl Default for ExponentialTrendBuilder {
    fn default() -> Self {
        Self {
            exp_rate: None,
            initial_distance: None,
            width_multiplier: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ExponentialTrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn exp_rate(mut self, value: f64) -> Self {
        self.exp_rate = Some(value);
        self
    }

    #[inline(always)]
    pub fn initial_distance(mut self, value: f64) -> Self {
        self.initial_distance = Some(value);
        self
    }

    #[inline(always)]
    pub fn width_multiplier(mut self, value: f64) -> Self {
        self.width_multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> ExponentialTrendParams {
        ExponentialTrendParams {
            exp_rate: self.exp_rate,
            initial_distance: self.initial_distance,
            width_multiplier: self.width_multiplier,
        }
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<ExponentialTrendOutput, ExponentialTrendError> {
        let kernel = self.kernel;
        let input = ExponentialTrendInput::from_candles(candles, self.params());
        exponential_trend_with_kernel(&input, kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ExponentialTrendOutput, ExponentialTrendError> {
        let kernel = self.kernel;
        let input = ExponentialTrendInput::from_slices(high, low, close, self.params());
        exponential_trend_with_kernel(&input, kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<ExponentialTrendStream, ExponentialTrendError> {
        ExponentialTrendStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum ExponentialTrendError {
    #[error("exponential_trend: input data slice is empty.")]
    EmptyInputData,
    #[error("exponential_trend: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "exponential_trend: inconsistent slice lengths: high={high_len}, low={low_len}, close={close_len}"
    )]
    InconsistentSliceLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("exponential_trend: invalid exp_rate: {exp_rate}")]
    InvalidExpRate { exp_rate: f64 },
    #[error("exponential_trend: invalid initial_distance: {initial_distance}")]
    InvalidInitialDistance { initial_distance: f64 },
    #[error("exponential_trend: invalid width_multiplier: {width_multiplier}")]
    InvalidWidthMultiplier { width_multiplier: f64 },
    #[error("exponential_trend: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("exponential_trend: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "exponential_trend: invalid range for {axis}: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        axis: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("exponential_trend: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct PreparedExponentialTrend<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    exp_rate: f64,
    initial_distance: f64,
    width_multiplier: f64,
    all_valid: bool,
}

#[derive(Clone, Debug)]
struct AtrState {
    len: usize,
    count: usize,
    sum: f64,
    value: f64,
    prev_close: f64,
    have_prev_close: bool,
}

impl AtrState {
    #[inline(always)]
    fn new(len: usize) -> Self {
        Self {
            len,
            count: 0,
            sum: 0.0,
            value: f64::NAN,
            prev_close: f64::NAN,
            have_prev_close: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = f64::NAN;
        self.prev_close = f64::NAN;
        self.have_prev_close = false;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let prev_close = if self.have_prev_close {
            self.prev_close
        } else {
            close
        };
        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        self.prev_close = close;
        self.have_prev_close = true;

        if self.count < self.len {
            self.count += 1;
            self.sum += tr;
            if self.count < self.len {
                return None;
            }
            self.value = self.sum / self.len as f64;
            return Some(self.value);
        }

        self.value = ((self.value * (self.len - 1) as f64) + tr) / self.len as f64;
        Some(self.value)
    }
}

#[derive(Clone, Debug)]
struct ExponentialTrendState {
    atr_state: AtrState,
    prev_upper_band: f64,
    prev_lower_band: f64,
    prev_close: f64,
    prev_atr_ready: bool,
    initial_line: f64,
    prev_initial_line: f64,
    trend: i32,
    bars_since_change: usize,
    segment_index: usize,
}

impl ExponentialTrendState {
    #[inline(always)]
    fn new() -> Self {
        Self {
            atr_state: AtrState::new(ATR_LENGTH),
            prev_upper_band: f64::NAN,
            prev_lower_band: f64::NAN,
            prev_close: f64::NAN,
            prev_atr_ready: false,
            initial_line: 0.0,
            prev_initial_line: 0.0,
            trend: 0,
            bars_since_change: 0,
            segment_index: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.atr_state.reset();
        self.prev_upper_band = f64::NAN;
        self.prev_lower_band = f64::NAN;
        self.prev_close = f64::NAN;
        self.prev_atr_ready = false;
        self.initial_line = 0.0;
        self.prev_initial_line = 0.0;
        self.trend = 0;
        self.bars_since_change = 0;
        self.segment_index = 0;
    }

    #[inline(always)]
    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        exp_rate: f64,
        initial_distance: f64,
        width_multiplier: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.reset();
            return None;
        }

        Some(self.update_valid(
            high,
            low,
            close,
            exp_rate,
            initial_distance,
            width_multiplier,
        ))
    }

    #[inline(always)]
    fn update_valid(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        exp_rate: f64,
        initial_distance: f64,
        width_multiplier: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let atr = self.atr_state.update(high, low, close);
        let atr_ready = atr.is_some();
        let mut upper = f64::NAN;
        let mut lower = f64::NAN;
        let mut direction = 1i32;

        if let Some(atr_value) = atr {
            let src = (high + low) * 0.5;
            let raw_upper = src + initial_distance * atr_value;
            let raw_lower = src - initial_distance * atr_value;
            let prev_lower = if self.prev_lower_band.is_finite() {
                self.prev_lower_band
            } else {
                0.0
            };
            let prev_upper = if self.prev_upper_band.is_finite() {
                self.prev_upper_band
            } else {
                0.0
            };
            let prev_close = if self.prev_close.is_finite() {
                self.prev_close
            } else {
                close
            };

            lower = if raw_lower > prev_lower || prev_close < prev_lower {
                raw_lower
            } else {
                prev_lower
            };
            upper = if raw_upper < prev_upper || prev_close > prev_upper {
                raw_upper
            } else {
                prev_upper
            };

            direction = if !self.prev_atr_ready {
                1
            } else if close < lower {
                1
            } else {
                -1
            };
        }

        let prev_trend = self.trend;
        let prev_initial = self.prev_initial_line;
        let prev_close = self.prev_close;

        if self.segment_index == SEED_BAR_INDEX && upper.is_finite() && lower.is_finite() {
            if direction < 0 {
                self.initial_line = lower;
                self.trend = 1;
            } else {
                self.initial_line = upper;
                self.trend = -1;
            }
        }

        let crossover = self.initial_line.is_finite()
            && prev_close.is_finite()
            && prev_initial.is_finite()
            && close > self.initial_line
            && prev_close <= prev_initial;
        let crossunder = self.initial_line.is_finite()
            && prev_close.is_finite()
            && prev_initial.is_finite()
            && close < self.initial_line
            && prev_close >= prev_initial;

        if crossover && lower.is_finite() {
            self.initial_line = lower;
            self.trend = 1;
        } else if crossunder && upper.is_finite() {
            self.initial_line = upper;
            self.trend = -1;
        }

        if self.trend != prev_trend {
            self.bars_since_change = 0;
        } else if self.trend != 0 {
            self.bars_since_change = self.bars_since_change.saturating_add(1);
        }

        if self.trend != 0 {
            let exp_multiplier = 1.0
                + (self.trend as f64) * (1.0 - (-exp_rate * self.bars_since_change as f64).exp());
            self.initial_line *= exp_multiplier;
        }

        let mut uptrend_base = f64::NAN;
        let mut downtrend_base = f64::NAN;
        let mut uptrend_extension = f64::NAN;
        let mut downtrend_extension = f64::NAN;
        let mut bullish_change = f64::NAN;
        let mut bearish_change = f64::NAN;

        if let Some(atr_value) = atr {
            let extension = self.initial_line
                + if self.trend > 0 {
                    atr_value * width_multiplier
                } else {
                    -atr_value * width_multiplier
                };

            if self.trend == 1 {
                uptrend_base = self.initial_line;
                uptrend_extension = extension;
            } else if self.trend == -1 {
                downtrend_base = self.initial_line;
                downtrend_extension = extension;
            }

            if crossover {
                bullish_change = self.initial_line - atr_value;
            }
            if crossunder {
                bearish_change = self.initial_line + atr_value;
            }
        }

        self.prev_upper_band = upper;
        self.prev_lower_band = lower;
        self.prev_close = close;
        self.prev_initial_line = self.initial_line;
        self.prev_atr_ready = atr_ready;
        self.segment_index = self.segment_index.saturating_add(1);

        (
            uptrend_base,
            downtrend_base,
            uptrend_extension,
            downtrend_extension,
            bullish_change,
            bearish_change,
        )
    }
}

#[derive(Clone, Debug)]
pub struct ExponentialTrendStream {
    params: ExponentialTrendParams,
    state: ExponentialTrendState,
}

impl ExponentialTrendStream {
    #[inline(always)]
    pub fn try_new(params: ExponentialTrendParams) -> Result<Self, ExponentialTrendError> {
        validate_params(
            params.exp_rate.unwrap_or(DEFAULT_EXP_RATE),
            params.initial_distance.unwrap_or(DEFAULT_INITIAL_DISTANCE),
            params.width_multiplier.unwrap_or(DEFAULT_WIDTH_MULTIPLIER),
            usize::MAX,
        )?;
        Ok(Self {
            params,
            state: ExponentialTrendState::new(),
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.state.update(
            high,
            low,
            close,
            self.params.exp_rate.unwrap_or(DEFAULT_EXP_RATE),
            self.params
                .initial_distance
                .unwrap_or(DEFAULT_INITIAL_DISTANCE),
            self.params
                .width_multiplier
                .unwrap_or(DEFAULT_WIDTH_MULTIPLIER),
        )
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.state.reset();
    }
}

#[inline(always)]
fn required_valid_bars() -> usize {
    SEED_BAR_INDEX + 1
}

#[inline(always)]
fn validate_params(
    exp_rate: f64,
    initial_distance: f64,
    width_multiplier: f64,
    data_len: usize,
) -> Result<(), ExponentialTrendError> {
    if !exp_rate.is_finite() || !(0.0..=MAX_EXP_RATE).contains(&exp_rate) {
        return Err(ExponentialTrendError::InvalidExpRate { exp_rate });
    }
    if !initial_distance.is_finite() || initial_distance < MIN_DISTANCE {
        return Err(ExponentialTrendError::InvalidInitialDistance { initial_distance });
    }
    if !width_multiplier.is_finite() || width_multiplier < MIN_WIDTH_MULTIPLIER {
        return Err(ExponentialTrendError::InvalidWidthMultiplier { width_multiplier });
    }
    if data_len != usize::MAX && data_len < required_valid_bars() {
        return Err(ExponentialTrendError::NotEnoughValidData {
            needed: required_valid_bars(),
            valid: data_len,
        });
    }
    Ok(())
}

fn analyze_valid_segments(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(usize, usize), ExponentialTrendError> {
    if high.is_empty() {
        return Err(ExponentialTrendError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(ExponentialTrendError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let mut valid = 0usize;
    let mut run = 0usize;
    let mut max_run = 0usize;

    for i in 0..high.len() {
        if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
            valid += 1;
            run += 1;
            max_run = max_run.max(run);
        } else {
            run = 0;
        }
    }

    if valid == 0 {
        return Err(ExponentialTrendError::AllValuesNaN);
    }

    Ok((valid, max_run))
}

fn prepare_input<'a>(
    input: &'a ExponentialTrendInput<'a>,
    _kernel: Kernel,
) -> Result<PreparedExponentialTrend<'a>, ExponentialTrendError> {
    let (high, low, close) = input.as_hlc();
    let exp_rate = input.get_exp_rate();
    let initial_distance = input.get_initial_distance();
    let width_multiplier = input.get_width_multiplier();
    validate_params(exp_rate, initial_distance, width_multiplier, usize::MAX)?;

    let (_, max_run) = analyze_valid_segments(high, low, close)?;
    if max_run < required_valid_bars() {
        return Err(ExponentialTrendError::NotEnoughValidData {
            needed: required_valid_bars(),
            valid: max_run,
        });
    }
    let all_valid = max_run == close.len();

    Ok(PreparedExponentialTrend {
        high,
        low,
        close,
        exp_rate,
        initial_distance,
        width_multiplier,
        all_valid: max_run == close.len(),
    })
}

#[inline(always)]
fn compute_row_unchecked(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    exp_rate: f64,
    initial_distance: f64,
    width_multiplier: f64,
    all_valid: bool,
    uptrend_base_out: &mut [f64],
    downtrend_base_out: &mut [f64],
    uptrend_extension_out: &mut [f64],
    downtrend_extension_out: &mut [f64],
    bullish_change_out: &mut [f64],
    bearish_change_out: &mut [f64],
) {
    let expected = close.len();
    let mut state = ExponentialTrendState::new();
    if all_valid {
        for i in 0..expected {
            let (ub, db, ue, de, bc, brc) = state.update_valid(
                high[i],
                low[i],
                close[i],
                exp_rate,
                initial_distance,
                width_multiplier,
            );
            uptrend_base_out[i] = ub;
            downtrend_base_out[i] = db;
            uptrend_extension_out[i] = ue;
            downtrend_extension_out[i] = de;
            bullish_change_out[i] = bc;
            bearish_change_out[i] = brc;
        }
        return;
    }

    for i in 0..expected {
        if let Some((ub, db, ue, de, bc, brc)) = state.update(
            high[i],
            low[i],
            close[i],
            exp_rate,
            initial_distance,
            width_multiplier,
        ) {
            uptrend_base_out[i] = ub;
            downtrend_base_out[i] = db;
            uptrend_extension_out[i] = ue;
            downtrend_extension_out[i] = de;
            bullish_change_out[i] = bc;
            bearish_change_out[i] = brc;
        } else {
            uptrend_base_out[i] = f64::NAN;
            downtrend_base_out[i] = f64::NAN;
            uptrend_extension_out[i] = f64::NAN;
            downtrend_extension_out[i] = f64::NAN;
            bullish_change_out[i] = f64::NAN;
            bearish_change_out[i] = f64::NAN;
        }
    }
}

fn compute_row(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    exp_rate: f64,
    initial_distance: f64,
    width_multiplier: f64,
    all_valid: bool,
    uptrend_base_out: &mut [f64],
    downtrend_base_out: &mut [f64],
    uptrend_extension_out: &mut [f64],
    downtrend_extension_out: &mut [f64],
    bullish_change_out: &mut [f64],
    bearish_change_out: &mut [f64],
) -> Result<(), ExponentialTrendError> {
    let expected = close.len();
    for out in [
        &mut *uptrend_base_out,
        &mut *downtrend_base_out,
        &mut *uptrend_extension_out,
        &mut *downtrend_extension_out,
        &mut *bullish_change_out,
        &mut *bearish_change_out,
    ] {
        if out.len() != expected {
            return Err(ExponentialTrendError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
    }

    compute_row_unchecked(
        high,
        low,
        close,
        exp_rate,
        initial_distance,
        width_multiplier,
        all_valid,
        uptrend_base_out,
        downtrend_base_out,
        uptrend_extension_out,
        downtrend_extension_out,
        bullish_change_out,
        bearish_change_out,
    );

    Ok(())
}

#[inline(always)]
pub fn exponential_trend(
    input: &ExponentialTrendInput,
) -> Result<ExponentialTrendOutput, ExponentialTrendError> {
    exponential_trend_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn exponential_trend_with_kernel(
    input: &ExponentialTrendInput,
    kernel: Kernel,
) -> Result<ExponentialTrendOutput, ExponentialTrendError> {
    let prepared = prepare_input(input, kernel)?;
    let len = prepared.close.len();
    let mut uptrend_base = alloc_uninit_f64(len);
    let mut downtrend_base = alloc_uninit_f64(len);
    let mut uptrend_extension = alloc_uninit_f64(len);
    let mut downtrend_extension = alloc_uninit_f64(len);
    let mut bullish_change = alloc_uninit_f64(len);
    let mut bearish_change = alloc_uninit_f64(len);

    compute_row_unchecked(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.exp_rate,
        prepared.initial_distance,
        prepared.width_multiplier,
        prepared.all_valid,
        &mut uptrend_base,
        &mut downtrend_base,
        &mut uptrend_extension,
        &mut downtrend_extension,
        &mut bullish_change,
        &mut bearish_change,
    );

    Ok(ExponentialTrendOutput {
        uptrend_base,
        downtrend_base,
        uptrend_extension,
        downtrend_extension,
        bullish_change,
        bearish_change,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn exponential_trend_into(
    uptrend_base_out: &mut [f64],
    downtrend_base_out: &mut [f64],
    uptrend_extension_out: &mut [f64],
    downtrend_extension_out: &mut [f64],
    bullish_change_out: &mut [f64],
    bearish_change_out: &mut [f64],
    input: &ExponentialTrendInput,
) -> Result<(), ExponentialTrendError> {
    exponential_trend_into_slice(
        uptrend_base_out,
        downtrend_base_out,
        uptrend_extension_out,
        downtrend_extension_out,
        bullish_change_out,
        bearish_change_out,
        input,
        Kernel::Auto,
    )
}

pub fn exponential_trend_into_slice(
    uptrend_base_out: &mut [f64],
    downtrend_base_out: &mut [f64],
    uptrend_extension_out: &mut [f64],
    downtrend_extension_out: &mut [f64],
    bullish_change_out: &mut [f64],
    bearish_change_out: &mut [f64],
    input: &ExponentialTrendInput,
    kernel: Kernel,
) -> Result<(), ExponentialTrendError> {
    let prepared = prepare_input(input, kernel)?;
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.exp_rate,
        prepared.initial_distance,
        prepared.width_multiplier,
        prepared.all_valid,
        uptrend_base_out,
        downtrend_base_out,
        uptrend_extension_out,
        downtrend_extension_out,
        bullish_change_out,
        bearish_change_out,
    )
}

#[derive(Clone, Debug)]
pub struct ExponentialTrendBatchRange {
    pub exp_rate: (f64, f64, f64),
    pub initial_distance: (f64, f64, f64),
    pub width_multiplier: (f64, f64, f64),
}

impl Default for ExponentialTrendBatchRange {
    fn default() -> Self {
        Self {
            exp_rate: (DEFAULT_EXP_RATE, DEFAULT_EXP_RATE, 0.0),
            initial_distance: (DEFAULT_INITIAL_DISTANCE, DEFAULT_INITIAL_DISTANCE, 0.0),
            width_multiplier: (DEFAULT_WIDTH_MULTIPLIER, DEFAULT_WIDTH_MULTIPLIER, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExponentialTrendBatchBuilder {
    range: ExponentialTrendBatchRange,
    kernel: Kernel,
}

impl ExponentialTrendBatchBuilder {
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
    pub fn exp_rate_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.exp_rate = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn initial_distance_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.initial_distance = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn width_multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.width_multiplier = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<ExponentialTrendBatchOutput, ExponentialTrendError> {
        exponential_trend_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ExponentialTrendBatchOutput, ExponentialTrendError> {
        exponential_trend_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct ExponentialTrendBatchOutput {
    pub uptrend_base: Vec<f64>,
    pub downtrend_base: Vec<f64>,
    pub uptrend_extension: Vec<f64>,
    pub downtrend_extension: Vec<f64>,
    pub bullish_change: Vec<f64>,
    pub bearish_change: Vec<f64>,
    pub combos: Vec<ExponentialTrendParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn axis_f64(
    axis: &'static str,
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, ExponentialTrendError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(ExponentialTrendError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() <= f64::EPSILON || step == 0.0 {
        return Ok(vec![start]);
    }
    if step < 0.0 {
        return Err(ExponentialTrendError::InvalidRange {
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
            out.push(value.min(end));
            value += step;
        }
    } else {
        let mut value = start;
        while value >= end - eps {
            out.push(value.max(end));
            value -= step;
        }
    }

    if out.is_empty() || (out.last().copied().unwrap_or(start) - end).abs() > eps {
        return Err(ExponentialTrendError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }

    Ok(out)
}

pub fn expand_grid_exponential_trend(
    sweep: &ExponentialTrendBatchRange,
) -> Result<Vec<ExponentialTrendParams>, ExponentialTrendError> {
    let exp_rates = axis_f64("exp_rate", sweep.exp_rate)?;
    let initial_distances = axis_f64("initial_distance", sweep.initial_distance)?;
    let width_multipliers = axis_f64("width_multiplier", sweep.width_multiplier)?;
    let mut out =
        Vec::with_capacity(exp_rates.len() * initial_distances.len() * width_multipliers.len());
    for &exp_rate in &exp_rates {
        for &initial_distance in &initial_distances {
            for &width_multiplier in &width_multipliers {
                out.push(ExponentialTrendParams {
                    exp_rate: Some(exp_rate),
                    initial_distance: Some(initial_distance),
                    width_multiplier: Some(width_multiplier),
                });
            }
        }
    }
    Ok(out)
}

fn batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ExponentialTrendBatchRange,
    parallel: bool,
    uptrend_base_out: &mut [f64],
    downtrend_base_out: &mut [f64],
    uptrend_extension_out: &mut [f64],
    downtrend_extension_out: &mut [f64],
    bullish_change_out: &mut [f64],
    bearish_change_out: &mut [f64],
) -> Result<Vec<ExponentialTrendParams>, ExponentialTrendError> {
    let (_, max_run) = analyze_valid_segments(high, low, close)?;
    if max_run < required_valid_bars() {
        return Err(ExponentialTrendError::NotEnoughValidData {
            needed: required_valid_bars(),
            valid: max_run,
        });
    }
    let all_valid = max_run == close.len();

    let combos = expand_grid_exponential_trend(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let expected = rows * cols;

    for out in [
        &mut *uptrend_base_out,
        &mut *downtrend_base_out,
        &mut *uptrend_extension_out,
        &mut *downtrend_extension_out,
        &mut *bullish_change_out,
        &mut *bearish_change_out,
    ] {
        if out.len() != expected {
            return Err(ExponentialTrendError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
    }

    for params in &combos {
        validate_params(
            params.exp_rate.unwrap_or(DEFAULT_EXP_RATE),
            params.initial_distance.unwrap_or(DEFAULT_INITIAL_DISTANCE),
            params.width_multiplier.unwrap_or(DEFAULT_WIDTH_MULTIPLIER),
            usize::MAX,
        )?;
    }

    let do_row = |row: usize,
                  uptrend_base_row: &mut [f64],
                  downtrend_base_row: &mut [f64],
                  uptrend_extension_row: &mut [f64],
                  downtrend_extension_row: &mut [f64],
                  bullish_change_row: &mut [f64],
                  bearish_change_row: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            high,
            low,
            close,
            params.exp_rate.unwrap_or(DEFAULT_EXP_RATE),
            params.initial_distance.unwrap_or(DEFAULT_INITIAL_DISTANCE),
            params.width_multiplier.unwrap_or(DEFAULT_WIDTH_MULTIPLIER),
            all_valid,
            uptrend_base_row,
            downtrend_base_row,
            uptrend_extension_row,
            downtrend_extension_row,
            bullish_change_row,
            bearish_change_row,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            uptrend_base_out
                .par_chunks_mut(cols)
                .zip(downtrend_base_out.par_chunks_mut(cols))
                .zip(uptrend_extension_out.par_chunks_mut(cols))
                .zip(downtrend_extension_out.par_chunks_mut(cols))
                .zip(bullish_change_out.par_chunks_mut(cols))
                .zip(bearish_change_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(
                    |(
                        row,
                        (
                            (
                                (
                                    ((uptrend_base_row, downtrend_base_row), uptrend_extension_row),
                                    downtrend_extension_row,
                                ),
                                bullish_change_row,
                            ),
                            bearish_change_row,
                        ),
                    )| {
                        do_row(
                            row,
                            uptrend_base_row,
                            downtrend_base_row,
                            uptrend_extension_row,
                            downtrend_extension_row,
                            bullish_change_row,
                            bearish_change_row,
                        )
                    },
                )?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let start = row * cols;
                let end = start + cols;
                do_row(
                    row,
                    &mut uptrend_base_out[start..end],
                    &mut downtrend_base_out[start..end],
                    &mut uptrend_extension_out[start..end],
                    &mut downtrend_extension_out[start..end],
                    &mut bullish_change_out[start..end],
                    &mut bearish_change_out[start..end],
                )?;
            }
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            do_row(
                row,
                &mut uptrend_base_out[start..end],
                &mut downtrend_base_out[start..end],
                &mut uptrend_extension_out[start..end],
                &mut downtrend_extension_out[start..end],
                &mut bullish_change_out[start..end],
                &mut bearish_change_out[start..end],
            )?;
        }
    }

    Ok(combos)
}

pub fn exponential_trend_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ExponentialTrendBatchRange,
    kernel: Kernel,
) -> Result<ExponentialTrendBatchOutput, ExponentialTrendError> {
    match kernel {
        Kernel::Auto => {
            let _ = detect_best_batch_kernel();
        }
        k if !k.is_batch() => return Err(ExponentialTrendError::InvalidKernelForBatch(k)),
        _ => {}
    }

    let rows = expand_grid_exponential_trend(sweep)?.len();
    let cols = close.len();
    let mut uptrend_base_mu = make_uninit_matrix(rows, cols);
    let mut downtrend_base_mu = make_uninit_matrix(rows, cols);
    let mut uptrend_extension_mu = make_uninit_matrix(rows, cols);
    let mut downtrend_extension_mu = make_uninit_matrix(rows, cols);
    let mut bullish_change_mu = make_uninit_matrix(rows, cols);
    let mut bearish_change_mu = make_uninit_matrix(rows, cols);

    let combos = {
        let uptrend_base: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(
                uptrend_base_mu.as_mut_ptr() as *mut f64,
                uptrend_base_mu.len(),
            )
        };
        let downtrend_base: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(
                downtrend_base_mu.as_mut_ptr() as *mut f64,
                downtrend_base_mu.len(),
            )
        };
        let uptrend_extension: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(
                uptrend_extension_mu.as_mut_ptr() as *mut f64,
                uptrend_extension_mu.len(),
            )
        };
        let downtrend_extension: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(
                downtrend_extension_mu.as_mut_ptr() as *mut f64,
                downtrend_extension_mu.len(),
            )
        };
        let bullish_change: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(
                bullish_change_mu.as_mut_ptr() as *mut f64,
                bullish_change_mu.len(),
            )
        };
        let bearish_change: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(
                bearish_change_mu.as_mut_ptr() as *mut f64,
                bearish_change_mu.len(),
            )
        };

        batch_inner_into(
            high,
            low,
            close,
            sweep,
            !cfg!(target_arch = "wasm32"),
            uptrend_base,
            downtrend_base,
            uptrend_extension,
            downtrend_extension,
            bullish_change,
            bearish_change,
        )?
    };

    let mut uptrend_base_guard = ManuallyDrop::new(uptrend_base_mu);
    let mut downtrend_base_guard = ManuallyDrop::new(downtrend_base_mu);
    let mut uptrend_extension_guard = ManuallyDrop::new(uptrend_extension_mu);
    let mut downtrend_extension_guard = ManuallyDrop::new(downtrend_extension_mu);
    let mut bullish_change_guard = ManuallyDrop::new(bullish_change_mu);
    let mut bearish_change_guard = ManuallyDrop::new(bearish_change_mu);

    Ok(ExponentialTrendBatchOutput {
        uptrend_base: unsafe {
            Vec::from_raw_parts(
                uptrend_base_guard.as_mut_ptr() as *mut f64,
                uptrend_base_guard.len(),
                uptrend_base_guard.capacity(),
            )
        },
        downtrend_base: unsafe {
            Vec::from_raw_parts(
                downtrend_base_guard.as_mut_ptr() as *mut f64,
                downtrend_base_guard.len(),
                downtrend_base_guard.capacity(),
            )
        },
        uptrend_extension: unsafe {
            Vec::from_raw_parts(
                uptrend_extension_guard.as_mut_ptr() as *mut f64,
                uptrend_extension_guard.len(),
                uptrend_extension_guard.capacity(),
            )
        },
        downtrend_extension: unsafe {
            Vec::from_raw_parts(
                downtrend_extension_guard.as_mut_ptr() as *mut f64,
                downtrend_extension_guard.len(),
                downtrend_extension_guard.capacity(),
            )
        },
        bullish_change: unsafe {
            Vec::from_raw_parts(
                bullish_change_guard.as_mut_ptr() as *mut f64,
                bullish_change_guard.len(),
                bullish_change_guard.capacity(),
            )
        },
        bearish_change: unsafe {
            Vec::from_raw_parts(
                bearish_change_guard.as_mut_ptr() as *mut f64,
                bearish_change_guard.len(),
                bearish_change_guard.capacity(),
            )
        },
        combos,
        rows,
        cols,
    })
}

pub fn exponential_trend_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ExponentialTrendBatchRange,
    kernel: Kernel,
) -> Result<ExponentialTrendBatchOutput, ExponentialTrendError> {
    exponential_trend_batch_with_kernel(high, low, close, sweep, kernel)
}

pub fn exponential_trend_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ExponentialTrendBatchRange,
    kernel: Kernel,
) -> Result<ExponentialTrendBatchOutput, ExponentialTrendError> {
    exponential_trend_batch_with_kernel(high, low, close, sweep, kernel)
}

#[cfg(feature = "python")]
#[pyfunction(name = "exponential_trend")]
#[pyo3(signature = (high, low, close, exp_rate=DEFAULT_EXP_RATE, initial_distance=DEFAULT_INITIAL_DISTANCE, width_multiplier=DEFAULT_WIDTH_MULTIPLIER, kernel=None))]
pub fn exponential_trend_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    exp_rate: f64,
    initial_distance: f64,
    width_multiplier: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let kernel = validate_kernel(kernel, false)?;
    let input = ExponentialTrendInput::from_slices(
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        ExponentialTrendParams {
            exp_rate: Some(exp_rate),
            initial_distance: Some(initial_distance),
            width_multiplier: Some(width_multiplier),
        },
    );
    let out = py
        .allow_threads(|| exponential_trend_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("uptrend_base", out.uptrend_base.into_pyarray(py))?;
    dict.set_item("downtrend_base", out.downtrend_base.into_pyarray(py))?;
    dict.set_item("uptrend_extension", out.uptrend_extension.into_pyarray(py))?;
    dict.set_item(
        "downtrend_extension",
        out.downtrend_extension.into_pyarray(py),
    )?;
    dict.set_item("bullish_change", out.bullish_change.into_pyarray(py))?;
    dict.set_item("bearish_change", out.bearish_change.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "exponential_trend_batch")]
#[pyo3(signature = (high, low, close, exp_rate_range=(DEFAULT_EXP_RATE, DEFAULT_EXP_RATE, 0.0), initial_distance_range=(DEFAULT_INITIAL_DISTANCE, DEFAULT_INITIAL_DISTANCE, 0.0), width_multiplier_range=(DEFAULT_WIDTH_MULTIPLIER, DEFAULT_WIDTH_MULTIPLIER, 0.0), kernel=None))]
pub fn exponential_trend_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    exp_rate_range: (f64, f64, f64),
    initial_distance_range: (f64, f64, f64),
    width_multiplier_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let kernel = validate_kernel(kernel, true)?;
    let out = exponential_trend_batch_with_kernel(
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        &ExponentialTrendBatchRange {
            exp_rate: exp_rate_range,
            initial_distance: initial_distance_range,
            width_multiplier: width_multiplier_range,
        },
        kernel,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item(
        "uptrend_base",
        out.uptrend_base
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "downtrend_base",
        out.downtrend_base
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "uptrend_extension",
        out.uptrend_extension
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "downtrend_extension",
        out.downtrend_extension
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "bullish_change",
        out.bullish_change
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "bearish_change",
        out.bearish_change
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "exp_rates",
        out.combos
            .iter()
            .map(|combo| combo.exp_rate.unwrap_or(DEFAULT_EXP_RATE))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "initial_distances",
        out.combos
            .iter()
            .map(|combo| combo.initial_distance.unwrap_or(DEFAULT_INITIAL_DISTANCE))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "width_multipliers",
        out.combos
            .iter()
            .map(|combo| combo.width_multiplier.unwrap_or(DEFAULT_WIDTH_MULTIPLIER))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "ExponentialTrendStream")]
pub struct ExponentialTrendStreamPy {
    inner: ExponentialTrendStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ExponentialTrendStreamPy {
    #[new]
    #[pyo3(signature = (exp_rate=None, initial_distance=None, width_multiplier=None))]
    pub fn new(
        exp_rate: Option<f64>,
        initial_distance: Option<f64>,
        width_multiplier: Option<f64>,
    ) -> PyResult<Self> {
        let inner = ExponentialTrendStream::try_new(ExponentialTrendParams {
            exp_rate,
            initial_distance,
            width_multiplier,
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.inner.update(high, low, close)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct ExponentialTrendBatchConfig {
    pub exp_rate_range: (f64, f64, f64),
    pub initial_distance_range: (f64, f64, f64),
    pub width_multiplier_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn exponential_trend_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    exp_rate: f64,
    initial_distance: f64,
    width_multiplier: f64,
) -> Result<JsValue, JsValue> {
    let input = ExponentialTrendInput::from_slices(
        high,
        low,
        close,
        ExponentialTrendParams {
            exp_rate: Some(exp_rate),
            initial_distance: Some(initial_distance),
            width_multiplier: Some(width_multiplier),
        },
    );
    let out = exponential_trend_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn exponential_trend_alloc(len: usize) -> *mut f64 {
    let mut values = Vec::<f64>::with_capacity(len);
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn exponential_trend_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn exponential_trend_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    uptrend_base_ptr: *mut f64,
    downtrend_base_ptr: *mut f64,
    uptrend_extension_ptr: *mut f64,
    downtrend_extension_ptr: *mut f64,
    bullish_change_ptr: *mut f64,
    bearish_change_ptr: *mut f64,
    len: usize,
    exp_rate: f64,
    initial_distance: f64,
    width_multiplier: f64,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || uptrend_base_ptr.is_null()
        || downtrend_base_ptr.is_null()
        || uptrend_extension_ptr.is_null()
        || downtrend_extension_ptr.is_null()
        || bullish_change_ptr.is_null()
        || bearish_change_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let input = ExponentialTrendInput::from_slices(
            std::slice::from_raw_parts(high_ptr, len),
            std::slice::from_raw_parts(low_ptr, len),
            std::slice::from_raw_parts(close_ptr, len),
            ExponentialTrendParams {
                exp_rate: Some(exp_rate),
                initial_distance: Some(initial_distance),
                width_multiplier: Some(width_multiplier),
            },
        );
        let out = exponential_trend_with_kernel(&input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        std::slice::from_raw_parts_mut(uptrend_base_ptr, len).copy_from_slice(&out.uptrend_base);
        std::slice::from_raw_parts_mut(downtrend_base_ptr, len)
            .copy_from_slice(&out.downtrend_base);
        std::slice::from_raw_parts_mut(uptrend_extension_ptr, len)
            .copy_from_slice(&out.uptrend_extension);
        std::slice::from_raw_parts_mut(downtrend_extension_ptr, len)
            .copy_from_slice(&out.downtrend_extension);
        std::slice::from_raw_parts_mut(bullish_change_ptr, len)
            .copy_from_slice(&out.bullish_change);
        std::slice::from_raw_parts_mut(bearish_change_ptr, len)
            .copy_from_slice(&out.bearish_change);
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = exponential_trend_batch)]
pub fn exponential_trend_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: ExponentialTrendBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let out = exponential_trend_batch_with_kernel(
        high,
        low,
        close,
        &ExponentialTrendBatchRange {
            exp_rate: config.exp_rate_range,
            initial_distance: config.initial_distance_range,
            width_multiplier: config.width_multiplier_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = exponential_trend_batch_into)]
pub fn exponential_trend_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    uptrend_base_ptr: *mut f64,
    downtrend_base_ptr: *mut f64,
    uptrend_extension_ptr: *mut f64,
    downtrend_extension_ptr: *mut f64,
    bullish_change_ptr: *mut f64,
    bearish_change_ptr: *mut f64,
    len: usize,
    exp_rate_start: f64,
    exp_rate_end: f64,
    exp_rate_step: f64,
    initial_distance_start: f64,
    initial_distance_end: f64,
    initial_distance_step: f64,
    width_multiplier_start: f64,
    width_multiplier_end: f64,
    width_multiplier_step: f64,
) -> Result<usize, JsValue> {
    let sweep = ExponentialTrendBatchRange {
        exp_rate: (exp_rate_start, exp_rate_end, exp_rate_step),
        initial_distance: (
            initial_distance_start,
            initial_distance_end,
            initial_distance_step,
        ),
        width_multiplier: (
            width_multiplier_start,
            width_multiplier_end,
            width_multiplier_step,
        ),
    };
    let rows = expand_grid_exponential_trend(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows * cols overflow"))?;

    unsafe {
        let out = exponential_trend_batch_with_kernel(
            std::slice::from_raw_parts(high_ptr, len),
            std::slice::from_raw_parts(low_ptr, len),
            std::slice::from_raw_parts(close_ptr, len),
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        std::slice::from_raw_parts_mut(uptrend_base_ptr, total).copy_from_slice(&out.uptrend_base);
        std::slice::from_raw_parts_mut(downtrend_base_ptr, total)
            .copy_from_slice(&out.downtrend_base);
        std::slice::from_raw_parts_mut(uptrend_extension_ptr, total)
            .copy_from_slice(&out.uptrend_extension);
        std::slice::from_raw_parts_mut(downtrend_extension_ptr, total)
            .copy_from_slice(&out.downtrend_extension);
        std::slice::from_raw_parts_mut(bullish_change_ptr, total)
            .copy_from_slice(&out.bullish_change);
        std::slice::from_raw_parts_mut(bearish_change_ptr, total)
            .copy_from_slice(&out.bearish_change);
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct ExponentialTrendStreamWasm {
    inner: ExponentialTrendStream,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl ExponentialTrendStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        exp_rate: Option<f64>,
        initial_distance: Option<f64>,
        width_multiplier: Option<f64>,
    ) -> Result<ExponentialTrendStreamWasm, JsValue> {
        let inner = ExponentialTrendStream::try_new(ExponentialTrendParams {
            exp_rate,
            initial_distance,
            width_multiplier,
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.update(high, low, close))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_series_eq(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (lhs, rhs) in left.iter().zip(right.iter()) {
            assert!(lhs == rhs || (lhs.is_nan() && rhs.is_nan()));
        }
    }

    fn sample_hlc() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(160);
        let mut low = Vec::with_capacity(160);
        let mut close = Vec::with_capacity(160);

        for i in 0..60 {
            let c = 100.0 + (i as f64) * 0.65 + ((i % 5) as f64 - 2.0) * 0.08;
            close.push(c);
        }
        for i in 60..100 {
            let x = (i - 60) as f64;
            let c = 139.0 + x * 0.05 - ((i % 4) as f64) * 0.12;
            close.push(c);
        }
        for i in 100..120 {
            let x = (i - 100) as f64;
            let c = 140.0 + x * 0.55;
            close.push(c);
        }
        for i in 120..140 {
            let x = (i - 120) as f64;
            let c = 150.0 - x * 2.4;
            close.push(c);
        }
        for i in 140..160 {
            let x = (i - 140) as f64;
            let c = 102.0 + x * 1.8;
            close.push(c);
        }

        for (i, &c) in close.iter().enumerate() {
            let wiggle = ((i % 3) as f64) * 0.15;
            high.push(c + 1.6 + wiggle);
            low.push(c - 1.5 - wiggle * 0.8);
        }

        (high, low, close)
    }

    #[test]
    fn exponential_trend_outputs_present() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = sample_hlc();
        let input = ExponentialTrendInput::from_slices(
            &high,
            &low,
            &close,
            ExponentialTrendParams::default(),
        );
        let out = exponential_trend(&input)?;
        assert!(out.uptrend_base.iter().any(|v| v.is_finite()));
        assert!(out.downtrend_base.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn exponential_trend_into_matches_api() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = sample_hlc();
        let input = ExponentialTrendInput::from_slices(
            &high,
            &low,
            &close,
            ExponentialTrendParams::default(),
        );
        let baseline = exponential_trend(&input)?;
        let len = close.len();

        let mut uptrend_base = vec![f64::NAN; len];
        let mut downtrend_base = vec![f64::NAN; len];
        let mut uptrend_extension = vec![f64::NAN; len];
        let mut downtrend_extension = vec![f64::NAN; len];
        let mut bullish_change = vec![f64::NAN; len];
        let mut bearish_change = vec![f64::NAN; len];

        exponential_trend_into(
            &mut uptrend_base,
            &mut downtrend_base,
            &mut uptrend_extension,
            &mut downtrend_extension,
            &mut bullish_change,
            &mut bearish_change,
            &input,
        )?;

        assert_series_eq(&uptrend_base, &baseline.uptrend_base);
        assert_series_eq(&downtrend_base, &baseline.downtrend_base);
        assert_series_eq(&uptrend_extension, &baseline.uptrend_extension);
        assert_series_eq(&downtrend_extension, &baseline.downtrend_extension);
        assert_series_eq(&bullish_change, &baseline.bullish_change);
        assert_series_eq(&bearish_change, &baseline.bearish_change);
        Ok(())
    }

    #[test]
    fn exponential_trend_stream_matches_api() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = sample_hlc();
        let params = ExponentialTrendParams::default();
        let input = ExponentialTrendInput::from_slices(&high, &low, &close, params.clone());
        let batch = exponential_trend(&input)?;
        let mut stream = ExponentialTrendStream::try_new(params)?;

        let mut uptrend_base = vec![f64::NAN; close.len()];
        let mut downtrend_base = vec![f64::NAN; close.len()];
        let mut uptrend_extension = vec![f64::NAN; close.len()];
        let mut downtrend_extension = vec![f64::NAN; close.len()];
        let mut bullish_change = vec![f64::NAN; close.len()];
        let mut bearish_change = vec![f64::NAN; close.len()];

        for i in 0..close.len() {
            if let Some((ub, db, ue, de, bc, brc)) = stream.update(high[i], low[i], close[i]) {
                uptrend_base[i] = ub;
                downtrend_base[i] = db;
                uptrend_extension[i] = ue;
                downtrend_extension[i] = de;
                bullish_change[i] = bc;
                bearish_change[i] = brc;
            }
        }

        assert_series_eq(&uptrend_base, &batch.uptrend_base);
        assert_series_eq(&downtrend_base, &batch.downtrend_base);
        assert_series_eq(&uptrend_extension, &batch.uptrend_extension);
        assert_series_eq(&downtrend_extension, &batch.downtrend_extension);
        assert_series_eq(&bullish_change, &batch.bullish_change);
        assert_series_eq(&bearish_change, &batch.bearish_change);
        Ok(())
    }

    #[test]
    fn exponential_trend_batch_matches_single() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = sample_hlc();
        let batch = exponential_trend_batch_with_kernel(
            &high,
            &low,
            &close,
            &ExponentialTrendBatchRange {
                exp_rate: (DEFAULT_EXP_RATE, DEFAULT_EXP_RATE, 0.0),
                initial_distance: (DEFAULT_INITIAL_DISTANCE, DEFAULT_INITIAL_DISTANCE, 0.0),
                width_multiplier: (DEFAULT_WIDTH_MULTIPLIER, DEFAULT_WIDTH_MULTIPLIER, 0.0),
            },
            Kernel::ScalarBatch,
        )?;
        let input = ExponentialTrendInput::from_slices(
            &high,
            &low,
            &close,
            ExponentialTrendParams::default(),
        );
        let single = exponential_trend(&input)?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_eq(&batch.uptrend_base[..close.len()], &single.uptrend_base);
        Ok(())
    }

    #[test]
    fn exponential_trend_invalid_exp_rate() {
        let (high, low, close) = sample_hlc();
        let input = ExponentialTrendInput::from_slices(
            &high,
            &low,
            &close,
            ExponentialTrendParams {
                exp_rate: Some(0.8),
                initial_distance: Some(DEFAULT_INITIAL_DISTANCE),
                width_multiplier: Some(DEFAULT_WIDTH_MULTIPLIER),
            },
        );
        let err = exponential_trend(&input).unwrap_err();
        assert!(matches!(err, ExponentialTrendError::InvalidExpRate { .. }));
    }
}
