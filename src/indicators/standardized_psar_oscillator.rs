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
    alloc_uninit_f64, alloc_with_nan_prefix, detect_best_batch_kernel, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn standardized_psar_oscillator_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    start: f64,
    increment: f64,
    maximum: f64,
    standardization_length: usize,
    wma_length: usize,
    wma_lag: usize,
    pivot_left: usize,
    pivot_right: usize,
    plot_bullish: bool,
    plot_bearish: bool,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = standardized_psar_oscillator_js(
        high,
        low,
        close,
        start,
        increment,
        maximum,
        standardization_length,
        wma_length,
        wma_lag,
        pivot_left,
        pivot_right,
        plot_bullish,
        plot_bearish,
    )?;
    crate::write_wasm_object_f64_outputs("standardized_psar_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn standardized_psar_oscillator_batch_unified_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = standardized_psar_oscillator_batch_unified_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "standardized_psar_oscillator_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_START: f64 = 0.02;
const DEFAULT_INCREMENT: f64 = 0.0005;
const DEFAULT_MAXIMUM: f64 = 0.2;
const DEFAULT_STANDARDIZATION_LENGTH: usize = 21;
const DEFAULT_WMA_LENGTH: usize = 40;
const DEFAULT_WMA_LAG: usize = 3;
const DEFAULT_PIVOT_LEFT: usize = 15;
const DEFAULT_PIVOT_RIGHT: usize = 1;
const DEFAULT_PLOT_BULLISH: bool = true;
const DEFAULT_PLOT_BEARISH: bool = true;
const REVERSAL_LEVEL: f64 = 600.0;
const REVERSAL_MARKER: f64 = 900.0;
const MAX_PIVOT_BARS: usize = 80;

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

#[derive(Debug, Clone)]
pub enum StandardizedPsarOscillatorData<'a> {
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
pub struct StandardizedPsarOscillatorOutput {
    pub oscillator: Vec<f64>,
    pub ma: Vec<f64>,
    pub bullish_reversal: Vec<f64>,
    pub bearish_reversal: Vec<f64>,
    pub regular_bullish: Vec<f64>,
    pub regular_bearish: Vec<f64>,
    pub bullish_weakening: Vec<f64>,
    pub bearish_weakening: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct StandardizedPsarOscillatorParams {
    pub start: Option<f64>,
    pub increment: Option<f64>,
    pub maximum: Option<f64>,
    pub standardization_length: Option<usize>,
    pub wma_length: Option<usize>,
    pub wma_lag: Option<usize>,
    pub pivot_left: Option<usize>,
    pub pivot_right: Option<usize>,
    pub plot_bullish: Option<bool>,
    pub plot_bearish: Option<bool>,
}

impl Default for StandardizedPsarOscillatorParams {
    fn default() -> Self {
        Self {
            start: Some(DEFAULT_START),
            increment: Some(DEFAULT_INCREMENT),
            maximum: Some(DEFAULT_MAXIMUM),
            standardization_length: Some(DEFAULT_STANDARDIZATION_LENGTH),
            wma_length: Some(DEFAULT_WMA_LENGTH),
            wma_lag: Some(DEFAULT_WMA_LAG),
            pivot_left: Some(DEFAULT_PIVOT_LEFT),
            pivot_right: Some(DEFAULT_PIVOT_RIGHT),
            plot_bullish: Some(DEFAULT_PLOT_BULLISH),
            plot_bearish: Some(DEFAULT_PLOT_BEARISH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StandardizedPsarOscillatorInput<'a> {
    pub data: StandardizedPsarOscillatorData<'a>,
    pub params: StandardizedPsarOscillatorParams,
}

impl<'a> StandardizedPsarOscillatorInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles, params: StandardizedPsarOscillatorParams) -> Self {
        Self {
            data: StandardizedPsarOscillatorData::Candles { candles },
            params,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: StandardizedPsarOscillatorParams,
    ) -> Self {
        Self {
            data: StandardizedPsarOscillatorData::Slices { high, low, close },
            params,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, StandardizedPsarOscillatorParams::default())
    }

    #[inline(always)]
    pub fn get_start(&self) -> f64 {
        self.params.start.unwrap_or(DEFAULT_START)
    }

    #[inline(always)]
    pub fn get_increment(&self) -> f64 {
        self.params.increment.unwrap_or(DEFAULT_INCREMENT)
    }

    #[inline(always)]
    pub fn get_maximum(&self) -> f64 {
        self.params.maximum.unwrap_or(DEFAULT_MAXIMUM)
    }

    #[inline(always)]
    pub fn get_standardization_length(&self) -> usize {
        self.params
            .standardization_length
            .unwrap_or(DEFAULT_STANDARDIZATION_LENGTH)
    }

    #[inline(always)]
    pub fn get_wma_length(&self) -> usize {
        self.params.wma_length.unwrap_or(DEFAULT_WMA_LENGTH)
    }

    #[inline(always)]
    pub fn get_wma_lag(&self) -> usize {
        self.params.wma_lag.unwrap_or(DEFAULT_WMA_LAG)
    }

    #[inline(always)]
    pub fn get_pivot_left(&self) -> usize {
        self.params.pivot_left.unwrap_or(DEFAULT_PIVOT_LEFT)
    }

    #[inline(always)]
    pub fn get_pivot_right(&self) -> usize {
        self.params.pivot_right.unwrap_or(DEFAULT_PIVOT_RIGHT)
    }

    #[inline(always)]
    pub fn get_plot_bullish(&self) -> bool {
        self.params.plot_bullish.unwrap_or(DEFAULT_PLOT_BULLISH)
    }

    #[inline(always)]
    pub fn get_plot_bearish(&self) -> bool {
        self.params.plot_bearish.unwrap_or(DEFAULT_PLOT_BEARISH)
    }

    #[inline(always)]
    fn as_hlc(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            StandardizedPsarOscillatorData::Candles { candles } => (
                high_source(candles),
                low_source(candles),
                close_source(candles),
            ),
            StandardizedPsarOscillatorData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

impl<'a> AsRef<[f64]> for StandardizedPsarOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        self.as_hlc().2
    }
}

#[derive(Clone, Debug)]
pub struct StandardizedPsarOscillatorBuilder {
    start: Option<f64>,
    increment: Option<f64>,
    maximum: Option<f64>,
    standardization_length: Option<usize>,
    wma_length: Option<usize>,
    wma_lag: Option<usize>,
    pivot_left: Option<usize>,
    pivot_right: Option<usize>,
    plot_bullish: Option<bool>,
    plot_bearish: Option<bool>,
    kernel: Kernel,
}

impl Default for StandardizedPsarOscillatorBuilder {
    fn default() -> Self {
        Self {
            start: None,
            increment: None,
            maximum: None,
            standardization_length: None,
            wma_length: None,
            wma_lag: None,
            pivot_left: None,
            pivot_right: None,
            plot_bullish: None,
            plot_bearish: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StandardizedPsarOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn start(mut self, value: f64) -> Self {
        self.start = Some(value);
        self
    }

    #[inline(always)]
    pub fn increment(mut self, value: f64) -> Self {
        self.increment = Some(value);
        self
    }

    #[inline(always)]
    pub fn maximum(mut self, value: f64) -> Self {
        self.maximum = Some(value);
        self
    }

    #[inline(always)]
    pub fn standardization_length(mut self, value: usize) -> Self {
        self.standardization_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn wma_length(mut self, value: usize) -> Self {
        self.wma_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn wma_lag(mut self, value: usize) -> Self {
        self.wma_lag = Some(value);
        self
    }

    #[inline(always)]
    pub fn pivot_left(mut self, value: usize) -> Self {
        self.pivot_left = Some(value);
        self
    }

    #[inline(always)]
    pub fn pivot_right(mut self, value: usize) -> Self {
        self.pivot_right = Some(value);
        self
    }

    #[inline(always)]
    pub fn plot_bullish(mut self, value: bool) -> Self {
        self.plot_bullish = Some(value);
        self
    }

    #[inline(always)]
    pub fn plot_bearish(mut self, value: bool) -> Self {
        self.plot_bearish = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> StandardizedPsarOscillatorParams {
        StandardizedPsarOscillatorParams {
            start: self.start,
            increment: self.increment,
            maximum: self.maximum,
            standardization_length: self.standardization_length,
            wma_length: self.wma_length,
            wma_lag: self.wma_lag,
            pivot_left: self.pivot_left,
            pivot_right: self.pivot_right,
            plot_bullish: self.plot_bullish,
            plot_bearish: self.plot_bearish,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<StandardizedPsarOscillatorOutput, StandardizedPsarOscillatorError> {
        let kernel = self.kernel;
        let params = self.params();
        standardized_psar_oscillator_with_kernel(
            &StandardizedPsarOscillatorInput::from_candles(candles, params),
            kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<StandardizedPsarOscillatorOutput, StandardizedPsarOscillatorError> {
        let kernel = self.kernel;
        let params = self.params();
        standardized_psar_oscillator_with_kernel(
            &StandardizedPsarOscillatorInput::from_slices(high, low, close, params),
            kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<StandardizedPsarOscillatorStream, StandardizedPsarOscillatorError> {
        StandardizedPsarOscillatorStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum StandardizedPsarOscillatorError {
    #[error("standardized_psar_oscillator: input data slice is empty.")]
    EmptyInputData,
    #[error("standardized_psar_oscillator: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "standardized_psar_oscillator: inconsistent data lengths - high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    DataLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("standardized_psar_oscillator: invalid start: {start}")]
    InvalidStart { start: f64 },
    #[error("standardized_psar_oscillator: invalid increment: {increment}")]
    InvalidIncrement { increment: f64 },
    #[error("standardized_psar_oscillator: invalid maximum: {maximum}")]
    InvalidMaximum { maximum: f64 },
    #[error(
        "standardized_psar_oscillator: invalid standardization_length: {standardization_length}, data length = {data_len}"
    )]
    InvalidStandardizationLength {
        standardization_length: usize,
        data_len: usize,
    },
    #[error(
        "standardized_psar_oscillator: invalid wma_length: {wma_length}, data length = {data_len}"
    )]
    InvalidWmaLength { wma_length: usize, data_len: usize },
    #[error("standardized_psar_oscillator: invalid wma_lag: {wma_lag}")]
    InvalidWmaLag { wma_lag: usize },
    #[error(
        "standardized_psar_oscillator: invalid pivot_left: {pivot_left}, data length = {data_len}"
    )]
    InvalidPivotLeft { pivot_left: usize, data_len: usize },
    #[error(
        "standardized_psar_oscillator: invalid pivot_right: {pivot_right}, data length = {data_len}"
    )]
    InvalidPivotRight { pivot_right: usize, data_len: usize },
    #[error(
        "standardized_psar_oscillator: not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "standardized_psar_oscillator: output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "standardized_psar_oscillator: invalid range for {axis}: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        axis: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("standardized_psar_oscillator: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    start: f64,
    increment: f64,
    maximum: f64,
    standardization_length: usize,
    wma_length: usize,
    wma_lag: usize,
    pivot_left: usize,
    pivot_right: usize,
    plot_bullish: bool,
    plot_bearish: bool,
    warmup: usize,
    all_finite: bool,
}

#[inline(always)]
fn normalize_single_kernel(_kernel: Kernel) -> Kernel {
    Kernel::Scalar
}

#[inline(always)]
fn validate_params(
    start: f64,
    increment: f64,
    maximum: f64,
    standardization_length: usize,
    wma_length: usize,
    wma_lag: usize,
    pivot_left: usize,
    pivot_right: usize,
    data_len: usize,
) -> Result<(), StandardizedPsarOscillatorError> {
    if !start.is_finite() || start <= 0.0 {
        return Err(StandardizedPsarOscillatorError::InvalidStart { start });
    }
    if !increment.is_finite() || increment <= 0.0 {
        return Err(StandardizedPsarOscillatorError::InvalidIncrement { increment });
    }
    if !maximum.is_finite() || maximum <= 0.0 || maximum < start {
        return Err(StandardizedPsarOscillatorError::InvalidMaximum { maximum });
    }
    if standardization_length == 0 || standardization_length > data_len {
        return Err(
            StandardizedPsarOscillatorError::InvalidStandardizationLength {
                standardization_length,
                data_len,
            },
        );
    }
    if wma_length == 0 || wma_length > data_len {
        return Err(StandardizedPsarOscillatorError::InvalidWmaLength {
            wma_length,
            data_len,
        });
    }
    if wma_lag > data_len {
        return Err(StandardizedPsarOscillatorError::InvalidWmaLag { wma_lag });
    }
    if pivot_left == 0 || pivot_left > data_len {
        return Err(StandardizedPsarOscillatorError::InvalidPivotLeft {
            pivot_left,
            data_len,
        });
    }
    if pivot_right > data_len {
        return Err(StandardizedPsarOscillatorError::InvalidPivotRight {
            pivot_right,
            data_len,
        });
    }
    Ok(())
}

#[inline(always)]
fn analyze_valid_segments(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(usize, usize), StandardizedPsarOscillatorError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(StandardizedPsarOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(StandardizedPsarOscillatorError::DataLengthMismatch {
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
        Some(first) => Ok((first, max_run)),
        None => Err(StandardizedPsarOscillatorError::AllValuesNaN),
    }
}

#[inline(always)]
fn required_valid_bars(standardization_length: usize, wma_length: usize) -> usize {
    standardization_length.max(2) + wma_length - 1
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a StandardizedPsarOscillatorInput<'a>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, StandardizedPsarOscillatorError> {
    let _chosen = normalize_single_kernel(kernel);
    let (high, low, close) = input.as_hlc();
    let start = input.get_start();
    let increment = input.get_increment();
    let maximum = input.get_maximum();
    let standardization_length = input.get_standardization_length();
    let wma_length = input.get_wma_length();
    let wma_lag = input.get_wma_lag();
    let pivot_left = input.get_pivot_left();
    let pivot_right = input.get_pivot_right();
    validate_params(
        start,
        increment,
        maximum,
        standardization_length,
        wma_length,
        wma_lag,
        pivot_left,
        pivot_right,
        close.len(),
    )?;
    let (first_valid, max_run) = analyze_valid_segments(high, low, close)?;
    let needed = required_valid_bars(standardization_length, wma_length);
    if max_run < needed {
        return Err(StandardizedPsarOscillatorError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }
    Ok(PreparedInput {
        high,
        low,
        close,
        start,
        increment,
        maximum,
        standardization_length,
        wma_length,
        wma_lag,
        pivot_left,
        pivot_right,
        plot_bullish: input.get_plot_bullish(),
        plot_bearish: input.get_plot_bearish(),
        warmup: first_valid + needed - 1,
        all_finite: first_valid == 0 && max_run == close.len(),
    })
}

#[derive(Clone, Debug)]
struct EmaState {
    period: usize,
    alpha: f64,
    beta: f64,
    count: usize,
    mean: f64,
}

impl EmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            alpha,
            beta: 1.0 - alpha,
            count: 0,
            mean: f64::NAN,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.mean = f64::NAN;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        self.count += 1;
        if self.count == 1 {
            self.mean = value;
        } else if self.count <= self.period {
            let inv = 1.0 / self.count as f64;
            self.mean = (value - self.mean).mul_add(inv, self.mean);
        } else {
            self.mean = self.beta.mul_add(self.mean, self.alpha * value);
        }
        if self.count >= self.period {
            Some(self.mean)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct WmaState {
    buffer: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
    weighted_sum: f64,
    denominator: f64,
}

impl WmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            buffer: vec![0.0; period],
            head: 0,
            count: 0,
            sum: 0.0,
            weighted_sum: 0.0,
            denominator: (period * (period + 1) / 2) as f64,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
        self.weighted_sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        let len = self.buffer.len();
        if self.count < len {
            self.count += 1;
            self.buffer[self.head] = value;
            self.head += 1;
            if self.head == len {
                self.head = 0;
            }
            self.sum += value;
            self.weighted_sum += value * self.count as f64;
            if self.count == len {
                Some(self.weighted_sum / self.denominator)
            } else {
                None
            }
        } else {
            let oldest = self.buffer[self.head];
            let old_sum = self.sum;
            self.weighted_sum = self.weighted_sum - old_sum + value * len as f64;
            self.sum = old_sum - oldest + value;
            self.buffer[self.head] = value;
            self.head += 1;
            if self.head == len {
                self.head = 0;
            }
            Some(self.weighted_sum / self.denominator)
        }
    }
}

#[derive(Clone, Debug)]
struct PsarTrendState {
    trend_up: bool,
    sar: f64,
    ep: f64,
    acc: f64,
    prev_high: f64,
    prev_high2: f64,
    prev_low: f64,
    prev_low2: f64,
}

#[derive(Clone, Debug)]
struct PsarState {
    start: f64,
    increment: f64,
    maximum: f64,
    state: Option<PsarTrendState>,
    idx: usize,
}

impl PsarState {
    #[inline(always)]
    fn new(start: f64, increment: f64, maximum: f64) -> Self {
        Self {
            start,
            increment,
            maximum,
            state: None,
            idx: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.state = None;
        self.idx = 0;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        match self.state.as_mut() {
            None => {
                self.state = Some(PsarTrendState {
                    trend_up: false,
                    sar: f64::NAN,
                    ep: f64::NAN,
                    acc: self.start,
                    prev_high: high,
                    prev_high2: high,
                    prev_low: low,
                    prev_low2: low,
                });
                self.idx = 1;
                None
            }
            Some(st) if self.idx == 1 => {
                let trend_up = high > st.prev_high;
                let sar = if trend_up { st.prev_low } else { st.prev_high };
                let ep = if trend_up { high } else { low };

                st.prev_high2 = st.prev_high;
                st.prev_low2 = st.prev_low;
                st.prev_high = high;
                st.prev_low = low;
                st.trend_up = trend_up;
                st.sar = sar;
                st.ep = ep;
                st.acc = self.start;
                self.idx = 2;
                Some(sar)
            }
            Some(st) => {
                let mut next_sar = st.acc.mul_add(st.ep - st.sar, st.sar);

                if st.trend_up {
                    if low < next_sar {
                        st.trend_up = false;
                        next_sar = st.ep;
                        st.ep = low;
                        st.acc = self.start;
                    } else {
                        if high > st.ep {
                            st.ep = high;
                            st.acc = (st.acc + self.increment).min(self.maximum);
                        }
                        next_sar = next_sar.min(st.prev_low.min(st.prev_low2));
                    }
                } else if high > next_sar {
                    st.trend_up = true;
                    next_sar = st.ep;
                    st.ep = high;
                    st.acc = self.start;
                } else {
                    if low < st.ep {
                        st.ep = low;
                        st.acc = (st.acc + self.increment).min(self.maximum);
                    }
                    next_sar = next_sar.max(st.prev_high.max(st.prev_high2));
                }

                st.prev_high2 = st.prev_high;
                st.prev_low2 = st.prev_low;
                st.prev_high = high;
                st.prev_low = low;
                st.sar = next_sar;
                self.idx += 1;
                Some(next_sar)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PivotEvent {
    confirm_index: usize,
    oscillator: f64,
    price: f64,
}

type StandardizedPsarValues = (f64, f64, f64, f64, f64, f64, f64, f64);

#[derive(Clone, Debug)]
struct StandardizedPsarOscillatorState {
    psar: PsarState,
    range_ema: EmaState,
    wma: WmaState,
    wma_lag: usize,
    pivot_left: usize,
    pivot_right: usize,
    plot_bullish: bool,
    plot_bearish: bool,
    oscillator_history: Vec<f64>,
    ma_history: Vec<f64>,
    ma_lag_head: usize,
    low_history: Vec<f64>,
    high_history: Vec<f64>,
    previous_low_pivot: Option<PivotEvent>,
    previous_high_pivot: Option<PivotEvent>,
    previous_oscillator: f64,
}

impl StandardizedPsarOscillatorState {
    #[inline(always)]
    fn new(
        start: f64,
        increment: f64,
        maximum: f64,
        standardization_length: usize,
        wma_length: usize,
        wma_lag: usize,
        pivot_left: usize,
        pivot_right: usize,
        plot_bullish: bool,
        plot_bearish: bool,
    ) -> Self {
        Self {
            psar: PsarState::new(start, increment, maximum),
            range_ema: EmaState::new(standardization_length),
            wma: WmaState::new(wma_length),
            wma_lag,
            pivot_left,
            pivot_right,
            plot_bullish,
            plot_bearish,
            oscillator_history: Vec::new(),
            ma_history: Vec::new(),
            ma_lag_head: 0,
            low_history: Vec::new(),
            high_history: Vec::new(),
            previous_low_pivot: None,
            previous_high_pivot: None,
            previous_oscillator: f64::NAN,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.psar.reset();
        self.range_ema.reset();
        self.wma.reset();
        self.oscillator_history.clear();
        self.ma_history.clear();
        self.ma_lag_head = 0;
        self.low_history.clear();
        self.high_history.clear();
        self.previous_low_pivot = None;
        self.previous_high_pivot = None;
        self.previous_oscillator = f64::NAN;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<StandardizedPsarValues> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.reset();
            return None;
        }

        self.update_finite(high, low, close)
    }

    #[inline(always)]
    fn update_finite(&mut self, high: f64, low: f64, close: f64) -> Option<StandardizedPsarValues> {
        let psar = self.psar.update(high, low);
        let range = self.range_ema.update(high - low);
        let oscillator = match (psar, range) {
            (Some(sar), Some(ema_range)) if ema_range.is_finite() && ema_range != 0.0 => {
                (close - sar) / ema_range * 100.0
            }
            _ => f64::NAN,
        };

        let ma = if oscillator.is_finite() {
            self.wma.update(oscillator).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        let bearish_reversal = if self.previous_oscillator.is_finite()
            && oscillator.is_finite()
            && self.previous_oscillator >= REVERSAL_LEVEL
            && oscillator < REVERSAL_LEVEL
        {
            REVERSAL_MARKER
        } else {
            f64::NAN
        };

        let bullish_reversal = if self.previous_oscillator.is_finite()
            && oscillator.is_finite()
            && self.previous_oscillator <= -REVERSAL_LEVEL
            && oscillator > -REVERSAL_LEVEL
        {
            -REVERSAL_MARKER
        } else {
            f64::NAN
        };

        let lag_ma = if self.wma_lag == 0 || self.ma_history.len() < self.wma_lag {
            f64::NAN
        } else {
            self.ma_history[self.ma_lag_head]
        };

        let bullish_weakening = if ma.is_finite() && lag_ma.is_finite() {
            if oscillator > 0.0 && ma < lag_ma {
                1.0
            } else {
                0.0
            }
        } else {
            f64::NAN
        };

        let bearish_weakening = if ma.is_finite() && lag_ma.is_finite() {
            if oscillator < 0.0 && ma > lag_ma {
                1.0
            } else {
                0.0
            }
        } else {
            f64::NAN
        };

        self.previous_oscillator = oscillator;
        self.oscillator_history.push(oscillator);
        if self.wma_lag > 0 {
            if self.ma_history.len() < self.wma_lag {
                self.ma_history.push(ma);
            } else {
                self.ma_history[self.ma_lag_head] = ma;
                self.ma_lag_head += 1;
                if self.ma_lag_head == self.wma_lag {
                    self.ma_lag_head = 0;
                }
            }
        }
        self.low_history.push(low);
        self.high_history.push(high);

        let mut regular_bullish = f64::NAN;
        let mut regular_bearish = f64::NAN;
        let len = self.oscillator_history.len();
        let needed = self.pivot_left + self.pivot_right + 1;

        if len >= needed {
            let center = len - 1 - self.pivot_right;
            let start = center - self.pivot_left;
            let end = center + self.pivot_right;
            let center_oscillator = self.oscillator_history[center];

            if center_oscillator.is_finite() {
                let mut pivot_low = true;
                let mut pivot_high = true;

                for idx in start..=end {
                    let value = self.oscillator_history[idx];
                    if !value.is_finite() {
                        pivot_low = false;
                        pivot_high = false;
                        break;
                    }
                    if idx != center {
                        if value < center_oscillator {
                            pivot_low = false;
                        }
                        if value > center_oscillator {
                            pivot_high = false;
                        }
                    }
                    if !pivot_low && !pivot_high {
                        break;
                    }
                }

                let confirm_index = len - 1;

                if pivot_low {
                    let event = PivotEvent {
                        confirm_index,
                        oscillator: center_oscillator,
                        price: self.low_history[center],
                    };
                    if self.plot_bullish {
                        if let Some(previous) = self.previous_low_pivot {
                            let bars = event.confirm_index.saturating_sub(previous.confirm_index);
                            if (1..=MAX_PIVOT_BARS).contains(&bars)
                                && event.oscillator > previous.oscillator
                                && event.price < previous.price
                            {
                                regular_bullish = event.oscillator;
                            }
                        }
                    }
                    self.previous_low_pivot = Some(event);
                }

                if pivot_high {
                    let event = PivotEvent {
                        confirm_index,
                        oscillator: center_oscillator,
                        price: self.high_history[center],
                    };
                    if self.plot_bearish {
                        if let Some(previous) = self.previous_high_pivot {
                            let bars = event.confirm_index.saturating_sub(previous.confirm_index);
                            if (1..=MAX_PIVOT_BARS).contains(&bars)
                                && event.oscillator < previous.oscillator
                                && event.price > previous.price
                            {
                                regular_bearish = event.oscillator;
                            }
                        }
                    }
                    self.previous_high_pivot = Some(event);
                }
            }
        }

        if oscillator.is_finite() {
            Some((
                oscillator,
                ma,
                bullish_reversal,
                bearish_reversal,
                regular_bullish,
                regular_bearish,
                bullish_weakening,
                bearish_weakening,
            ))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct StandardizedPsarOscillatorStream {
    params: StandardizedPsarOscillatorParams,
    state: StandardizedPsarOscillatorState,
}

impl StandardizedPsarOscillatorStream {
    #[inline(always)]
    pub fn try_new(
        params: StandardizedPsarOscillatorParams,
    ) -> Result<Self, StandardizedPsarOscillatorError> {
        let start = params.start.unwrap_or(DEFAULT_START);
        let increment = params.increment.unwrap_or(DEFAULT_INCREMENT);
        let maximum = params.maximum.unwrap_or(DEFAULT_MAXIMUM);
        let standardization_length = params
            .standardization_length
            .unwrap_or(DEFAULT_STANDARDIZATION_LENGTH);
        let wma_length = params.wma_length.unwrap_or(DEFAULT_WMA_LENGTH);
        let wma_lag = params.wma_lag.unwrap_or(DEFAULT_WMA_LAG);
        let pivot_left = params.pivot_left.unwrap_or(DEFAULT_PIVOT_LEFT);
        let pivot_right = params.pivot_right.unwrap_or(DEFAULT_PIVOT_RIGHT);
        validate_params(
            start,
            increment,
            maximum,
            standardization_length,
            wma_length,
            wma_lag,
            pivot_left,
            pivot_right,
            usize::MAX,
        )?;
        Ok(Self {
            state: StandardizedPsarOscillatorState::new(
                start,
                increment,
                maximum,
                standardization_length,
                wma_length,
                wma_lag,
                pivot_left,
                pivot_right,
                params.plot_bullish.unwrap_or(DEFAULT_PLOT_BULLISH),
                params.plot_bearish.unwrap_or(DEFAULT_PLOT_BEARISH),
            ),
            params,
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64)> {
        self.state.update(high, low, close)
    }

    #[inline(always)]
    pub fn params(&self) -> &StandardizedPsarOscillatorParams {
        &self.params
    }
}

#[derive(Clone, Debug)]
pub struct StandardizedPsarOscillatorBatchRange {
    pub start: (f64, f64, f64),
    pub increment: (f64, f64, f64),
    pub maximum: (f64, f64, f64),
    pub standardization_length: (usize, usize, usize),
    pub wma_length: (usize, usize, usize),
    pub wma_lag: (usize, usize, usize),
    pub pivot_left: (usize, usize, usize),
    pub pivot_right: (usize, usize, usize),
    pub plot_bullish: bool,
    pub plot_bearish: bool,
}

impl Default for StandardizedPsarOscillatorBatchRange {
    fn default() -> Self {
        Self {
            start: (DEFAULT_START, DEFAULT_START, 0.0),
            increment: (DEFAULT_INCREMENT, DEFAULT_INCREMENT, 0.0),
            maximum: (DEFAULT_MAXIMUM, DEFAULT_MAXIMUM, 0.0),
            standardization_length: (
                DEFAULT_STANDARDIZATION_LENGTH,
                DEFAULT_STANDARDIZATION_LENGTH,
                0,
            ),
            wma_length: (DEFAULT_WMA_LENGTH, DEFAULT_WMA_LENGTH, 0),
            wma_lag: (DEFAULT_WMA_LAG, DEFAULT_WMA_LAG, 0),
            pivot_left: (DEFAULT_PIVOT_LEFT, DEFAULT_PIVOT_LEFT, 0),
            pivot_right: (DEFAULT_PIVOT_RIGHT, DEFAULT_PIVOT_RIGHT, 0),
            plot_bullish: DEFAULT_PLOT_BULLISH,
            plot_bearish: DEFAULT_PLOT_BEARISH,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StandardizedPsarOscillatorBatchBuilder {
    range: StandardizedPsarOscillatorBatchRange,
    kernel: Kernel,
}

impl StandardizedPsarOscillatorBatchBuilder {
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
    pub fn start_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.start = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn increment_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.increment = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn maximum_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.maximum = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn standardization_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.standardization_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn wma_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.wma_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn wma_lag_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.wma_lag = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn pivot_left_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.pivot_left = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn pivot_right_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.pivot_right = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn plot_bullish(mut self, value: bool) -> Self {
        self.range.plot_bullish = value;
        self
    }

    #[inline(always)]
    pub fn plot_bearish(mut self, value: bool) -> Self {
        self.range.plot_bearish = value;
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<StandardizedPsarOscillatorBatchOutput, StandardizedPsarOscillatorError> {
        standardized_psar_oscillator_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<StandardizedPsarOscillatorBatchOutput, StandardizedPsarOscillatorError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close)
    }
}

#[derive(Clone, Debug)]
pub struct StandardizedPsarOscillatorBatchOutput {
    pub oscillator: Vec<f64>,
    pub ma: Vec<f64>,
    pub bullish_reversal: Vec<f64>,
    pub bearish_reversal: Vec<f64>,
    pub regular_bullish: Vec<f64>,
    pub regular_bearish: Vec<f64>,
    pub bullish_weakening: Vec<f64>,
    pub bearish_weakening: Vec<f64>,
    pub combos: Vec<StandardizedPsarOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn axis_usize(
    axis: &'static str,
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, StandardizedPsarOscillatorError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            match value.checked_add(step) {
                Some(next) if next > value => value = next,
                _ => break,
            }
        }
    } else {
        let mut value = start;
        while value >= end {
            out.push(value);
            if value == end {
                break;
            }
            match value.checked_sub(step) {
                Some(next) if next < value => value = next,
                _ => break,
            }
        }
    }

    if out.is_empty() || !out.last().is_some_and(|value| *value == end) {
        return Err(StandardizedPsarOscillatorError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn axis_float(
    axis: &'static str,
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, StandardizedPsarOscillatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(StandardizedPsarOscillatorError::InvalidRange {
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
        return Err(StandardizedPsarOscillatorError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }

    let eps = step.abs() * 1e-9 + 1e-12;
    let mut out = Vec::new();
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
        return Err(StandardizedPsarOscillatorError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_standardized_psar_oscillator(
    sweep: &StandardizedPsarOscillatorBatchRange,
) -> Result<Vec<StandardizedPsarOscillatorParams>, StandardizedPsarOscillatorError> {
    let starts = axis_float("start", sweep.start)?;
    let increments = axis_float("increment", sweep.increment)?;
    let maximums = axis_float("maximum", sweep.maximum)?;
    let standardization_lengths =
        axis_usize("standardization_length", sweep.standardization_length)?;
    let wma_lengths = axis_usize("wma_length", sweep.wma_length)?;
    let wma_lags = axis_usize("wma_lag", sweep.wma_lag)?;
    let pivot_lefts = axis_usize("pivot_left", sweep.pivot_left)?;
    let pivot_rights = axis_usize("pivot_right", sweep.pivot_right)?;

    let total = starts
        .len()
        .checked_mul(increments.len())
        .and_then(|value| value.checked_mul(maximums.len()))
        .and_then(|value| value.checked_mul(standardization_lengths.len()))
        .and_then(|value| value.checked_mul(wma_lengths.len()))
        .and_then(|value| value.checked_mul(wma_lags.len()))
        .and_then(|value| value.checked_mul(pivot_lefts.len()))
        .and_then(|value| value.checked_mul(pivot_rights.len()))
        .ok_or(StandardizedPsarOscillatorError::InvalidRange {
            axis: "grid",
            start: "overflow".to_string(),
            end: "overflow".to_string(),
            step: "overflow".to_string(),
        })?;

    let mut out = Vec::with_capacity(total);
    for &start in &starts {
        for &increment in &increments {
            for &maximum in &maximums {
                for &standardization_length in &standardization_lengths {
                    for &wma_length in &wma_lengths {
                        for &wma_lag in &wma_lags {
                            for &pivot_left in &pivot_lefts {
                                for &pivot_right in &pivot_rights {
                                    out.push(StandardizedPsarOscillatorParams {
                                        start: Some(start),
                                        increment: Some(increment),
                                        maximum: Some(maximum),
                                        standardization_length: Some(standardization_length),
                                        wma_length: Some(wma_length),
                                        wma_lag: Some(wma_lag),
                                        pivot_left: Some(pivot_left),
                                        pivot_right: Some(pivot_right),
                                        plot_bullish: Some(sweep.plot_bullish),
                                        plot_bearish: Some(sweep.plot_bearish),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn compute_row(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    all_finite: bool,
    params: &StandardizedPsarOscillatorParams,
    oscillator_out: &mut [f64],
    ma_out: &mut [f64],
    bullish_reversal_out: &mut [f64],
    bearish_reversal_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    bullish_weakening_out: &mut [f64],
    bearish_weakening_out: &mut [f64],
) -> Result<(), StandardizedPsarOscillatorError> {
    let len = close.len();
    for out in [
        &mut *oscillator_out,
        &mut *ma_out,
        &mut *bullish_reversal_out,
        &mut *bearish_reversal_out,
        &mut *regular_bullish_out,
        &mut *regular_bearish_out,
        &mut *bullish_weakening_out,
        &mut *bearish_weakening_out,
    ] {
        if out.len() != len {
            return Err(StandardizedPsarOscillatorError::OutputLengthMismatch {
                expected: len,
                got: out.len(),
            });
        }
    }

    let mut state = StandardizedPsarOscillatorState::new(
        params.start.unwrap_or(DEFAULT_START),
        params.increment.unwrap_or(DEFAULT_INCREMENT),
        params.maximum.unwrap_or(DEFAULT_MAXIMUM),
        params
            .standardization_length
            .unwrap_or(DEFAULT_STANDARDIZATION_LENGTH),
        params.wma_length.unwrap_or(DEFAULT_WMA_LENGTH),
        params.wma_lag.unwrap_or(DEFAULT_WMA_LAG),
        params.pivot_left.unwrap_or(DEFAULT_PIVOT_LEFT),
        params.pivot_right.unwrap_or(DEFAULT_PIVOT_RIGHT),
        params.plot_bullish.unwrap_or(DEFAULT_PLOT_BULLISH),
        params.plot_bearish.unwrap_or(DEFAULT_PLOT_BEARISH),
    );
    state.oscillator_history.reserve_exact(len);
    state.ma_history.reserve_exact(state.wma_lag);
    state.low_history.reserve_exact(len);
    state.high_history.reserve_exact(len);

    if all_finite {
        for i in 0..len {
            write_standardized_psar_values(
                i,
                state.update_finite(high[i], low[i], close[i]),
                oscillator_out,
                ma_out,
                bullish_reversal_out,
                bearish_reversal_out,
                regular_bullish_out,
                regular_bearish_out,
                bullish_weakening_out,
                bearish_weakening_out,
            );
        }
        return Ok(());
    }

    for i in 0..len {
        write_standardized_psar_values(
            i,
            state.update(high[i], low[i], close[i]),
            oscillator_out,
            ma_out,
            bullish_reversal_out,
            bearish_reversal_out,
            regular_bullish_out,
            regular_bearish_out,
            bullish_weakening_out,
            bearish_weakening_out,
        );
    }

    Ok(())
}

#[inline(always)]
fn write_standardized_psar_values(
    i: usize,
    values: Option<StandardizedPsarValues>,
    oscillator_out: &mut [f64],
    ma_out: &mut [f64],
    bullish_reversal_out: &mut [f64],
    bearish_reversal_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    bullish_weakening_out: &mut [f64],
    bearish_weakening_out: &mut [f64],
) {
    if let Some((
        oscillator,
        ma,
        bullish_reversal,
        bearish_reversal,
        regular_bullish,
        regular_bearish,
        bullish_weakening,
        bearish_weakening,
    )) = values
    {
        oscillator_out[i] = oscillator;
        ma_out[i] = ma;
        bullish_reversal_out[i] = bullish_reversal;
        bearish_reversal_out[i] = bearish_reversal;
        regular_bullish_out[i] = regular_bullish;
        regular_bearish_out[i] = regular_bearish;
        bullish_weakening_out[i] = bullish_weakening;
        bearish_weakening_out[i] = bearish_weakening;
    } else {
        oscillator_out[i] = f64::NAN;
        ma_out[i] = f64::NAN;
        bullish_reversal_out[i] = f64::NAN;
        bearish_reversal_out[i] = f64::NAN;
        regular_bullish_out[i] = f64::NAN;
        regular_bearish_out[i] = f64::NAN;
        bullish_weakening_out[i] = f64::NAN;
        bearish_weakening_out[i] = f64::NAN;
    }
}

#[inline]
pub fn standardized_psar_oscillator(
    input: &StandardizedPsarOscillatorInput,
) -> Result<StandardizedPsarOscillatorOutput, StandardizedPsarOscillatorError> {
    standardized_psar_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn standardized_psar_oscillator_with_kernel(
    input: &StandardizedPsarOscillatorInput,
    kernel: Kernel,
) -> Result<StandardizedPsarOscillatorOutput, StandardizedPsarOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    let len = prepared.close.len();

    let mut oscillator = alloc_uninit_f64(len);
    let mut ma = alloc_uninit_f64(len);
    let mut bullish_reversal = alloc_uninit_f64(len);
    let mut bearish_reversal = alloc_uninit_f64(len);
    let mut regular_bullish = alloc_uninit_f64(len);
    let mut regular_bearish = alloc_uninit_f64(len);
    let mut bullish_weakening = alloc_uninit_f64(len);
    let mut bearish_weakening = alloc_uninit_f64(len);

    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.all_finite,
        &StandardizedPsarOscillatorParams {
            start: Some(prepared.start),
            increment: Some(prepared.increment),
            maximum: Some(prepared.maximum),
            standardization_length: Some(prepared.standardization_length),
            wma_length: Some(prepared.wma_length),
            wma_lag: Some(prepared.wma_lag),
            pivot_left: Some(prepared.pivot_left),
            pivot_right: Some(prepared.pivot_right),
            plot_bullish: Some(prepared.plot_bullish),
            plot_bearish: Some(prepared.plot_bearish),
        },
        &mut oscillator,
        &mut ma,
        &mut bullish_reversal,
        &mut bearish_reversal,
        &mut regular_bullish,
        &mut regular_bearish,
        &mut bullish_weakening,
        &mut bearish_weakening,
    )?;

    Ok(StandardizedPsarOscillatorOutput {
        oscillator,
        ma,
        bullish_reversal,
        bearish_reversal,
        regular_bullish,
        regular_bearish,
        bullish_weakening,
        bearish_weakening,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn standardized_psar_oscillator_into(
    oscillator_out: &mut [f64],
    ma_out: &mut [f64],
    bullish_reversal_out: &mut [f64],
    bearish_reversal_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    bullish_weakening_out: &mut [f64],
    bearish_weakening_out: &mut [f64],
    input: &StandardizedPsarOscillatorInput,
) -> Result<(), StandardizedPsarOscillatorError> {
    standardized_psar_oscillator_into_slice(
        oscillator_out,
        ma_out,
        bullish_reversal_out,
        bearish_reversal_out,
        regular_bullish_out,
        regular_bearish_out,
        bullish_weakening_out,
        bearish_weakening_out,
        input,
        Kernel::Auto,
    )
}

pub fn standardized_psar_oscillator_into_slice(
    oscillator_out: &mut [f64],
    ma_out: &mut [f64],
    bullish_reversal_out: &mut [f64],
    bearish_reversal_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    bullish_weakening_out: &mut [f64],
    bearish_weakening_out: &mut [f64],
    input: &StandardizedPsarOscillatorInput,
    kernel: Kernel,
) -> Result<(), StandardizedPsarOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.all_finite,
        &StandardizedPsarOscillatorParams {
            start: Some(prepared.start),
            increment: Some(prepared.increment),
            maximum: Some(prepared.maximum),
            standardization_length: Some(prepared.standardization_length),
            wma_length: Some(prepared.wma_length),
            wma_lag: Some(prepared.wma_lag),
            pivot_left: Some(prepared.pivot_left),
            pivot_right: Some(prepared.pivot_right),
            plot_bullish: Some(prepared.plot_bullish),
            plot_bearish: Some(prepared.plot_bearish),
        },
        oscillator_out,
        ma_out,
        bullish_reversal_out,
        bearish_reversal_out,
        regular_bullish_out,
        regular_bearish_out,
        bullish_weakening_out,
        bearish_weakening_out,
    )
}

fn standardized_psar_oscillator_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StandardizedPsarOscillatorBatchRange,
    parallel: bool,
    oscillator_out: &mut [f64],
    ma_out: &mut [f64],
    bullish_reversal_out: &mut [f64],
    bearish_reversal_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    bullish_weakening_out: &mut [f64],
    bearish_weakening_out: &mut [f64],
) -> Result<Vec<StandardizedPsarOscillatorParams>, StandardizedPsarOscillatorError> {
    let (_, max_run) = analyze_valid_segments(high, low, close)?;
    let combos = expand_grid_standardized_psar_oscillator(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let all_finite = max_run == cols;
    let expected =
        rows.checked_mul(cols)
            .ok_or(StandardizedPsarOscillatorError::OutputLengthMismatch {
                expected: usize::MAX,
                got: oscillator_out.len(),
            })?;

    for out in [
        &mut *oscillator_out,
        &mut *ma_out,
        &mut *bullish_reversal_out,
        &mut *bearish_reversal_out,
        &mut *regular_bullish_out,
        &mut *regular_bearish_out,
        &mut *bullish_weakening_out,
        &mut *bearish_weakening_out,
    ] {
        if out.len() != expected {
            return Err(StandardizedPsarOscillatorError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
    }

    for params in &combos {
        let standardization_length = params
            .standardization_length
            .unwrap_or(DEFAULT_STANDARDIZATION_LENGTH);
        let wma_length = params.wma_length.unwrap_or(DEFAULT_WMA_LENGTH);
        let needed = required_valid_bars(standardization_length, wma_length);
        if max_run < needed {
            return Err(StandardizedPsarOscillatorError::NotEnoughValidData {
                needed,
                valid: max_run,
            });
        }
        validate_params(
            params.start.unwrap_or(DEFAULT_START),
            params.increment.unwrap_or(DEFAULT_INCREMENT),
            params.maximum.unwrap_or(DEFAULT_MAXIMUM),
            standardization_length,
            wma_length,
            params.wma_lag.unwrap_or(DEFAULT_WMA_LAG),
            params.pivot_left.unwrap_or(DEFAULT_PIVOT_LEFT),
            params.pivot_right.unwrap_or(DEFAULT_PIVOT_RIGHT),
            cols,
        )?;
    }

    let do_row = |row: usize,
                  oscillator_row: &mut [f64],
                  ma_row: &mut [f64],
                  bullish_reversal_row: &mut [f64],
                  bearish_reversal_row: &mut [f64],
                  regular_bullish_row: &mut [f64],
                  regular_bearish_row: &mut [f64],
                  bullish_weakening_row: &mut [f64],
                  bearish_weakening_row: &mut [f64]| {
        compute_row(
            high,
            low,
            close,
            all_finite,
            &combos[row],
            oscillator_row,
            ma_row,
            bullish_reversal_row,
            bearish_reversal_row,
            regular_bullish_row,
            regular_bearish_row,
            bullish_weakening_row,
            bearish_weakening_row,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            oscillator_out
                .par_chunks_mut(cols)
                .zip(ma_out.par_chunks_mut(cols))
                .zip(bullish_reversal_out.par_chunks_mut(cols))
                .zip(bearish_reversal_out.par_chunks_mut(cols))
                .zip(regular_bullish_out.par_chunks_mut(cols))
                .zip(regular_bearish_out.par_chunks_mut(cols))
                .zip(bullish_weakening_out.par_chunks_mut(cols))
                .zip(bearish_weakening_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(
                    |(
                        row,
                        (
                            (
                                (
                                    (
                                        (
                                            ((oscillator_row, ma_row), bullish_reversal_row),
                                            bearish_reversal_row,
                                        ),
                                        regular_bullish_row,
                                    ),
                                    regular_bearish_row,
                                ),
                                bullish_weakening_row,
                            ),
                            bearish_weakening_row,
                        ),
                    )| {
                        do_row(
                            row,
                            oscillator_row,
                            ma_row,
                            bullish_reversal_row,
                            bearish_reversal_row,
                            regular_bullish_row,
                            regular_bearish_row,
                            bullish_weakening_row,
                            bearish_weakening_row,
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
                    &mut oscillator_out[start..end],
                    &mut ma_out[start..end],
                    &mut bullish_reversal_out[start..end],
                    &mut bearish_reversal_out[start..end],
                    &mut regular_bullish_out[start..end],
                    &mut regular_bearish_out[start..end],
                    &mut bullish_weakening_out[start..end],
                    &mut bearish_weakening_out[start..end],
                )?;
            }
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            do_row(
                row,
                &mut oscillator_out[start..end],
                &mut ma_out[start..end],
                &mut bullish_reversal_out[start..end],
                &mut bearish_reversal_out[start..end],
                &mut regular_bullish_out[start..end],
                &mut regular_bearish_out[start..end],
                &mut bullish_weakening_out[start..end],
                &mut bearish_weakening_out[start..end],
            )?;
        }
    }

    Ok(combos)
}

pub fn standardized_psar_oscillator_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StandardizedPsarOscillatorBatchRange,
    kernel: Kernel,
) -> Result<StandardizedPsarOscillatorBatchOutput, StandardizedPsarOscillatorError> {
    match kernel {
        Kernel::Auto => {
            let _ = detect_best_batch_kernel();
        }
        k if !k.is_batch() => {
            return Err(StandardizedPsarOscillatorError::InvalidKernelForBatch(k));
        }
        _ => {}
    }
    standardized_psar_oscillator_batch_par_slice(high, low, close, sweep, Kernel::ScalarBatch)
}

pub fn standardized_psar_oscillator_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StandardizedPsarOscillatorBatchRange,
    _kernel: Kernel,
) -> Result<StandardizedPsarOscillatorBatchOutput, StandardizedPsarOscillatorError> {
    standardized_psar_oscillator_batch_impl(high, low, close, sweep, false)
}

pub fn standardized_psar_oscillator_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StandardizedPsarOscillatorBatchRange,
    _kernel: Kernel,
) -> Result<StandardizedPsarOscillatorBatchOutput, StandardizedPsarOscillatorError> {
    standardized_psar_oscillator_batch_impl(high, low, close, sweep, true)
}

fn standardized_psar_oscillator_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StandardizedPsarOscillatorBatchRange,
    parallel: bool,
) -> Result<StandardizedPsarOscillatorBatchOutput, StandardizedPsarOscillatorError> {
    let rows = expand_grid_standardized_psar_oscillator(sweep)?.len();
    let cols = close.len();

    let oscillator_mu = make_uninit_matrix(rows, cols);
    let ma_mu = make_uninit_matrix(rows, cols);
    let bullish_reversal_mu = make_uninit_matrix(rows, cols);
    let bearish_reversal_mu = make_uninit_matrix(rows, cols);
    let regular_bullish_mu = make_uninit_matrix(rows, cols);
    let regular_bearish_mu = make_uninit_matrix(rows, cols);
    let bullish_weakening_mu = make_uninit_matrix(rows, cols);
    let bearish_weakening_mu = make_uninit_matrix(rows, cols);

    let mut oscillator_guard = ManuallyDrop::new(oscillator_mu);
    let mut ma_guard = ManuallyDrop::new(ma_mu);
    let mut bullish_reversal_guard = ManuallyDrop::new(bullish_reversal_mu);
    let mut bearish_reversal_guard = ManuallyDrop::new(bearish_reversal_mu);
    let mut regular_bullish_guard = ManuallyDrop::new(regular_bullish_mu);
    let mut regular_bearish_guard = ManuallyDrop::new(regular_bearish_mu);
    let mut bullish_weakening_guard = ManuallyDrop::new(bullish_weakening_mu);
    let mut bearish_weakening_guard = ManuallyDrop::new(bearish_weakening_mu);

    let oscillator_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            oscillator_guard.as_mut_ptr() as *mut f64,
            oscillator_guard.len(),
        )
    };
    let ma_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(ma_guard.as_mut_ptr() as *mut f64, ma_guard.len())
    };
    let bullish_reversal_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            bullish_reversal_guard.as_mut_ptr() as *mut f64,
            bullish_reversal_guard.len(),
        )
    };
    let bearish_reversal_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            bearish_reversal_guard.as_mut_ptr() as *mut f64,
            bearish_reversal_guard.len(),
        )
    };
    let regular_bullish_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            regular_bullish_guard.as_mut_ptr() as *mut f64,
            regular_bullish_guard.len(),
        )
    };
    let regular_bearish_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            regular_bearish_guard.as_mut_ptr() as *mut f64,
            regular_bearish_guard.len(),
        )
    };
    let bullish_weakening_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            bullish_weakening_guard.as_mut_ptr() as *mut f64,
            bullish_weakening_guard.len(),
        )
    };
    let bearish_weakening_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            bearish_weakening_guard.as_mut_ptr() as *mut f64,
            bearish_weakening_guard.len(),
        )
    };

    let combos = standardized_psar_oscillator_batch_inner_into(
        high,
        low,
        close,
        sweep,
        parallel,
        oscillator_out,
        ma_out,
        bullish_reversal_out,
        bearish_reversal_out,
        regular_bullish_out,
        regular_bearish_out,
        bullish_weakening_out,
        bearish_weakening_out,
    )?;

    let oscillator = unsafe {
        Vec::from_raw_parts(
            oscillator_guard.as_mut_ptr() as *mut f64,
            oscillator_guard.len(),
            oscillator_guard.capacity(),
        )
    };
    let ma = unsafe {
        Vec::from_raw_parts(
            ma_guard.as_mut_ptr() as *mut f64,
            ma_guard.len(),
            ma_guard.capacity(),
        )
    };
    let bullish_reversal = unsafe {
        Vec::from_raw_parts(
            bullish_reversal_guard.as_mut_ptr() as *mut f64,
            bullish_reversal_guard.len(),
            bullish_reversal_guard.capacity(),
        )
    };
    let bearish_reversal = unsafe {
        Vec::from_raw_parts(
            bearish_reversal_guard.as_mut_ptr() as *mut f64,
            bearish_reversal_guard.len(),
            bearish_reversal_guard.capacity(),
        )
    };
    let regular_bullish = unsafe {
        Vec::from_raw_parts(
            regular_bullish_guard.as_mut_ptr() as *mut f64,
            regular_bullish_guard.len(),
            regular_bullish_guard.capacity(),
        )
    };
    let regular_bearish = unsafe {
        Vec::from_raw_parts(
            regular_bearish_guard.as_mut_ptr() as *mut f64,
            regular_bearish_guard.len(),
            regular_bearish_guard.capacity(),
        )
    };
    let bullish_weakening = unsafe {
        Vec::from_raw_parts(
            bullish_weakening_guard.as_mut_ptr() as *mut f64,
            bullish_weakening_guard.len(),
            bullish_weakening_guard.capacity(),
        )
    };
    let bearish_weakening = unsafe {
        Vec::from_raw_parts(
            bearish_weakening_guard.as_mut_ptr() as *mut f64,
            bearish_weakening_guard.len(),
            bearish_weakening_guard.capacity(),
        )
    };

    Ok(StandardizedPsarOscillatorBatchOutput {
        oscillator,
        ma,
        bullish_reversal,
        bearish_reversal,
        regular_bullish,
        regular_bearish,
        bullish_weakening,
        bearish_weakening,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "standardized_psar_oscillator")]
#[pyo3(signature = (high, low, close, start=DEFAULT_START, increment=DEFAULT_INCREMENT, maximum=DEFAULT_MAXIMUM, standardization_length=DEFAULT_STANDARDIZATION_LENGTH, wma_length=DEFAULT_WMA_LENGTH, wma_lag=DEFAULT_WMA_LAG, pivot_left=DEFAULT_PIVOT_LEFT, pivot_right=DEFAULT_PIVOT_RIGHT, plot_bullish=DEFAULT_PLOT_BULLISH, plot_bearish=DEFAULT_PLOT_BEARISH, kernel=None))]
pub fn standardized_psar_oscillator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    start: f64,
    increment: f64,
    maximum: f64,
    standardization_length: usize,
    wma_length: usize,
    wma_lag: usize,
    pivot_left: usize,
    pivot_right: usize,
    plot_bullish: bool,
    plot_bearish: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = StandardizedPsarOscillatorInput::from_slices(
        high_slice,
        low_slice,
        close_slice,
        StandardizedPsarOscillatorParams {
            start: Some(start),
            increment: Some(increment),
            maximum: Some(maximum),
            standardization_length: Some(standardization_length),
            wma_length: Some(wma_length),
            wma_lag: Some(wma_lag),
            pivot_left: Some(pivot_left),
            pivot_right: Some(pivot_right),
            plot_bullish: Some(plot_bullish),
            plot_bearish: Some(plot_bearish),
        },
    );

    let out = py
        .allow_threads(|| standardized_psar_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("oscillator", out.oscillator.into_pyarray(py))?;
    dict.set_item("ma", out.ma.into_pyarray(py))?;
    dict.set_item("bullish_reversal", out.bullish_reversal.into_pyarray(py))?;
    dict.set_item("bearish_reversal", out.bearish_reversal.into_pyarray(py))?;
    dict.set_item("regular_bullish", out.regular_bullish.into_pyarray(py))?;
    dict.set_item("regular_bearish", out.regular_bearish.into_pyarray(py))?;
    dict.set_item("bullish_weakening", out.bullish_weakening.into_pyarray(py))?;
    dict.set_item("bearish_weakening", out.bearish_weakening.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "standardized_psar_oscillator_batch")]
#[pyo3(signature = (high, low, close, start_range=(DEFAULT_START, DEFAULT_START, 0.0), increment_range=(DEFAULT_INCREMENT, DEFAULT_INCREMENT, 0.0), maximum_range=(DEFAULT_MAXIMUM, DEFAULT_MAXIMUM, 0.0), standardization_length_range=(DEFAULT_STANDARDIZATION_LENGTH, DEFAULT_STANDARDIZATION_LENGTH, 0), wma_length_range=(DEFAULT_WMA_LENGTH, DEFAULT_WMA_LENGTH, 0), wma_lag_range=(DEFAULT_WMA_LAG, DEFAULT_WMA_LAG, 0), pivot_left_range=(DEFAULT_PIVOT_LEFT, DEFAULT_PIVOT_LEFT, 0), pivot_right_range=(DEFAULT_PIVOT_RIGHT, DEFAULT_PIVOT_RIGHT, 0), plot_bullish=DEFAULT_PLOT_BULLISH, plot_bearish=DEFAULT_PLOT_BEARISH, kernel=None))]
pub fn standardized_psar_oscillator_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    start_range: (f64, f64, f64),
    increment_range: (f64, f64, f64),
    maximum_range: (f64, f64, f64),
    standardization_length_range: (usize, usize, usize),
    wma_length_range: (usize, usize, usize),
    wma_lag_range: (usize, usize, usize),
    pivot_left_range: (usize, usize, usize),
    pivot_right_range: (usize, usize, usize),
    plot_bullish: bool,
    plot_bearish: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = StandardizedPsarOscillatorBatchRange {
        start: start_range,
        increment: increment_range,
        maximum: maximum_range,
        standardization_length: standardization_length_range,
        wma_length: wma_length_range,
        wma_lag: wma_lag_range,
        pivot_left: pivot_left_range,
        pivot_right: pivot_right_range,
        plot_bullish,
        plot_bearish,
    };
    let out = py
        .allow_threads(|| {
            standardized_psar_oscillator_batch_with_kernel(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                kernel,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = out.rows;
    let cols = out.cols;
    let dict = PyDict::new(py);
    let oscillator_arr = out.oscillator.into_pyarray(py);
    let ma_arr = out.ma.into_pyarray(py);
    let bullish_reversal_arr = out.bullish_reversal.into_pyarray(py);
    let bearish_reversal_arr = out.bearish_reversal.into_pyarray(py);
    let regular_bullish_arr = out.regular_bullish.into_pyarray(py);
    let regular_bearish_arr = out.regular_bearish.into_pyarray(py);
    let bullish_weakening_arr = out.bullish_weakening.into_pyarray(py);
    let bearish_weakening_arr = out.bearish_weakening.into_pyarray(py);
    let combos = out.combos;
    dict.set_item("oscillator", oscillator_arr.reshape((rows, cols))?)?;
    dict.set_item("ma", ma_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "bullish_reversal",
        bullish_reversal_arr.reshape((rows, cols))?,
    )?;
    dict.set_item(
        "bearish_reversal",
        bearish_reversal_arr.reshape((rows, cols))?,
    )?;
    dict.set_item(
        "regular_bullish",
        regular_bullish_arr.reshape((rows, cols))?,
    )?;
    dict.set_item(
        "regular_bearish",
        regular_bearish_arr.reshape((rows, cols))?,
    )?;
    dict.set_item(
        "bullish_weakening",
        bullish_weakening_arr.reshape((rows, cols))?,
    )?;
    dict.set_item(
        "bearish_weakening",
        bearish_weakening_arr.reshape((rows, cols))?,
    )?;
    dict.set_item(
        "starts",
        combos
            .iter()
            .map(|p| p.start.unwrap_or(DEFAULT_START))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "increments",
        combos
            .iter()
            .map(|p| p.increment.unwrap_or(DEFAULT_INCREMENT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "maximums",
        combos
            .iter()
            .map(|p| p.maximum.unwrap_or(DEFAULT_MAXIMUM))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "standardization_lengths",
        combos
            .iter()
            .map(|p| {
                p.standardization_length
                    .unwrap_or(DEFAULT_STANDARDIZATION_LENGTH) as u64
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "wma_lengths",
        combos
            .iter()
            .map(|p| p.wma_length.unwrap_or(DEFAULT_WMA_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "wma_lags",
        combos
            .iter()
            .map(|p| p.wma_lag.unwrap_or(DEFAULT_WMA_LAG) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "pivot_lefts",
        combos
            .iter()
            .map(|p| p.pivot_left.unwrap_or(DEFAULT_PIVOT_LEFT) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "pivot_rights",
        combos
            .iter()
            .map(|p| p.pivot_right.unwrap_or(DEFAULT_PIVOT_RIGHT) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "plot_bullish",
        combos
            .iter()
            .map(|p| p.plot_bullish.unwrap_or(DEFAULT_PLOT_BULLISH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "plot_bearish",
        combos
            .iter()
            .map(|p| p.plot_bearish.unwrap_or(DEFAULT_PLOT_BEARISH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "StandardizedPsarOscillatorStream")]
pub struct StandardizedPsarOscillatorStreamPy {
    inner: StandardizedPsarOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StandardizedPsarOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (start=DEFAULT_START, increment=DEFAULT_INCREMENT, maximum=DEFAULT_MAXIMUM, standardization_length=DEFAULT_STANDARDIZATION_LENGTH, wma_length=DEFAULT_WMA_LENGTH, wma_lag=DEFAULT_WMA_LAG, pivot_left=DEFAULT_PIVOT_LEFT, pivot_right=DEFAULT_PIVOT_RIGHT, plot_bullish=DEFAULT_PLOT_BULLISH, plot_bearish=DEFAULT_PLOT_BEARISH))]
    pub fn new(
        start: f64,
        increment: f64,
        maximum: f64,
        standardization_length: usize,
        wma_length: usize,
        wma_lag: usize,
        pivot_left: usize,
        pivot_right: usize,
        plot_bullish: bool,
        plot_bearish: bool,
    ) -> PyResult<Self> {
        let inner = StandardizedPsarOscillatorStream::try_new(StandardizedPsarOscillatorParams {
            start: Some(start),
            increment: Some(increment),
            maximum: Some(maximum),
            standardization_length: Some(standardization_length),
            wma_length: Some(wma_length),
            wma_lag: Some(wma_lag),
            pivot_left: Some(pivot_left),
            pivot_right: Some(pivot_right),
            plot_bullish: Some(plot_bullish),
            plot_bearish: Some(plot_bearish),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64)> {
        self.inner.update(high, low, close)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct StandardizedPsarOscillatorBatchConfig {
    pub start_range: (f64, f64, f64),
    pub increment_range: (f64, f64, f64),
    pub maximum_range: (f64, f64, f64),
    pub standardization_length_range: (usize, usize, usize),
    pub wma_length_range: (usize, usize, usize),
    pub wma_lag_range: (usize, usize, usize),
    pub pivot_left_range: (usize, usize, usize),
    pub pivot_right_range: (usize, usize, usize),
    pub plot_bullish: bool,
    pub plot_bearish: bool,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct StandardizedPsarOscillatorBatchJsOutput {
    pub oscillator: Vec<f64>,
    pub ma: Vec<f64>,
    pub bullish_reversal: Vec<f64>,
    pub bearish_reversal: Vec<f64>,
    pub regular_bullish: Vec<f64>,
    pub regular_bearish: Vec<f64>,
    pub bullish_weakening: Vec<f64>,
    pub bearish_weakening: Vec<f64>,
    pub combos: Vec<StandardizedPsarOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn standardized_psar_oscillator_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    start: f64,
    increment: f64,
    maximum: f64,
    standardization_length: usize,
    wma_length: usize,
    wma_lag: usize,
    pivot_left: usize,
    pivot_right: usize,
    plot_bullish: bool,
    plot_bearish: bool,
) -> Result<JsValue, JsValue> {
    let input = StandardizedPsarOscillatorInput::from_slices(
        high,
        low,
        close,
        StandardizedPsarOscillatorParams {
            start: Some(start),
            increment: Some(increment),
            maximum: Some(maximum),
            standardization_length: Some(standardization_length),
            wma_length: Some(wma_length),
            wma_lag: Some(wma_lag),
            pivot_left: Some(pivot_left),
            pivot_right: Some(pivot_right),
            plot_bullish: Some(plot_bullish),
            plot_bearish: Some(plot_bearish),
        },
    );
    let output = standardized_psar_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn standardized_psar_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn standardized_psar_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn standardized_psar_oscillator_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    oscillator_ptr: *mut f64,
    ma_ptr: *mut f64,
    bullish_reversal_ptr: *mut f64,
    bearish_reversal_ptr: *mut f64,
    regular_bullish_ptr: *mut f64,
    regular_bearish_ptr: *mut f64,
    bullish_weakening_ptr: *mut f64,
    bearish_weakening_ptr: *mut f64,
    len: usize,
    start: f64,
    increment: f64,
    maximum: f64,
    standardization_length: usize,
    wma_length: usize,
    wma_lag: usize,
    pivot_left: usize,
    pivot_right: usize,
    plot_bullish: bool,
    plot_bearish: bool,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || oscillator_ptr.is_null()
        || ma_ptr.is_null()
        || bullish_reversal_ptr.is_null()
        || bearish_reversal_ptr.is_null()
        || regular_bullish_ptr.is_null()
        || regular_bearish_ptr.is_null()
        || bullish_weakening_ptr.is_null()
        || bearish_weakening_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = StandardizedPsarOscillatorInput::from_slices(
            high,
            low,
            close,
            StandardizedPsarOscillatorParams {
                start: Some(start),
                increment: Some(increment),
                maximum: Some(maximum),
                standardization_length: Some(standardization_length),
                wma_length: Some(wma_length),
                wma_lag: Some(wma_lag),
                pivot_left: Some(pivot_left),
                pivot_right: Some(pivot_right),
                plot_bullish: Some(plot_bullish),
                plot_bearish: Some(plot_bearish),
            },
        );
        let output = standardized_psar_oscillator_with_kernel(&input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        std::slice::from_raw_parts_mut(oscillator_ptr, len).copy_from_slice(&output.oscillator);
        std::slice::from_raw_parts_mut(ma_ptr, len).copy_from_slice(&output.ma);
        std::slice::from_raw_parts_mut(bullish_reversal_ptr, len)
            .copy_from_slice(&output.bullish_reversal);
        std::slice::from_raw_parts_mut(bearish_reversal_ptr, len)
            .copy_from_slice(&output.bearish_reversal);
        std::slice::from_raw_parts_mut(regular_bullish_ptr, len)
            .copy_from_slice(&output.regular_bullish);
        std::slice::from_raw_parts_mut(regular_bearish_ptr, len)
            .copy_from_slice(&output.regular_bearish);
        std::slice::from_raw_parts_mut(bullish_weakening_ptr, len)
            .copy_from_slice(&output.bullish_weakening);
        std::slice::from_raw_parts_mut(bearish_weakening_ptr, len)
            .copy_from_slice(&output.bearish_weakening);
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = standardized_psar_oscillator_batch)]
pub fn standardized_psar_oscillator_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: StandardizedPsarOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = StandardizedPsarOscillatorBatchRange {
        start: config.start_range,
        increment: config.increment_range,
        maximum: config.maximum_range,
        standardization_length: config.standardization_length_range,
        wma_length: config.wma_length_range,
        wma_lag: config.wma_lag_range,
        pivot_left: config.pivot_left_range,
        pivot_right: config.pivot_right_range,
        plot_bullish: config.plot_bullish,
        plot_bearish: config.plot_bearish,
    };
    let output =
        standardized_psar_oscillator_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&StandardizedPsarOscillatorBatchJsOutput {
        oscillator: output.oscillator,
        ma: output.ma,
        bullish_reversal: output.bullish_reversal,
        bearish_reversal: output.bearish_reversal,
        regular_bullish: output.regular_bullish,
        regular_bearish: output.regular_bearish,
        bullish_weakening: output.bullish_weakening,
        bearish_weakening: output.bearish_weakening,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn standardized_psar_oscillator_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    oscillator_ptr: *mut f64,
    ma_ptr: *mut f64,
    bullish_reversal_ptr: *mut f64,
    bearish_reversal_ptr: *mut f64,
    regular_bullish_ptr: *mut f64,
    regular_bearish_ptr: *mut f64,
    bullish_weakening_ptr: *mut f64,
    bearish_weakening_ptr: *mut f64,
    len: usize,
    start_start: f64,
    start_end: f64,
    start_step: f64,
    increment_start: f64,
    increment_end: f64,
    increment_step: f64,
    maximum_start: f64,
    maximum_end: f64,
    maximum_step: f64,
    standardization_length_start: usize,
    standardization_length_end: usize,
    standardization_length_step: usize,
    wma_length_start: usize,
    wma_length_end: usize,
    wma_length_step: usize,
    wma_lag_start: usize,
    wma_lag_end: usize,
    wma_lag_step: usize,
    pivot_left_start: usize,
    pivot_left_end: usize,
    pivot_left_step: usize,
    pivot_right_start: usize,
    pivot_right_end: usize,
    pivot_right_step: usize,
    plot_bullish: bool,
    plot_bearish: bool,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || oscillator_ptr.is_null()
        || ma_ptr.is_null()
        || bullish_reversal_ptr.is_null()
        || bearish_reversal_ptr.is_null()
        || regular_bullish_ptr.is_null()
        || regular_bearish_ptr.is_null()
        || bullish_weakening_ptr.is_null()
        || bearish_weakening_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = StandardizedPsarOscillatorBatchRange {
        start: (start_start, start_end, start_step),
        increment: (increment_start, increment_end, increment_step),
        maximum: (maximum_start, maximum_end, maximum_step),
        standardization_length: (
            standardization_length_start,
            standardization_length_end,
            standardization_length_step,
        ),
        wma_length: (wma_length_start, wma_length_end, wma_length_step),
        wma_lag: (wma_lag_start, wma_lag_end, wma_lag_step),
        pivot_left: (pivot_left_start, pivot_left_end, pivot_left_step),
        pivot_right: (pivot_right_start, pivot_right_end, pivot_right_step),
        plot_bullish,
        plot_bearish,
    };
    let rows = expand_grid_standardized_psar_oscillator(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;

    unsafe {
        standardized_psar_oscillator_batch_inner_into(
            std::slice::from_raw_parts(high_ptr, len),
            std::slice::from_raw_parts(low_ptr, len),
            std::slice::from_raw_parts(close_ptr, len),
            &sweep,
            false,
            std::slice::from_raw_parts_mut(oscillator_ptr, total),
            std::slice::from_raw_parts_mut(ma_ptr, total),
            std::slice::from_raw_parts_mut(bullish_reversal_ptr, total),
            std::slice::from_raw_parts_mut(bearish_reversal_ptr, total),
            std::slice::from_raw_parts_mut(regular_bullish_ptr, total),
            std::slice::from_raw_parts_mut(regular_bearish_ptr, total),
            std::slice::from_raw_parts_mut(bullish_weakening_ptr, total),
            std::slice::from_raw_parts_mut(bearish_weakening_ptr, total),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mixed_data(size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = vec![0.0; size];
        let mut low = vec![0.0; size];
        let mut close = vec![0.0; size];
        for i in 0..size {
            let x = i as f64;
            let c = 100.0 + 0.18 * x + (x * 0.23).sin() * 5.5 + (x * 0.07).cos() * 2.0;
            close[i] = c;
            high[i] = c + 1.1 + (i % 3) as f64 * 0.05;
            low[i] = c - 1.0 - (i % 2) as f64 * 0.05;
        }
        (high, low, close)
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() || e.is_nan() {
                assert!(a.is_nan() && e.is_nan(), "NaN mismatch at {}", idx);
            } else {
                assert!((a - e).abs() <= tol, "mismatch at {}", idx);
            }
        }
    }

    #[test]
    fn standardized_psar_oscillator_stream_matches_batch() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = mixed_data(220);
        let params = StandardizedPsarOscillatorParams {
            start: Some(0.03),
            increment: Some(0.001),
            maximum: Some(0.25),
            standardization_length: Some(14),
            wma_length: Some(12),
            wma_lag: Some(2),
            pivot_left: Some(8),
            pivot_right: Some(1),
            plot_bullish: Some(true),
            plot_bearish: Some(true),
        };
        let batch = standardized_psar_oscillator(&StandardizedPsarOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            params.clone(),
        ))?;
        let mut stream = StandardizedPsarOscillatorStream::try_new(params)?;
        let mut oscillator = vec![f64::NAN; close.len()];
        let mut ma = vec![f64::NAN; close.len()];
        for i in 0..close.len() {
            if let Some((o, m, _, _, _, _, _, _)) = stream.update(high[i], low[i], close[i]) {
                oscillator[i] = o;
                ma[i] = m;
            }
        }
        assert_close(&oscillator, &batch.oscillator, 1e-12);
        assert_close(&ma, &batch.ma, 1e-12);
        Ok(())
    }

    #[test]
    fn standardized_psar_oscillator_batch_matches_single() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = mixed_data(180);
        let sweep = StandardizedPsarOscillatorBatchRange {
            start: (0.02, 0.03, 0.01),
            increment: (0.0005, 0.0005, 0.0),
            maximum: (0.2, 0.2, 0.0),
            standardization_length: (10, 11, 1),
            wma_length: (8, 8, 0),
            wma_lag: (2, 2, 0),
            pivot_left: (6, 6, 0),
            pivot_right: (1, 1, 0),
            plot_bullish: true,
            plot_bearish: true,
        };
        let batch = standardized_psar_oscillator_batch_with_kernel(
            &high,
            &low,
            &close,
            &sweep,
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, close.len());
        Ok(())
    }
}
