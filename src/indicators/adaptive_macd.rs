#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_macd_output_into_js(
    data: &[f64],
    length: usize,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_macd_js(data, length, fast_period, slow_period, signal_period)?;
    crate::write_wasm_object_f64_outputs("adaptive_macd_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_macd_batch_unified_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_macd_batch_unified_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "adaptive_macd_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 20;
const DEFAULT_FAST_PERIOD: usize = 10;
const DEFAULT_SLOW_PERIOD: usize = 20;
const DEFAULT_SIGNAL_PERIOD: usize = 9;
const MIN_PERIOD: usize = 2;
const CORR_EPSILON: f64 = 1e-12;

impl<'a> AsRef<[f64]> for AdaptiveMacdInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AdaptiveMacdData::Slice(slice) => slice,
            AdaptiveMacdData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AdaptiveMacdData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AdaptiveMacdOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMacdOutputField {
    Macd,
    Signal,
    Hist,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AdaptiveMacdParams {
    pub length: Option<usize>,
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub signal_period: Option<usize>,
}

impl Default for AdaptiveMacdParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            fast_period: Some(DEFAULT_FAST_PERIOD),
            slow_period: Some(DEFAULT_SLOW_PERIOD),
            signal_period: Some(DEFAULT_SIGNAL_PERIOD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveMacdInput<'a> {
    pub data: AdaptiveMacdData<'a>,
    pub params: AdaptiveMacdParams,
}

impl<'a> AdaptiveMacdInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: AdaptiveMacdParams) -> Self {
        Self {
            data: AdaptiveMacdData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: AdaptiveMacdParams) -> Self {
        Self {
            data: AdaptiveMacdData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", AdaptiveMacdParams::default())
    }

    #[inline(always)]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline(always)]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(DEFAULT_FAST_PERIOD)
    }

    #[inline(always)]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(DEFAULT_SLOW_PERIOD)
    }

    #[inline(always)]
    pub fn get_signal_period(&self) -> usize {
        self.params.signal_period.unwrap_or(DEFAULT_SIGNAL_PERIOD)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AdaptiveMacdBuilder {
    length: Option<usize>,
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    signal_period: Option<usize>,
    kernel: Kernel,
}

impl Default for AdaptiveMacdBuilder {
    fn default() -> Self {
        Self {
            length: None,
            fast_period: None,
            slow_period: None,
            signal_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdaptiveMacdBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline(always)]
    pub fn fast_period(mut self, fast_period: usize) -> Self {
        self.fast_period = Some(fast_period);
        self
    }

    #[inline(always)]
    pub fn slow_period(mut self, slow_period: usize) -> Self {
        self.slow_period = Some(slow_period);
        self
    }

    #[inline(always)]
    pub fn signal_period(mut self, signal_period: usize) -> Self {
        self.signal_period = Some(signal_period);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> AdaptiveMacdParams {
        AdaptiveMacdParams {
            length: self.length,
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            signal_period: self.signal_period,
        }
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<AdaptiveMacdOutput, AdaptiveMacdError> {
        adaptive_macd_with_kernel(
            &AdaptiveMacdInput::from_candles(candles, "close", self.params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<AdaptiveMacdOutput, AdaptiveMacdError> {
        adaptive_macd_with_kernel(
            &AdaptiveMacdInput::from_slice(data, self.params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AdaptiveMacdStream, AdaptiveMacdError> {
        AdaptiveMacdStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum AdaptiveMacdError {
    #[error("adaptive_macd: input data slice is empty.")]
    EmptyInputData,
    #[error("adaptive_macd: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "adaptive_macd: invalid period: length = {length}, fast = {fast}, slow = {slow}, signal = {signal}, data length = {data_len}"
    )]
    InvalidPeriod {
        length: usize,
        fast: usize,
        slow: usize,
        signal: usize,
        data_len: usize,
    },
    #[error("adaptive_macd: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("adaptive_macd: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "adaptive_macd: invalid range for {axis}: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        axis: &'static str,
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("adaptive_macd: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct PreparedInput<'a> {
    data: &'a [f64],
    first_valid: usize,
    length: usize,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    warmup: usize,
    kernel: Kernel,
}

#[derive(Clone, Copy, Debug)]
struct AdaptiveMacdSpec {
    delta_coeff: f64,
    recur_coeff: f64,
    trend_coeff: f64,
    cycle_coeff: f64,
}

#[derive(Clone, Debug)]
pub struct AdaptiveMacdStream {
    params: AdaptiveMacdParams,
    state: AdaptiveMacdState,
}

#[derive(Clone, Debug)]
struct AdaptiveMacdState {
    corr: RollingCorrelationState,
    signal: EmaLikeState,
    prev_close: f64,
    prev_macd1: f64,
    prev_macd2: f64,
    spec: AdaptiveMacdSpec,
}

#[derive(Clone, Debug)]
struct RollingCorrelationState {
    length: usize,
    ring: Vec<f64>,
    head: usize,
    count: usize,
    sum_y: f64,
    sum_y2: f64,
    sum_xy: f64,
    sum_x: f64,
    denom_x: f64,
}

#[derive(Clone, Debug)]
struct EmaLikeState {
    period: usize,
    alpha: f64,
    beta: f64,
    count: usize,
    sum: f64,
    value: f64,
    started: bool,
}

#[derive(Clone, Debug)]
pub struct AdaptiveMacdBatchRange {
    pub length: (usize, usize, usize),
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub signal_period: (usize, usize, usize),
}

impl Default for AdaptiveMacdBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            fast_period: (DEFAULT_FAST_PERIOD, DEFAULT_FAST_PERIOD, 0),
            slow_period: (DEFAULT_SLOW_PERIOD, DEFAULT_SLOW_PERIOD, 0),
            signal_period: (DEFAULT_SIGNAL_PERIOD, DEFAULT_SIGNAL_PERIOD, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdaptiveMacdBatchBuilder {
    range: AdaptiveMacdBatchRange,
    kernel: Kernel,
}

#[derive(Clone, Debug)]
pub struct AdaptiveMacdBatchOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
    pub combos: Vec<AdaptiveMacdParams>,
    pub rows: usize,
    pub cols: usize,
}

impl AdaptiveMacdBatchBuilder {
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
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn signal_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<AdaptiveMacdBatchOutput, AdaptiveMacdError> {
        adaptive_macd_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn with_default_slice(
        data: &[f64],
        kernel: Kernel,
    ) -> Result<AdaptiveMacdBatchOutput, AdaptiveMacdError> {
        AdaptiveMacdBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(data)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<AdaptiveMacdBatchOutput, AdaptiveMacdError> {
        self.apply_slice(source_type(candles, source))
    }

    #[inline(always)]
    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<AdaptiveMacdBatchOutput, AdaptiveMacdError> {
        AdaptiveMacdBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles, "close")
    }
}

#[inline(always)]
fn normalize_single_kernel_to_scalar(_kernel: Kernel) -> Kernel {
    Kernel::Scalar
}

#[inline(always)]
fn validate_periods(
    length: usize,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    data_len: usize,
) -> Result<(), AdaptiveMacdError> {
    if length < MIN_PERIOD
        || fast_period < MIN_PERIOD
        || slow_period < MIN_PERIOD
        || signal_period < MIN_PERIOD
        || length > data_len
        || fast_period > data_len
        || slow_period > data_len
        || signal_period > data_len
    {
        return Err(AdaptiveMacdError::InvalidPeriod {
            length,
            fast: fast_period,
            slow: slow_period,
            signal: signal_period,
            data_len,
        });
    }
    Ok(())
}

#[inline(always)]
fn build_spec(fast_period: usize, slow_period: usize) -> AdaptiveMacdSpec {
    let a1 = 2.0 / (fast_period as f64 + 1.0);
    let a2 = 2.0 / (slow_period as f64 + 1.0);
    AdaptiveMacdSpec {
        delta_coeff: a1 - a2,
        recur_coeff: 2.0 - a1 - a2,
        trend_coeff: (1.0 - a1) * (1.0 - a2),
        cycle_coeff: (1.0 - a1) / (1.0 - a2),
    }
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a AdaptiveMacdInput<'a>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, AdaptiveMacdError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(AdaptiveMacdError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(AdaptiveMacdError::AllValuesNaN)?;

    let length = input.get_length();
    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();
    let signal_period = input.get_signal_period();

    validate_periods(length, fast_period, slow_period, signal_period, data.len())?;

    let valid = data.len() - first_valid;
    if valid < length {
        return Err(AdaptiveMacdError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }

    Ok(PreparedInput {
        data,
        first_valid,
        length,
        fast_period,
        slow_period,
        signal_period,
        warmup: first_valid + length - 1,
        kernel: normalize_single_kernel_to_scalar(kernel),
    })
}

impl RollingCorrelationState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        let sum_x = (length.saturating_sub(1) * length) as f64 * 0.5;
        let sum_x2 = (length.saturating_sub(1) * length * (2 * length - 1)) as f64 / 6.0;
        let n = length as f64;
        Self {
            length,
            ring: vec![0.0; length],
            head: 0,
            count: 0,
            sum_y: 0.0,
            sum_y2: 0.0,
            sum_xy: 0.0,
            sum_x,
            denom_x: n.mul_add(sum_x2, -(sum_x * sum_x)),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum_y = 0.0;
        self.sum_y2 = 0.0;
        self.sum_xy = 0.0;
    }

    #[inline(always)]
    fn corr_sq(&self) -> f64 {
        let n = self.length as f64;
        let denom_y = n.mul_add(self.sum_y2, -(self.sum_y * self.sum_y));
        if denom_y <= CORR_EPSILON {
            return 0.0;
        }
        let num = n.mul_add(self.sum_xy, -(self.sum_x * self.sum_y));
        ((num * num) / (self.denom_x * denom_y)).clamp(0.0, 1.0)
    }

    #[inline(always)]
    fn push(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        if self.count < self.length {
            let idx = self.count;
            self.ring[self.head] = value;
            self.head += 1;
            if self.head == self.length {
                self.head = 0;
            }
            self.count += 1;
            self.sum_y += value;
            self.sum_y2 += value * value;
            self.sum_xy += (idx as f64) * value;
            return if self.count == self.length {
                Some(self.corr_sq())
            } else {
                None
            };
        }

        let old = self.ring[self.head];
        let prev_sum_y = self.sum_y;
        let prev_sum_xy = self.sum_xy;

        self.ring[self.head] = value;
        self.head += 1;
        if self.head == self.length {
            self.head = 0;
        }

        self.sum_y = prev_sum_y - old + value;
        self.sum_y2 = self.sum_y2 - old * old + value * value;
        self.sum_xy = prev_sum_xy - (prev_sum_y - old) + (self.length as f64 - 1.0) * value;

        Some(self.corr_sq())
    }
}

impl EmaLikeState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            alpha,
            beta: 1.0 - alpha,
            count: 0,
            sum: 0.0,
            value: f64::NAN,
            started: false,
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return if self.started { Some(self.value) } else { None };
        }
        if !self.started {
            self.started = true;
            self.count = 1;
            self.sum = value;
            self.value = value;
            return Some(value);
        }
        if self.count < self.period {
            self.count += 1;
            self.sum += value;
            self.value = self.sum / self.count as f64;
            return Some(self.value);
        }
        self.value = self.beta.mul_add(self.value, self.alpha * value);
        Some(self.value)
    }
}

impl AdaptiveMacdState {
    #[inline(always)]
    fn new(params: &AdaptiveMacdParams) -> Result<Self, AdaptiveMacdError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let fast_period = params.fast_period.unwrap_or(DEFAULT_FAST_PERIOD);
        let slow_period = params.slow_period.unwrap_or(DEFAULT_SLOW_PERIOD);
        let signal_period = params.signal_period.unwrap_or(DEFAULT_SIGNAL_PERIOD);
        validate_periods(length, fast_period, slow_period, signal_period, usize::MAX)?;
        Ok(Self {
            corr: RollingCorrelationState::new(length),
            signal: EmaLikeState::new(signal_period),
            prev_close: f64::NAN,
            prev_macd1: f64::NAN,
            prev_macd2: f64::NAN,
            spec: build_spec(fast_period, slow_period),
        })
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> (f64, f64, f64) {
        let current_macd = if value.is_finite() {
            let corr_sq = self.corr.push(value);
            if self.prev_close.is_finite() {
                if let Some(corr_sq) = corr_sq {
                    let r2 = 0.5 * corr_sq + 0.5;
                    let k = r2 * self.spec.trend_coeff + (1.0 - r2) * self.spec.cycle_coeff;
                    let prev1 = if self.prev_macd1.is_finite() {
                        self.prev_macd1
                    } else {
                        0.0
                    };
                    let prev2 = if self.prev_macd2.is_finite() {
                        self.prev_macd2
                    } else {
                        0.0
                    };
                    (value - self.prev_close) * self.spec.delta_coeff
                        + self.spec.recur_coeff * prev1
                        - k * prev2
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            }
        } else {
            self.corr.reset();
            f64::NAN
        };

        self.prev_close = value;
        self.prev_macd2 = self.prev_macd1;
        self.prev_macd1 = current_macd;

        let signal = self.signal.update(current_macd).unwrap_or(f64::NAN);
        let hist = if current_macd.is_finite() && signal.is_finite() {
            current_macd - signal
        } else {
            f64::NAN
        };
        (current_macd, signal, hist)
    }
}

impl AdaptiveMacdStream {
    #[inline(always)]
    pub fn try_new(params: AdaptiveMacdParams) -> Result<Self, AdaptiveMacdError> {
        Ok(Self {
            state: AdaptiveMacdState::new(&params)?,
            params,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        let (macd, signal, hist) = self.state.update(value);
        if macd.is_finite() {
            Some((macd, signal, hist))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn params(&self) -> &AdaptiveMacdParams {
        &self.params
    }
}

#[inline(always)]
fn compute_row(
    data: &[f64],
    params: &AdaptiveMacdParams,
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
) -> Result<(), AdaptiveMacdError> {
    if macd_out.len() != data.len()
        || signal_out.len() != data.len()
        || hist_out.len() != data.len()
    {
        return Err(AdaptiveMacdError::OutputLengthMismatch {
            expected: data.len(),
            got: macd_out.len().max(signal_out.len()).max(hist_out.len()),
        });
    }

    let mut state = AdaptiveMacdState::new(params)?;
    for i in 0..data.len() {
        let (macd, signal, hist) = state.update(data[i]);
        macd_out[i] = macd;
        signal_out[i] = signal;
        hist_out[i] = hist;
    }
    Ok(())
}

#[inline(always)]
fn compute_output_row(
    data: &[f64],
    params: &AdaptiveMacdParams,
    field: AdaptiveMacdOutputField,
    out: &mut [f64],
) -> Result<(), AdaptiveMacdError> {
    if out.len() != data.len() {
        return Err(AdaptiveMacdError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let mut state = AdaptiveMacdState::new(params)?;
    for i in 0..data.len() {
        let (macd, signal, hist) = state.update(data[i]);
        out[i] = match field {
            AdaptiveMacdOutputField::Macd => macd,
            AdaptiveMacdOutputField::Signal => signal,
            AdaptiveMacdOutputField::Hist => hist,
        };
    }
    Ok(())
}

#[inline]
pub fn adaptive_macd(input: &AdaptiveMacdInput) -> Result<AdaptiveMacdOutput, AdaptiveMacdError> {
    adaptive_macd_with_kernel(input, Kernel::Auto)
}

pub fn adaptive_macd_with_kernel(
    input: &AdaptiveMacdInput,
    kernel: Kernel,
) -> Result<AdaptiveMacdOutput, AdaptiveMacdError> {
    let prepared = prepare_input(input, kernel)?;
    let _ = prepared.kernel;
    let mut macd = alloc_with_nan_prefix(prepared.data.len(), prepared.warmup);
    let mut signal = alloc_with_nan_prefix(prepared.data.len(), prepared.warmup);
    let mut hist = alloc_with_nan_prefix(prepared.data.len(), prepared.warmup);
    compute_row(
        prepared.data,
        &AdaptiveMacdParams {
            length: Some(prepared.length),
            fast_period: Some(prepared.fast_period),
            slow_period: Some(prepared.slow_period),
            signal_period: Some(prepared.signal_period),
        },
        &mut macd,
        &mut signal,
        &mut hist,
    )?;
    Ok(AdaptiveMacdOutput { macd, signal, hist })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn adaptive_macd_into(
    input: &AdaptiveMacdInput,
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
) -> Result<(), AdaptiveMacdError> {
    adaptive_macd_into_slice(macd_out, signal_out, hist_out, input, Kernel::Auto)
}

pub fn adaptive_macd_into_slice(
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
    input: &AdaptiveMacdInput,
    kernel: Kernel,
) -> Result<(), AdaptiveMacdError> {
    let prepared = prepare_input(input, kernel)?;
    let _ = prepared.kernel;
    compute_row(
        prepared.data,
        &AdaptiveMacdParams {
            length: Some(prepared.length),
            fast_period: Some(prepared.fast_period),
            slow_period: Some(prepared.slow_period),
            signal_period: Some(prepared.signal_period),
        },
        macd_out,
        signal_out,
        hist_out,
    )
}

pub fn adaptive_macd_output_into_slice(
    out: &mut [f64],
    input: &AdaptiveMacdInput,
    kernel: Kernel,
    field: AdaptiveMacdOutputField,
) -> Result<(), AdaptiveMacdError> {
    let prepared = prepare_input(input, kernel)?;
    let _ = prepared.kernel;
    compute_output_row(
        prepared.data,
        &AdaptiveMacdParams {
            length: Some(prepared.length),
            fast_period: Some(prepared.fast_period),
            slow_period: Some(prepared.slow_period),
            signal_period: Some(prepared.signal_period),
        },
        field,
        out,
    )
}

#[inline(always)]
fn axis_values(
    axis: &'static str,
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, AdaptiveMacdError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    if start < end {
        let mut out = Vec::new();
        let mut current = start;
        loop {
            out.push(current);
            match current.checked_add(step) {
                Some(next) if next <= end => current = next,
                Some(_) | None => break,
            }
        }
        if out.is_empty() {
            return Err(AdaptiveMacdError::InvalidRange {
                axis,
                start,
                end,
                step,
            });
        }
        return Ok(out);
    }

    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current <= end || current < step {
            break;
        }
        current -= step;
        if current < end {
            break;
        }
    }
    if out.is_empty() {
        return Err(AdaptiveMacdError::InvalidRange {
            axis,
            start,
            end,
            step,
        });
    }
    Ok(out)
}

#[inline]
pub fn expand_grid(
    sweep: &AdaptiveMacdBatchRange,
) -> Result<Vec<AdaptiveMacdParams>, AdaptiveMacdError> {
    let lengths = axis_values("length", sweep.length.0, sweep.length.1, sweep.length.2)?;
    let fasts = axis_values(
        "fast_period",
        sweep.fast_period.0,
        sweep.fast_period.1,
        sweep.fast_period.2,
    )?;
    let slows = axis_values(
        "slow_period",
        sweep.slow_period.0,
        sweep.slow_period.1,
        sweep.slow_period.2,
    )?;
    let signals = axis_values(
        "signal_period",
        sweep.signal_period.0,
        sweep.signal_period.1,
        sweep.signal_period.2,
    )?;

    let mut out = Vec::new();
    for &length in &lengths {
        for &fast_period in &fasts {
            for &slow_period in &slows {
                for &signal_period in &signals {
                    out.push(AdaptiveMacdParams {
                        length: Some(length),
                        fast_period: Some(fast_period),
                        slow_period: Some(slow_period),
                        signal_period: Some(signal_period),
                    });
                }
            }
        }
    }
    Ok(out)
}

fn adaptive_macd_batch_inner_into(
    data: &[f64],
    sweep: &AdaptiveMacdBatchRange,
    parallel: bool,
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
) -> Result<Vec<AdaptiveMacdParams>, AdaptiveMacdError> {
    if data.is_empty() {
        return Err(AdaptiveMacdError::EmptyInputData);
    }
    if data.iter().all(|value| value.is_nan()) {
        return Err(AdaptiveMacdError::AllValuesNaN);
    }

    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(AdaptiveMacdError::OutputLengthMismatch {
            expected: usize::MAX,
            got: macd_out.len(),
        })?;
    if macd_out.len() != expected || signal_out.len() != expected || hist_out.len() != expected {
        return Err(AdaptiveMacdError::OutputLengthMismatch {
            expected,
            got: macd_out.len().max(signal_out.len()).max(hist_out.len()),
        });
    }

    for params in &combos {
        validate_periods(
            params.length.unwrap_or(DEFAULT_LENGTH),
            params.fast_period.unwrap_or(DEFAULT_FAST_PERIOD),
            params.slow_period.unwrap_or(DEFAULT_SLOW_PERIOD),
            params.signal_period.unwrap_or(DEFAULT_SIGNAL_PERIOD),
            cols,
        )?;
    }

    let do_row =
        |row: usize, macd_row: &mut [f64], signal_row: &mut [f64], hist_row: &mut [f64]| {
            compute_row(data, &combos[row], macd_row, signal_row, hist_row)
        };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            macd_out
                .par_chunks_mut(cols)
                .zip(signal_out.par_chunks_mut(cols))
                .zip(hist_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, ((macd_row, signal_row), hist_row))| {
                    do_row(row, macd_row, signal_row, hist_row)
                })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, ((macd_row, signal_row), hist_row)) in macd_out
                .chunks_mut(cols)
                .zip(signal_out.chunks_mut(cols))
                .zip(hist_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, macd_row, signal_row, hist_row)?;
            }
        }
    } else {
        for (row, ((macd_row, signal_row), hist_row)) in macd_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .zip(hist_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, macd_row, signal_row, hist_row)?;
        }
    }

    Ok(combos)
}

pub fn adaptive_macd_batch_with_kernel(
    data: &[f64],
    sweep: &AdaptiveMacdBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveMacdBatchOutput, AdaptiveMacdError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AdaptiveMacdError::InvalidKernelForBatch(kernel)),
    };
    let _ = batch_kernel;
    adaptive_macd_batch_par_slice(data, sweep, Kernel::Scalar)
}

pub fn adaptive_macd_batch_slice(
    data: &[f64],
    sweep: &AdaptiveMacdBatchRange,
    _kernel: Kernel,
) -> Result<AdaptiveMacdBatchOutput, AdaptiveMacdError> {
    adaptive_macd_batch_impl(data, sweep, false)
}

pub fn adaptive_macd_batch_par_slice(
    data: &[f64],
    sweep: &AdaptiveMacdBatchRange,
    _kernel: Kernel,
) -> Result<AdaptiveMacdBatchOutput, AdaptiveMacdError> {
    adaptive_macd_batch_impl(data, sweep, true)
}

fn adaptive_macd_batch_impl(
    data: &[f64],
    sweep: &AdaptiveMacdBatchRange,
    parallel: bool,
) -> Result<AdaptiveMacdBatchOutput, AdaptiveMacdError> {
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    let mut macd_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    let mut hist_mu = make_uninit_matrix(rows, cols);

    let mut macd_guard = ManuallyDrop::new(macd_mu);
    let mut signal_guard = ManuallyDrop::new(signal_mu);
    let mut hist_guard = ManuallyDrop::new(hist_mu);

    let macd_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(macd_guard.as_mut_ptr() as *mut f64, macd_guard.len())
    };
    let signal_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };
    let hist_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(hist_guard.as_mut_ptr() as *mut f64, hist_guard.len())
    };

    let combos =
        adaptive_macd_batch_inner_into(data, sweep, parallel, macd_out, signal_out, hist_out)?;

    let macd = unsafe {
        Vec::from_raw_parts(
            macd_guard.as_mut_ptr() as *mut f64,
            macd_guard.len(),
            macd_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };
    let hist = unsafe {
        Vec::from_raw_parts(
            hist_guard.as_mut_ptr() as *mut f64,
            hist_guard.len(),
            hist_guard.capacity(),
        )
    };

    Ok(AdaptiveMacdBatchOutput {
        macd,
        signal,
        hist,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "adaptive_macd")]
#[pyo3(signature = (data, length=DEFAULT_LENGTH, fast_period=DEFAULT_FAST_PERIOD, slow_period=DEFAULT_SLOW_PERIOD, signal_period=DEFAULT_SIGNAL_PERIOD, kernel=None))]
pub fn adaptive_macd_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = AdaptiveMacdInput::from_slice(
        slice_in,
        AdaptiveMacdParams {
            length: Some(length),
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            signal_period: Some(signal_period),
        },
    );
    let result = py
        .allow_threads(|| adaptive_macd_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        result.macd.into_pyarray(py),
        result.signal.into_pyarray(py),
        result.hist.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "adaptive_macd_batch")]
#[pyo3(signature = (data, length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0), fast_period_range=(DEFAULT_FAST_PERIOD, DEFAULT_FAST_PERIOD, 0), slow_period_range=(DEFAULT_SLOW_PERIOD, DEFAULT_SLOW_PERIOD, 0), signal_period_range=(DEFAULT_SIGNAL_PERIOD, DEFAULT_SIGNAL_PERIOD, 0), kernel=None))]
pub fn adaptive_macd_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    fast_period_range: (usize, usize, usize),
    slow_period_range: (usize, usize, usize),
    signal_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let _ = validate_kernel(kernel, true)?;
    let sweep = AdaptiveMacdBatchRange {
        length: length_range,
        fast_period: fast_period_range,
        slow_period: slow_period_range,
        signal_period: signal_period_range,
    };
    let rows = expand_grid(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .len();
    let cols = slice_in.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let macd_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let hist_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let macd_slice = unsafe { macd_arr.as_slice_mut()? };
    let signal_slice = unsafe { signal_arr.as_slice_mut()? };
    let hist_slice = unsafe { hist_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            adaptive_macd_batch_inner_into(
                slice_in,
                &sweep,
                true,
                macd_slice,
                signal_slice,
                hist_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("macd", macd_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item("hist", hist_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "fast_periods",
        combos
            .iter()
            .map(|params| params.fast_period.unwrap_or(DEFAULT_FAST_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_periods",
        combos
            .iter()
            .map(|params| params.slow_period.unwrap_or(DEFAULT_SLOW_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_periods",
        combos
            .iter()
            .map(|params| params.signal_period.unwrap_or(DEFAULT_SIGNAL_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "AdaptiveMacdStream")]
pub struct AdaptiveMacdStreamPy {
    inner: AdaptiveMacdStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdaptiveMacdStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, fast_period=DEFAULT_FAST_PERIOD, slow_period=DEFAULT_SLOW_PERIOD, signal_period=DEFAULT_SIGNAL_PERIOD))]
    pub fn new(
        length: usize,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> PyResult<Self> {
        let inner = AdaptiveMacdStream::try_new(AdaptiveMacdParams {
            length: Some(length),
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            signal_period: Some(signal_period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.inner.update(value)
    }
}

#[cfg(feature = "python")]
pub fn register_adaptive_macd_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adaptive_macd_py, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_macd_batch_py, m)?)?;
    m.add_class::<AdaptiveMacdStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdaptiveMacdBatchConfig {
    pub length_range: (usize, usize, usize),
    pub fast_period_range: (usize, usize, usize),
    pub slow_period_range: (usize, usize, usize),
    pub signal_period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AdaptiveMacdBatchJsOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
    pub combos: Vec<AdaptiveMacdParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_macd_js(
    data: &[f64],
    length: usize,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Result<JsValue, JsValue> {
    let input = AdaptiveMacdInput::from_slice(
        data,
        AdaptiveMacdParams {
            length: Some(length),
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            signal_period: Some(signal_period),
        },
    );
    let output = adaptive_macd_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_macd_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_macd_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_macd_into(
    in_ptr: *const f64,
    macd_ptr: *mut f64,
    signal_ptr: *mut f64,
    hist_ptr: *mut f64,
    len: usize,
    length: usize,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || macd_ptr.is_null() || signal_ptr.is_null() || hist_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = AdaptiveMacdInput::from_slice(
            data,
            AdaptiveMacdParams {
                length: Some(length),
                fast_period: Some(fast_period),
                slow_period: Some(slow_period),
                signal_period: Some(signal_period),
            },
        );

        let aliased = in_ptr == macd_ptr
            || in_ptr == signal_ptr
            || in_ptr == hist_ptr
            || macd_ptr == signal_ptr
            || macd_ptr == hist_ptr
            || signal_ptr == hist_ptr;

        if aliased {
            let out = adaptive_macd_with_kernel(&input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(macd_ptr, len).copy_from_slice(&out.macd);
            std::slice::from_raw_parts_mut(signal_ptr, len).copy_from_slice(&out.signal);
            std::slice::from_raw_parts_mut(hist_ptr, len).copy_from_slice(&out.hist);
        } else {
            let macd_out = std::slice::from_raw_parts_mut(macd_ptr, len);
            let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
            let hist_out = std::slice::from_raw_parts_mut(hist_ptr, len);
            adaptive_macd_into_slice(macd_out, signal_out, hist_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = adaptive_macd_batch)]
pub fn adaptive_macd_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: AdaptiveMacdBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = AdaptiveMacdBatchRange {
        length: config.length_range,
        fast_period: config.fast_period_range,
        slow_period: config.slow_period_range,
        signal_period: config.signal_period_range,
    };
    let output = adaptive_macd_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_output = AdaptiveMacdBatchJsOutput {
        macd: output.macd,
        signal: output.signal,
        hist: output.hist,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_macd_batch_into(
    in_ptr: *const f64,
    macd_ptr: *mut f64,
    signal_ptr: *mut f64,
    hist_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    fast_period_start: usize,
    fast_period_end: usize,
    fast_period_step: usize,
    slow_period_start: usize,
    slow_period_end: usize,
    slow_period_step: usize,
    signal_period_start: usize,
    signal_period_end: usize,
    signal_period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || macd_ptr.is_null() || signal_ptr.is_null() || hist_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = AdaptiveMacdBatchRange {
        length: (length_start, length_end, length_step),
        fast_period: (fast_period_start, fast_period_end, fast_period_step),
        slow_period: (slow_period_start, slow_period_end, slow_period_step),
        signal_period: (signal_period_start, signal_period_end, signal_period_step),
    };
    let rows = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let macd_out = std::slice::from_raw_parts_mut(macd_ptr, total);
        let signal_out = std::slice::from_raw_parts_mut(signal_ptr, total);
        let hist_out = std::slice::from_raw_parts_mut(hist_ptr, total);
        adaptive_macd_batch_inner_into(data, &sweep, false, macd_out, signal_out, hist_out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn linear_data(size: usize) -> Vec<f64> {
        (0..size).map(|i| i as f64).collect()
    }

    fn constant_data(size: usize, value: f64) -> Vec<f64> {
        vec![value; size]
    }

    fn linear_reference(
        size: usize,
        length: usize,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> AdaptiveMacdOutput {
        let spec = build_spec(fast_period, slow_period);
        let mut macd = vec![f64::NAN; size];
        let mut signal = vec![f64::NAN; size];
        let mut hist = vec![f64::NAN; size];
        let mut signal_state = EmaLikeState::new(signal_period);
        let k = spec.trend_coeff;
        for i in (length - 1)..size {
            let prev1 = if i >= 1 && macd[i - 1].is_finite() {
                macd[i - 1]
            } else {
                0.0
            };
            let prev2 = if i >= 2 && macd[i - 2].is_finite() {
                macd[i - 2]
            } else {
                0.0
            };
            macd[i] = spec.delta_coeff + spec.recur_coeff * prev1 - k * prev2;
            signal[i] = signal_state.update(macd[i]).unwrap_or(f64::NAN);
            hist[i] = macd[i] - signal[i];
        }
        AdaptiveMacdOutput { macd, signal, hist }
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() || e.is_nan() {
                assert!(
                    a.is_nan() && e.is_nan(),
                    "NaN mismatch at idx {}: actual={} expected={}",
                    idx,
                    a,
                    e
                );
            } else {
                assert!(
                    (a - e).abs() <= tol,
                    "value mismatch at idx {}: actual={} expected={} tol={}",
                    idx,
                    a,
                    e,
                    tol
                );
            }
        }
    }

    #[test]
    fn adaptive_macd_linear_trend_matches_reference() -> Result<(), Box<dyn StdError>> {
        let data = linear_data(32);
        let params = AdaptiveMacdParams {
            length: Some(5),
            fast_period: Some(4),
            slow_period: Some(9),
            signal_period: Some(3),
        };
        let input = AdaptiveMacdInput::from_slice(&data, params.clone());
        let output = adaptive_macd(&input)?;
        let expected = linear_reference(32, 5, 4, 9, 3);
        assert_close(&output.macd, &expected.macd, 1e-12);
        assert_close(&output.signal, &expected.signal, 1e-12);
        assert_close(&output.hist, &expected.hist, 1e-12);
        Ok(())
    }

    #[test]
    fn adaptive_macd_constant_series_flattens_to_zero() -> Result<(), Box<dyn StdError>> {
        let data = constant_data(24, 100.0);
        let input = AdaptiveMacdInput::from_slice(
            &data,
            AdaptiveMacdParams {
                length: Some(6),
                fast_period: Some(5),
                slow_period: Some(10),
                signal_period: Some(4),
            },
        );
        let output = adaptive_macd(&input)?;
        for i in 0..5 {
            assert!(output.macd[i].is_nan());
            assert!(output.signal[i].is_nan());
            assert!(output.hist[i].is_nan());
        }
        for i in 5..data.len() {
            assert!(output.macd[i].abs() <= 1e-12);
            assert!(output.signal[i].abs() <= 1e-12);
            assert!(output.hist[i].abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn adaptive_macd_nan_gap_restarts_macd() -> Result<(), Box<dyn StdError>> {
        let data = vec![
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            f64::NAN,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
        ];
        let input = AdaptiveMacdInput::from_slice(
            &data,
            AdaptiveMacdParams {
                length: Some(4),
                fast_period: Some(3),
                slow_period: Some(6),
                signal_period: Some(3),
            },
        );
        let output = adaptive_macd(&input)?;
        assert!(output.macd[..3].iter().all(|v| v.is_nan()));
        assert!(output.macd[3].is_finite());
        assert!(output.macd[6].is_nan());
        assert!(output.macd[7].is_nan());
        assert!(output.macd[8].is_nan());
        assert!(output.macd[9].is_nan());
        assert!(output.macd[10].is_finite());
        assert!(output.hist[6].is_nan());
        Ok(())
    }

    #[test]
    fn adaptive_macd_into_matches_single() -> Result<(), Box<dyn StdError>> {
        let data = linear_data(28);
        let input = AdaptiveMacdInput::from_slice(
            &data,
            AdaptiveMacdParams {
                length: Some(5),
                fast_period: Some(4),
                slow_period: Some(8),
                signal_period: Some(3),
            },
        );
        let output = adaptive_macd(&input)?;
        let mut macd = vec![0.0; data.len()];
        let mut signal = vec![0.0; data.len()];
        let mut hist = vec![0.0; data.len()];
        adaptive_macd_into_slice(&mut macd, &mut signal, &mut hist, &input, Kernel::Auto)?;
        assert_close(&macd, &output.macd, 1e-12);
        assert_close(&signal, &output.signal, 1e-12);
        assert_close(&hist, &output.hist, 1e-12);
        Ok(())
    }

    #[test]
    fn adaptive_macd_stream_matches_batch() -> Result<(), Box<dyn StdError>> {
        let data = linear_data(28);
        let params = AdaptiveMacdParams {
            length: Some(5),
            fast_period: Some(4),
            slow_period: Some(8),
            signal_period: Some(3),
        };
        let input = AdaptiveMacdInput::from_slice(&data, params.clone());
        let batch = adaptive_macd(&input)?;
        let mut stream = AdaptiveMacdStream::try_new(params)?;
        let mut macd = Vec::with_capacity(data.len());
        let mut signal = Vec::with_capacity(data.len());
        let mut hist = Vec::with_capacity(data.len());
        for value in data {
            match stream.update(value) {
                Some((m, s, h)) => {
                    macd.push(m);
                    signal.push(s);
                    hist.push(h);
                }
                None => {
                    macd.push(f64::NAN);
                    signal.push(f64::NAN);
                    hist.push(f64::NAN);
                }
            }
        }
        assert_close(&macd, &batch.macd, 1e-12);
        assert_close(&signal, &batch.signal, 1e-12);
        assert_close(&hist, &batch.hist, 1e-12);
        Ok(())
    }

    #[test]
    fn adaptive_macd_batch_matches_single() -> Result<(), Box<dyn StdError>> {
        let data = linear_data(26);
        let sweep = AdaptiveMacdBatchRange {
            length: (4, 5, 1),
            fast_period: (3, 4, 1),
            slow_period: (6, 7, 1),
            signal_period: (3, 3, 0),
        };
        let batch = adaptive_macd_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
        assert_eq!(batch.rows, 8);
        assert_eq!(batch.cols, data.len());
        for (row, params) in batch.combos.iter().enumerate() {
            let input = AdaptiveMacdInput::from_slice(&data, params.clone());
            let single = adaptive_macd(&input)?;
            let start = row * batch.cols;
            let end = start + batch.cols;
            assert_close(&batch.macd[start..end], &single.macd, 1e-12);
            assert_close(&batch.signal[start..end], &single.signal, 1e-12);
            assert_close(&batch.hist[start..end], &single.hist, 1e-12);
        }
        Ok(())
    }

    #[test]
    fn adaptive_macd_invalid_period_errors() {
        let data = linear_data(10);
        let input = AdaptiveMacdInput::from_slice(
            &data,
            AdaptiveMacdParams {
                length: Some(1),
                fast_period: Some(3),
                slow_period: Some(6),
                signal_period: Some(3),
            },
        );
        assert!(matches!(
            adaptive_macd(&input),
            Err(AdaptiveMacdError::InvalidPeriod { .. })
        ));
    }

    #[test]
    fn adaptive_macd_all_nan_errors() {
        let data = vec![f64::NAN; 12];
        let input = AdaptiveMacdInput::from_slice(&data, AdaptiveMacdParams::default());
        assert!(matches!(
            adaptive_macd(&input),
            Err(AdaptiveMacdError::AllValuesNaN)
        ));
    }

    #[test]
    fn adaptive_macd_default_candles_smoke() -> Result<(), Box<dyn StdError>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        let input = AdaptiveMacdInput::with_default_candles(&candles);
        let output = adaptive_macd(&input)?;
        assert_eq!(output.macd.len(), candles.close.len());
        assert_eq!(output.signal.len(), candles.close.len());
        assert_eq!(output.hist.len(), candles.close.len());
        Ok(())
    }
}
