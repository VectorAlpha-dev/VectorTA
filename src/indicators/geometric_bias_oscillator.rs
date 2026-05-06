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
pub fn geometric_bias_oscillator_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    multiplier: f64,
    atr_length: usize,
    smooth: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values =
        geometric_bias_oscillator_js(high, low, close, length, multiplier, atr_length, smooth)?;
    crate::write_wasm_f64_output("geometric_bias_oscillator_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn geometric_bias_oscillator_batch_unified_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = geometric_bias_oscillator_batch_unified_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "geometric_bias_oscillator_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 100;
const DEFAULT_MULTIPLIER: f64 = 2.0;
const DEFAULT_ATR_LENGTH: usize = 14;
const DEFAULT_SMOOTH: usize = 1;
const MIN_LENGTH: usize = 10;
const MAX_LENGTH: usize = 500;
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
fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
    (high - low)
        .max((high - prev_close).abs())
        .max((low - prev_close).abs())
}

#[derive(Debug, Clone)]
pub enum GeometricBiasOscillatorData<'a> {
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
pub struct GeometricBiasOscillatorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct GeometricBiasOscillatorParams {
    pub length: Option<usize>,
    pub multiplier: Option<f64>,
    pub atr_length: Option<usize>,
    pub smooth: Option<usize>,
}

impl Default for GeometricBiasOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            multiplier: Some(DEFAULT_MULTIPLIER),
            atr_length: Some(DEFAULT_ATR_LENGTH),
            smooth: Some(DEFAULT_SMOOTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeometricBiasOscillatorInput<'a> {
    pub data: GeometricBiasOscillatorData<'a>,
    pub params: GeometricBiasOscillatorParams,
}

impl<'a> GeometricBiasOscillatorInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles, params: GeometricBiasOscillatorParams) -> Self {
        Self {
            data: GeometricBiasOscillatorData::Candles { candles },
            params,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: GeometricBiasOscillatorParams,
    ) -> Self {
        Self {
            data: GeometricBiasOscillatorData::Slices { high, low, close },
            params,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, GeometricBiasOscillatorParams::default())
    }

    #[inline(always)]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline(always)]
    pub fn get_multiplier(&self) -> f64 {
        self.params.multiplier.unwrap_or(DEFAULT_MULTIPLIER)
    }

    #[inline(always)]
    pub fn get_atr_length(&self) -> usize {
        self.params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH)
    }

    #[inline(always)]
    pub fn get_smooth(&self) -> usize {
        self.params.smooth.unwrap_or(DEFAULT_SMOOTH)
    }

    #[inline(always)]
    fn as_hlc(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            GeometricBiasOscillatorData::Candles { candles } => (
                high_source(candles),
                low_source(candles),
                close_source(candles),
            ),
            GeometricBiasOscillatorData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

impl<'a> AsRef<[f64]> for GeometricBiasOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        self.as_hlc().2
    }
}

#[derive(Clone, Debug)]
pub struct GeometricBiasOscillatorBuilder {
    length: Option<usize>,
    multiplier: Option<f64>,
    atr_length: Option<usize>,
    smooth: Option<usize>,
    kernel: Kernel,
}

impl Default for GeometricBiasOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            multiplier: None,
            atr_length: None,
            smooth: None,
            kernel: Kernel::Auto,
        }
    }
}

impl GeometricBiasOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline(always)]
    pub fn multiplier(mut self, value: f64) -> Self {
        self.multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn atr_length(mut self, value: usize) -> Self {
        self.atr_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn smooth(mut self, value: usize) -> Self {
        self.smooth = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> GeometricBiasOscillatorParams {
        GeometricBiasOscillatorParams {
            length: self.length,
            multiplier: self.multiplier,
            atr_length: self.atr_length,
            smooth: self.smooth,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<GeometricBiasOscillatorOutput, GeometricBiasOscillatorError> {
        let kernel = self.kernel;
        let params = self.params();
        geometric_bias_oscillator_with_kernel(
            &GeometricBiasOscillatorInput::from_candles(candles, params),
            kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<GeometricBiasOscillatorOutput, GeometricBiasOscillatorError> {
        let kernel = self.kernel;
        let params = self.params();
        geometric_bias_oscillator_with_kernel(
            &GeometricBiasOscillatorInput::from_slices(high, low, close, params),
            kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<GeometricBiasOscillatorStream, GeometricBiasOscillatorError> {
        GeometricBiasOscillatorStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum GeometricBiasOscillatorError {
    #[error("geometric_bias_oscillator: input data slice is empty.")]
    EmptyInputData,
    #[error("geometric_bias_oscillator: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "geometric_bias_oscillator: inconsistent data lengths - high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    DataLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "geometric_bias_oscillator: invalid length: length = {length}, expected {min_length}..={max_length}, data length = {data_len}"
    )]
    InvalidLength {
        length: usize,
        min_length: usize,
        max_length: usize,
        data_len: usize,
    },
    #[error(
        "geometric_bias_oscillator: invalid atr_length: atr_length = {atr_length}, data length = {data_len}"
    )]
    InvalidAtrLength { atr_length: usize, data_len: usize },
    #[error("geometric_bias_oscillator: invalid multiplier: {multiplier}")]
    InvalidMultiplier { multiplier: f64 },
    #[error("geometric_bias_oscillator: invalid smooth: {smooth}")]
    InvalidSmooth { smooth: usize },
    #[error(
        "geometric_bias_oscillator: not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "geometric_bias_oscillator: output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "geometric_bias_oscillator: invalid range for {axis}: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        axis: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("geometric_bias_oscillator: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    length: usize,
    multiplier: f64,
    atr_length: usize,
    smooth: usize,
    warmup: usize,
}

#[inline(always)]
fn normalize_single_kernel(_kernel: Kernel) -> Kernel {
    Kernel::Scalar
}

#[inline(always)]
fn validate_params(
    length: usize,
    multiplier: f64,
    atr_length: usize,
    smooth: usize,
    data_len: usize,
) -> Result<(), GeometricBiasOscillatorError> {
    if length < MIN_LENGTH || length > MAX_LENGTH || length > data_len {
        return Err(GeometricBiasOscillatorError::InvalidLength {
            length,
            min_length: MIN_LENGTH,
            max_length: MAX_LENGTH,
            data_len,
        });
    }
    if atr_length == 0 || atr_length > data_len {
        return Err(GeometricBiasOscillatorError::InvalidAtrLength {
            atr_length,
            data_len,
        });
    }
    if !multiplier.is_finite() || multiplier < MIN_MULTIPLIER {
        return Err(GeometricBiasOscillatorError::InvalidMultiplier { multiplier });
    }
    if smooth == 0 {
        return Err(GeometricBiasOscillatorError::InvalidSmooth { smooth });
    }
    Ok(())
}

#[inline(always)]
fn analyze_valid_segments(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(usize, usize), GeometricBiasOscillatorError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(GeometricBiasOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(GeometricBiasOscillatorError::DataLengthMismatch {
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
        None => Err(GeometricBiasOscillatorError::AllValuesNaN),
    }
}

#[inline(always)]
fn required_valid_bars(length: usize, atr_length: usize, smooth: usize) -> usize {
    length.max(atr_length) + smooth - 1
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a GeometricBiasOscillatorInput<'a>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, GeometricBiasOscillatorError> {
    let _chosen = normalize_single_kernel(kernel);
    let (high, low, close) = input.as_hlc();
    let length = input.get_length();
    let multiplier = input.get_multiplier();
    let atr_length = input.get_atr_length();
    let smooth = input.get_smooth();
    validate_params(length, multiplier, atr_length, smooth, close.len())?;
    let (first_valid, max_run) = analyze_valid_segments(high, low, close)?;
    let needed = required_valid_bars(length, atr_length, smooth);
    if max_run < needed {
        return Err(GeometricBiasOscillatorError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }
    Ok(PreparedInput {
        high,
        low,
        close,
        length,
        multiplier,
        atr_length,
        smooth,
        warmup: first_valid + needed - 1,
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
struct SmaState {
    buffer: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
}

impl SmaState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            buffer: vec![0.0; length.max(1)],
            head: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.count == self.buffer.len() {
            self.sum -= self.buffer[self.head];
        } else {
            self.count += 1;
        }
        self.buffer[self.head] = value;
        self.sum += value;
        self.head += 1;
        if self.head == self.buffer.len() {
            self.head = 0;
        }
        if self.count == self.buffer.len() {
            Some(self.sum / self.buffer.len() as f64)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct PriceWindow {
    buffer: Vec<f64>,
    head: usize,
    count: usize,
}

impl PriceWindow {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            buffer: vec![0.0; length],
            head: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
    }

    #[inline(always)]
    fn push(&mut self, value: f64) {
        self.buffer[self.head] = value;
        self.head += 1;
        if self.head == self.buffer.len() {
            self.head = 0;
        }
        if self.count < self.buffer.len() {
            self.count += 1;
        }
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.count == self.buffer.len()
    }

    #[inline(always)]
    fn ordered_into(&self, out: &mut [f64]) {
        let len = self.buffer.len();
        let start = if self.count == len { self.head } else { 0 };
        for (i, dst) in out.iter_mut().enumerate().take(len) {
            *dst = self.buffer[(start + i) % len];
        }
    }
}

#[inline(always)]
fn point_line_distance(x1: f64, y1: f64, x2: f64, y2: f64, x0: f64, y0: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let denominator = (dx * dx + dy * dy).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        (dy.mul_add(x0, (-dx).mul_add(y0, x2.mul_add(y1, -y2 * x1)))).abs() / denominator
    }
}

fn compute_raw_geometric_bias(
    ordered: &[f64],
    atr: f64,
    threshold: f64,
    keep: &mut [bool],
    stack: &mut Vec<(usize, usize)>,
) -> f64 {
    if !atr.is_finite() || atr <= 0.0 {
        return 0.0;
    }

    let n = ordered.len();
    keep.fill(false);
    keep[0] = true;
    keep[n - 1] = true;

    stack.clear();
    stack.push((0, n - 1));

    let atr_inv = 1.0 / atr;

    while let Some((first_idx, last_idx)) = stack.pop() {
        if last_idx <= first_idx + 1 {
            continue;
        }

        let x1 = first_idx as f64;
        let y1 = ordered[first_idx] * atr_inv;
        let x2 = last_idx as f64;
        let y2 = ordered[last_idx] * atr_inv;

        let mut max_dist = 0.0;
        let mut split_idx = first_idx;

        for i in (first_idx + 1)..last_idx {
            let distance = point_line_distance(x1, y1, x2, y2, i as f64, ordered[i] * atr_inv);
            if distance > max_dist {
                max_dist = distance;
                split_idx = i;
            }
        }

        if max_dist > threshold {
            keep[split_idx] = true;
            stack.push((first_idx, split_idx));
            stack.push((split_idx, last_idx));
        }
    }

    let mut bull_sum = 0.0;
    let mut bear_sum = 0.0;
    let mut last_kept = 0usize;
    for i in 1..n {
        if keep[i] {
            let diff = ordered[i] - ordered[last_kept];
            if diff > 0.0 {
                bull_sum += diff;
            } else if diff < 0.0 {
                bear_sum += -diff;
            }
            last_kept = i;
        }
    }

    let total = bull_sum + bear_sum;
    if total > 0.0 {
        ((bull_sum - bear_sum) / total) * 100.0
    } else {
        0.0
    }
}

#[derive(Clone, Debug)]
struct GeometricBiasOscillatorState {
    atr: AtrState,
    smoother: SmaState,
    prices: PriceWindow,
    prev_close: f64,
    multiplier: f64,
    ordered: Vec<f64>,
    keep: Vec<bool>,
    stack: Vec<(usize, usize)>,
}

impl GeometricBiasOscillatorState {
    #[inline(always)]
    fn new(length: usize, multiplier: f64, atr_length: usize, smooth: usize) -> Self {
        Self {
            atr: AtrState::new(atr_length),
            smoother: SmaState::new(smooth),
            prices: PriceWindow::new(length),
            prev_close: f64::NAN,
            multiplier,
            ordered: vec![0.0; length],
            keep: vec![false; length],
            stack: Vec::with_capacity(length),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.atr.reset();
        self.smoother.reset();
        self.prices.reset();
        self.prev_close = f64::NAN;
        self.stack.clear();
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.reset();
            return None;
        }

        let tr = if self.prev_close.is_finite() {
            true_range(high, low, self.prev_close)
        } else {
            high - low
        };
        self.prev_close = close;
        self.prices.push(close);

        let atr = self.atr.update(tr)?;
        if !self.prices.is_full() {
            return None;
        }

        self.prices.ordered_into(&mut self.ordered);
        let raw = compute_raw_geometric_bias(
            &self.ordered,
            atr,
            self.multiplier,
            &mut self.keep,
            &mut self.stack,
        );
        self.smoother.update(raw)
    }
}

#[derive(Clone, Debug)]
pub struct GeometricBiasOscillatorStream {
    params: GeometricBiasOscillatorParams,
    state: GeometricBiasOscillatorState,
}

impl GeometricBiasOscillatorStream {
    #[inline(always)]
    pub fn try_new(
        params: GeometricBiasOscillatorParams,
    ) -> Result<Self, GeometricBiasOscillatorError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let multiplier = params.multiplier.unwrap_or(DEFAULT_MULTIPLIER);
        let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
        let smooth = params.smooth.unwrap_or(DEFAULT_SMOOTH);
        validate_params(length, multiplier, atr_length, smooth, usize::MAX)?;
        Ok(Self {
            state: GeometricBiasOscillatorState::new(length, multiplier, atr_length, smooth),
            params,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.state.update(high, low, close)
    }

    #[inline(always)]
    pub fn params(&self) -> &GeometricBiasOscillatorParams {
        &self.params
    }
}

#[derive(Clone, Debug)]
pub struct GeometricBiasOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub multiplier: (f64, f64, f64),
    pub atr_length: (usize, usize, usize),
    pub smooth: (usize, usize, usize),
}

impl Default for GeometricBiasOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            multiplier: (DEFAULT_MULTIPLIER, DEFAULT_MULTIPLIER, 0.0),
            atr_length: (DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0),
            smooth: (DEFAULT_SMOOTH, DEFAULT_SMOOTH, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GeometricBiasOscillatorBatchBuilder {
    range: GeometricBiasOscillatorBatchRange,
    kernel: Kernel,
}

impl GeometricBiasOscillatorBatchBuilder {
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
    pub fn multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.multiplier = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn atr_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.atr_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn smooth_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<GeometricBiasOscillatorBatchOutput, GeometricBiasOscillatorError> {
        geometric_bias_oscillator_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<GeometricBiasOscillatorBatchOutput, GeometricBiasOscillatorError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close)
    }
}

#[derive(Clone, Debug)]
pub struct GeometricBiasOscillatorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GeometricBiasOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn axis_usize(
    axis: &'static str,
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, GeometricBiasOscillatorError> {
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
        return Err(GeometricBiasOscillatorError::InvalidRange {
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
) -> Result<Vec<f64>, GeometricBiasOscillatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(GeometricBiasOscillatorError::InvalidRange {
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
        return Err(GeometricBiasOscillatorError::InvalidRange {
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
        return Err(GeometricBiasOscillatorError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_geometric_bias_oscillator(
    sweep: &GeometricBiasOscillatorBatchRange,
) -> Result<Vec<GeometricBiasOscillatorParams>, GeometricBiasOscillatorError> {
    let lengths = axis_usize("length", sweep.length)?;
    let multipliers = axis_float("multiplier", sweep.multiplier)?;
    let atr_lengths = axis_usize("atr_length", sweep.atr_length)?;
    let smooths = axis_usize("smooth", sweep.smooth)?;

    let total = lengths
        .len()
        .checked_mul(multipliers.len())
        .and_then(|value| value.checked_mul(atr_lengths.len()))
        .and_then(|value| value.checked_mul(smooths.len()))
        .ok_or(GeometricBiasOscillatorError::InvalidRange {
            axis: "grid",
            start: lengths.len().to_string(),
            end: multipliers.len().to_string(),
            step: atr_lengths.len().to_string(),
        })?;

    let mut out = Vec::with_capacity(total);
    for &length in &lengths {
        for &multiplier in &multipliers {
            for &atr_length in &atr_lengths {
                for &smooth in &smooths {
                    out.push(GeometricBiasOscillatorParams {
                        length: Some(length),
                        multiplier: Some(multiplier),
                        atr_length: Some(atr_length),
                        smooth: Some(smooth),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
fn compute_row(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    multiplier: f64,
    atr_length: usize,
    smooth: usize,
    out: &mut [f64],
) -> Result<(), GeometricBiasOscillatorError> {
    if out.len() != close.len() {
        return Err(GeometricBiasOscillatorError::OutputLengthMismatch {
            expected: close.len(),
            got: out.len(),
        });
    }

    let mut state = GeometricBiasOscillatorState::new(length, multiplier, atr_length, smooth);
    for i in 0..close.len() {
        out[i] = match state.update(high[i], low[i], close[i]) {
            Some(value) => value,
            None => f64::NAN,
        };
    }
    Ok(())
}

#[inline]
pub fn geometric_bias_oscillator(
    input: &GeometricBiasOscillatorInput,
) -> Result<GeometricBiasOscillatorOutput, GeometricBiasOscillatorError> {
    geometric_bias_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn geometric_bias_oscillator_with_kernel(
    input: &GeometricBiasOscillatorInput,
    kernel: Kernel,
) -> Result<GeometricBiasOscillatorOutput, GeometricBiasOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    let mut values = alloc_with_nan_prefix(prepared.close.len(), prepared.warmup);
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.length,
        prepared.multiplier,
        prepared.atr_length,
        prepared.smooth,
        &mut values,
    )?;
    Ok(GeometricBiasOscillatorOutput { values })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn geometric_bias_oscillator_into(
    out: &mut [f64],
    input: &GeometricBiasOscillatorInput,
) -> Result<(), GeometricBiasOscillatorError> {
    geometric_bias_oscillator_into_slice(out, input, Kernel::Auto)
}

pub fn geometric_bias_oscillator_into_slice(
    out: &mut [f64],
    input: &GeometricBiasOscillatorInput,
    kernel: Kernel,
) -> Result<(), GeometricBiasOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.length,
        prepared.multiplier,
        prepared.atr_length,
        prepared.smooth,
        out,
    )
}

fn geometric_bias_oscillator_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GeometricBiasOscillatorBatchRange,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<GeometricBiasOscillatorParams>, GeometricBiasOscillatorError> {
    let (_, max_run) = analyze_valid_segments(high, low, close)?;
    let combos = expand_grid_geometric_bias_oscillator(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or(GeometricBiasOscillatorError::OutputLengthMismatch {
                expected: usize::MAX,
                got: out.len(),
            })?;
    if out.len() != expected {
        return Err(GeometricBiasOscillatorError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    for params in &combos {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let multiplier = params.multiplier.unwrap_or(DEFAULT_MULTIPLIER);
        let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
        let smooth = params.smooth.unwrap_or(DEFAULT_SMOOTH);
        validate_params(length, multiplier, atr_length, smooth, cols)?;
        let needed = required_valid_bars(length, atr_length, smooth);
        if max_run < needed {
            return Err(GeometricBiasOscillatorError::NotEnoughValidData {
                needed,
                valid: max_run,
            });
        }
    }

    let do_row = |row: usize, row_out: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            high,
            low,
            close,
            params.length.unwrap_or(DEFAULT_LENGTH),
            params.multiplier.unwrap_or(DEFAULT_MULTIPLIER),
            params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH),
            params.smooth.unwrap_or(DEFAULT_SMOOTH),
            row_out,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(row, row_out)| do_row(row, row_out))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, row_out) in out.chunks_mut(cols).enumerate() {
                do_row(row, row_out)?;
            }
        }
    } else {
        for (row, row_out) in out.chunks_mut(cols).enumerate() {
            do_row(row, row_out)?;
        }
    }

    Ok(combos)
}

pub fn geometric_bias_oscillator_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GeometricBiasOscillatorBatchRange,
    kernel: Kernel,
) -> Result<GeometricBiasOscillatorBatchOutput, GeometricBiasOscillatorError> {
    match kernel {
        Kernel::Auto => {
            let _ = detect_best_batch_kernel();
        }
        k if !k.is_batch() => return Err(GeometricBiasOscillatorError::InvalidKernelForBatch(k)),
        _ => {}
    }
    geometric_bias_oscillator_batch_par_slice(high, low, close, sweep, Kernel::ScalarBatch)
}

pub fn geometric_bias_oscillator_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GeometricBiasOscillatorBatchRange,
    _kernel: Kernel,
) -> Result<GeometricBiasOscillatorBatchOutput, GeometricBiasOscillatorError> {
    geometric_bias_oscillator_batch_impl(high, low, close, sweep, false)
}

pub fn geometric_bias_oscillator_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GeometricBiasOscillatorBatchRange,
    _kernel: Kernel,
) -> Result<GeometricBiasOscillatorBatchOutput, GeometricBiasOscillatorError> {
    geometric_bias_oscillator_batch_impl(high, low, close, sweep, true)
}

fn geometric_bias_oscillator_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GeometricBiasOscillatorBatchRange,
    parallel: bool,
) -> Result<GeometricBiasOscillatorBatchOutput, GeometricBiasOscillatorError> {
    let rows = expand_grid_geometric_bias_oscillator(sweep)?.len();
    let cols = close.len();

    let out_mu = make_uninit_matrix(rows, cols);
    let mut out_guard = ManuallyDrop::new(out_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(out_guard.as_mut_ptr() as *mut f64, out_guard.len())
    };

    let combos =
        geometric_bias_oscillator_batch_inner_into(high, low, close, sweep, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            out_guard.as_mut_ptr() as *mut f64,
            out_guard.len(),
            out_guard.capacity(),
        )
    };

    Ok(GeometricBiasOscillatorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "geometric_bias_oscillator")]
#[pyo3(signature = (high, low, close, length=DEFAULT_LENGTH, multiplier=DEFAULT_MULTIPLIER, atr_length=DEFAULT_ATR_LENGTH, smooth=DEFAULT_SMOOTH, kernel=None))]
pub fn geometric_bias_oscillator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    multiplier: f64,
    atr_length: usize,
    smooth: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = GeometricBiasOscillatorInput::from_slices(
        high_slice,
        low_slice,
        close_slice,
        GeometricBiasOscillatorParams {
            length: Some(length),
            multiplier: Some(multiplier),
            atr_length: Some(atr_length),
            smooth: Some(smooth),
        },
    );
    let output = py
        .allow_threads(|| geometric_bias_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "geometric_bias_oscillator_batch")]
#[pyo3(signature = (high, low, close, length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0), multiplier_range=(DEFAULT_MULTIPLIER, DEFAULT_MULTIPLIER, 0.0), atr_length_range=(DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0), smooth_range=(DEFAULT_SMOOTH, DEFAULT_SMOOTH, 0), kernel=None))]
pub fn geometric_bias_oscillator_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    multiplier_range: (f64, f64, f64),
    atr_length_range: (usize, usize, usize),
    smooth_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = GeometricBiasOscillatorBatchRange {
        length: length_range,
        multiplier: multiplier_range,
        atr_length: atr_length_range,
        smooth: smooth_range,
    };

    let combos = expand_grid_geometric_bias_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close_slice.len();
    let total = rows.checked_mul(cols).ok_or_else(|| {
        PyValueError::new_err("rows*cols overflow in geometric_bias_oscillator_batch")
    })?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            geometric_bias_oscillator_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                !matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch),
                out_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "multipliers",
        combos
            .iter()
            .map(|params| params.multiplier.unwrap_or(DEFAULT_MULTIPLIER))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "atr_lengths",
        combos
            .iter()
            .map(|params| params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooths",
        combos
            .iter()
            .map(|params| params.smooth.unwrap_or(DEFAULT_SMOOTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "GeometricBiasOscillatorStream")]
pub struct GeometricBiasOscillatorStreamPy {
    stream: GeometricBiasOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl GeometricBiasOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, multiplier=DEFAULT_MULTIPLIER, atr_length=DEFAULT_ATR_LENGTH, smooth=DEFAULT_SMOOTH))]
    fn new(length: usize, multiplier: f64, atr_length: usize, smooth: usize) -> PyResult<Self> {
        let stream = GeometricBiasOscillatorStream::try_new(GeometricBiasOscillatorParams {
            length: Some(length),
            multiplier: Some(multiplier),
            atr_length: Some(atr_length),
            smooth: Some(smooth),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct GeometricBiasOscillatorBatchConfig {
    pub length_range: (usize, usize, usize),
    pub multiplier_range: (f64, f64, f64),
    pub atr_length_range: (usize, usize, usize),
    pub smooth_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct GeometricBiasOscillatorBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GeometricBiasOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn geometric_bias_oscillator_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    multiplier: f64,
    atr_length: usize,
    smooth: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = GeometricBiasOscillatorInput::from_slices(
        high,
        low,
        close,
        GeometricBiasOscillatorParams {
            length: Some(length),
            multiplier: Some(multiplier),
            atr_length: Some(atr_length),
            smooth: Some(smooth),
        },
    );
    let mut out = vec![0.0; close.len()];
    geometric_bias_oscillator_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn geometric_bias_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn geometric_bias_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn geometric_bias_oscillator_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    multiplier: f64,
    atr_length: usize,
    smooth: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = GeometricBiasOscillatorInput::from_slices(
            high,
            low,
            close,
            GeometricBiasOscillatorParams {
                length: Some(length),
                multiplier: Some(multiplier),
                atr_length: Some(atr_length),
                smooth: Some(smooth),
            },
        );

        if high_ptr as *const u8 == out_ptr as *const u8
            || low_ptr as *const u8 == out_ptr as *const u8
            || close_ptr as *const u8 == out_ptr as *const u8
        {
            let output = geometric_bias_oscillator_with_kernel(&input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&output.values);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            geometric_bias_oscillator_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = geometric_bias_oscillator_batch)]
pub fn geometric_bias_oscillator_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: GeometricBiasOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = GeometricBiasOscillatorBatchRange {
        length: config.length_range,
        multiplier: config.multiplier_range,
        atr_length: config.atr_length_range,
        smooth: config.smooth_range,
    };
    let output =
        geometric_bias_oscillator_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_output = GeometricBiasOscillatorBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn geometric_bias_oscillator_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    multiplier_start: f64,
    multiplier_end: f64,
    multiplier_step: f64,
    atr_length_start: usize,
    atr_length_end: usize,
    atr_length_step: usize,
    smooth_start: usize,
    smooth_end: usize,
    smooth_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = GeometricBiasOscillatorBatchRange {
        length: (length_start, length_end, length_step),
        multiplier: (multiplier_start, multiplier_end, multiplier_step),
        atr_length: (atr_length_start, atr_length_end, atr_length_step),
        smooth: (smooth_start, smooth_end, smooth_step),
    };
    let rows = expand_grid_geometric_bias_oscillator(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        geometric_bias_oscillator_batch_inner_into(high, low, close, &sweep, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn trend_data(size: usize, slope: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(size);
        let mut low = Vec::with_capacity(size);
        let mut close = Vec::with_capacity(size);
        for i in 0..size {
            let c = 100.0 + slope * i as f64;
            close.push(c);
            high.push(c + 1.0);
            low.push(c - 1.0);
        }
        (high, low, close)
    }

    fn mixed_data(size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(size);
        let mut low = Vec::with_capacity(size);
        let mut close = Vec::with_capacity(size);
        for i in 0..size {
            let x = i as f64;
            let c = 100.0 + 0.35 * x + (x * 0.27).sin() * 4.0 + (x * 0.11).cos() * 1.5;
            close.push(c);
            high.push(c + 1.2 + (i % 3) as f64 * 0.1);
            low.push(c - 1.1 - (i % 2) as f64 * 0.1);
        }
        (high, low, close)
    }

    #[test]
    fn geometric_bias_oscillator_increasing_trend_reaches_hundred() -> Result<(), Box<dyn StdError>>
    {
        let (high, low, close) = trend_data(160, 1.0);
        let input = GeometricBiasOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            GeometricBiasOscillatorParams {
                length: Some(20),
                multiplier: Some(2.0),
                atr_length: Some(14),
                smooth: Some(3),
            },
        );
        let out = geometric_bias_oscillator(&input)?;
        let start = required_valid_bars(20, 14, 3) - 1;
        for &value in &out.values[start..] {
            assert!((value - 100.0).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn geometric_bias_oscillator_decreasing_trend_reaches_negative_hundred(
    ) -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = trend_data(160, -1.0);
        let input = GeometricBiasOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            GeometricBiasOscillatorParams {
                length: Some(20),
                multiplier: Some(2.0),
                atr_length: Some(14),
                smooth: Some(3),
            },
        );
        let out = geometric_bias_oscillator(&input)?;
        let start = required_valid_bars(20, 14, 3) - 1;
        for &value in &out.values[start..] {
            assert!((value + 100.0).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn geometric_bias_oscillator_flat_zero_atr_returns_zero() -> Result<(), Box<dyn StdError>> {
        let size = 64usize;
        let high = vec![100.0; size];
        let low = vec![100.0; size];
        let close = vec![100.0; size];
        let input = GeometricBiasOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            GeometricBiasOscillatorParams {
                length: Some(10),
                multiplier: Some(2.0),
                atr_length: Some(5),
                smooth: Some(2),
            },
        );
        let out = geometric_bias_oscillator(&input)?;
        let start = required_valid_bars(10, 5, 2) - 1;
        for &value in &out.values[start..] {
            assert!(value.abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn geometric_bias_oscillator_nan_gap_restart() -> Result<(), Box<dyn StdError>> {
        let (mut high, mut low, mut close) = mixed_data(180);
        high[120] = f64::NAN;
        low[120] = f64::NAN;
        close[120] = f64::NAN;

        let params = GeometricBiasOscillatorParams {
            length: Some(18),
            multiplier: Some(2.0),
            atr_length: Some(10),
            smooth: Some(3),
        };
        let out = geometric_bias_oscillator(&GeometricBiasOscillatorInput::from_slices(
            &high, &low, &close, params,
        ))?;

        let needed = required_valid_bars(18, 10, 3);
        let restart_end = (120 + needed).min(out.values.len());
        assert!(out.values[120..restart_end].iter().all(|v| v.is_nan()));
        Ok(())
    }

    #[test]
    fn geometric_bias_oscillator_stream_matches_batch() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = mixed_data(200);
        let params = GeometricBiasOscillatorParams {
            length: Some(24),
            multiplier: Some(2.4),
            atr_length: Some(12),
            smooth: Some(4),
        };
        let batch = geometric_bias_oscillator(&GeometricBiasOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            params.clone(),
        ))?;
        let mut stream = GeometricBiasOscillatorStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            streamed.push(stream.update(high[i], low[i], close[i]).unwrap_or(f64::NAN));
        }
        for i in 0..close.len() {
            let a = batch.values[i];
            let b = streamed[i];
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn geometric_bias_oscillator_batch_matches_single() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = mixed_data(170);
        let sweep = GeometricBiasOscillatorBatchRange {
            length: (10, 12, 2),
            multiplier: (1.5, 2.0, 0.5),
            atr_length: (5, 6, 1),
            smooth: (1, 2, 1),
        };
        let batch = geometric_bias_oscillator_batch_with_kernel(
            &high,
            &low,
            &close,
            &sweep,
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 16);
        assert_eq!(batch.cols, close.len());

        for (row, params) in batch.combos.iter().enumerate() {
            let single = geometric_bias_oscillator(&GeometricBiasOscillatorInput::from_slices(
                &high,
                &low,
                &close,
                params.clone(),
            ))?;
            let start = row * batch.cols;
            for i in 0..batch.cols {
                let a = batch.values[start + i];
                let b = single.values[i];
                if a.is_nan() || b.is_nan() {
                    assert!(a.is_nan() && b.is_nan());
                } else {
                    assert!((a - b).abs() <= 1e-12);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn geometric_bias_oscillator_invalid_length_errors() {
        let (high, low, close) = mixed_data(64);
        let err = geometric_bias_oscillator(&GeometricBiasOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            GeometricBiasOscillatorParams {
                length: Some(9),
                multiplier: Some(2.0),
                atr_length: Some(14),
                smooth: Some(1),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            GeometricBiasOscillatorError::InvalidLength { .. }
        ));
    }

    #[test]
    fn geometric_bias_oscillator_all_nan_errors() {
        let high = vec![f64::NAN; 32];
        let low = vec![f64::NAN; 32];
        let close = vec![f64::NAN; 32];
        let err = geometric_bias_oscillator(&GeometricBiasOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            GeometricBiasOscillatorParams {
                length: Some(10),
                multiplier: Some(2.0),
                atr_length: Some(14),
                smooth: Some(1),
            },
        ))
        .unwrap_err();
        assert!(matches!(err, GeometricBiasOscillatorError::AllValuesNaN));
    }

    #[test]
    fn geometric_bias_oscillator_default_candles_smoke() -> Result<(), Box<dyn StdError>> {
        let path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(path)?;
        let input = GeometricBiasOscillatorInput::with_default_candles(&candles);
        let out = geometric_bias_oscillator(&input)?;
        assert_eq!(out.values.len(), candles.close.len());
        Ok(())
    }
}
