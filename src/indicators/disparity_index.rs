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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DisparityIndexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DisparityIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DisparityIndexParams {
    pub ema_period: Option<usize>,
    pub lookback_period: Option<usize>,
    pub smoothing_period: Option<usize>,
    pub smoothing_type: Option<String>,
}

impl Default for DisparityIndexParams {
    fn default() -> Self {
        Self {
            ema_period: Some(14),
            lookback_period: Some(14),
            smoothing_period: Some(9),
            smoothing_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DisparityIndexInput<'a> {
    pub data: DisparityIndexData<'a>,
    pub params: DisparityIndexParams,
}

impl<'a> DisparityIndexInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: DisparityIndexParams,
    ) -> Self {
        Self {
            data: DisparityIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: DisparityIndexParams) -> Self {
        Self {
            data: DisparityIndexData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", DisparityIndexParams::default())
    }

    #[inline]
    pub fn get_ema_period(&self) -> usize {
        self.params.ema_period.unwrap_or(14)
    }

    #[inline]
    pub fn get_lookback_period(&self) -> usize {
        self.params.lookback_period.unwrap_or(14)
    }

    #[inline]
    pub fn get_smoothing_period(&self) -> usize {
        self.params.smoothing_period.unwrap_or(9)
    }

    #[inline]
    pub fn get_smoothing_type(&self) -> String {
        self.params
            .smoothing_type
            .clone()
            .unwrap_or_else(|| "ema".to_string())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DisparityIndexBuilder {
    ema_period: Option<usize>,
    lookback_period: Option<usize>,
    smoothing_period: Option<usize>,
    smoothing_type: Option<&'static str>,
    kernel: Kernel,
}

impl Default for DisparityIndexBuilder {
    fn default() -> Self {
        Self {
            ema_period: None,
            lookback_period: None,
            smoothing_period: None,
            smoothing_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DisparityIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn ema_period(mut self, value: usize) -> Self {
        self.ema_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn lookback_period(mut self, value: usize) -> Self {
        self.lookback_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn smoothing_period(mut self, value: usize) -> Self {
        self.smoothing_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn smoothing_type(mut self, value: &'static str) -> Self {
        self.smoothing_type = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<DisparityIndexOutput, DisparityIndexError> {
        let params = DisparityIndexParams {
            ema_period: self.ema_period,
            lookback_period: self.lookback_period,
            smoothing_period: self.smoothing_period,
            smoothing_type: self.smoothing_type.map(str::to_string),
        };
        disparity_index_with_kernel(
            &DisparityIndexInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<DisparityIndexOutput, DisparityIndexError> {
        let params = DisparityIndexParams {
            ema_period: self.ema_period,
            lookback_period: self.lookback_period,
            smoothing_period: self.smoothing_period,
            smoothing_type: self.smoothing_type.map(str::to_string),
        };
        disparity_index_with_kernel(&DisparityIndexInput::from_slice(data, params), self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DisparityIndexStream, DisparityIndexError> {
        DisparityIndexStream::try_new(DisparityIndexParams {
            ema_period: self.ema_period,
            lookback_period: self.lookback_period,
            smoothing_period: self.smoothing_period,
            smoothing_type: self.smoothing_type.map(str::to_string),
        })
    }
}

#[derive(Debug, Error)]
pub enum DisparityIndexError {
    #[error("disparity_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("disparity_index: All values are NaN.")]
    AllValuesNaN,
    #[error("disparity_index: Invalid ema_period: {ema_period}")]
    InvalidEmaPeriod { ema_period: usize },
    #[error("disparity_index: Invalid lookback_period: {lookback_period}")]
    InvalidLookbackPeriod { lookback_period: usize },
    #[error("disparity_index: Invalid smoothing_period: {smoothing_period}")]
    InvalidSmoothingPeriod { smoothing_period: usize },
    #[error("disparity_index: Invalid smoothing_type: {smoothing_type}")]
    InvalidSmoothingType { smoothing_type: String },
    #[error("disparity_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("disparity_index: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("disparity_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("disparity_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("disparity_index: Output length mismatch: dst = {dst_len}, expected = {expected_len}")]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("disparity_index: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmoothingKind {
    Ema,
    Sma,
}

#[derive(Debug, Clone)]
struct ValidatedDisparityIndexParams {
    ema_period: usize,
    lookback_period: usize,
    smoothing_period: usize,
    smoothing_type: String,
    smoothing_kind: SmoothingKind,
}

#[inline(always)]
fn input_slice<'a>(input: &'a DisparityIndexInput<'a>) -> &'a [f64] {
    match &input.data {
        DisparityIndexData::Slice(slice) => slice,
        DisparityIndexData::Candles { candles, source } => source_type(candles, source),
    }
}

#[inline(always)]
fn normalize_smoothing_type(value: &str) -> Option<SmoothingKind> {
    let normalized = value.trim();
    if normalized.eq_ignore_ascii_case("ema") {
        Some(SmoothingKind::Ema)
    } else if normalized.eq_ignore_ascii_case("sma") {
        Some(SmoothingKind::Sma)
    } else {
        None
    }
}

#[inline(always)]
fn validate_params_raw(
    ema_period: usize,
    lookback_period: usize,
    smoothing_period: usize,
    smoothing_type: &str,
) -> Result<ValidatedDisparityIndexParams, DisparityIndexError> {
    if ema_period == 0 {
        return Err(DisparityIndexError::InvalidEmaPeriod { ema_period });
    }
    if lookback_period == 0 {
        return Err(DisparityIndexError::InvalidLookbackPeriod { lookback_period });
    }
    if smoothing_period == 0 {
        return Err(DisparityIndexError::InvalidSmoothingPeriod { smoothing_period });
    }
    let smoothing_kind = normalize_smoothing_type(smoothing_type).ok_or_else(|| {
        DisparityIndexError::InvalidSmoothingType {
            smoothing_type: smoothing_type.to_string(),
        }
    })?;
    Ok(ValidatedDisparityIndexParams {
        ema_period,
        lookback_period,
        smoothing_period,
        smoothing_type: match smoothing_kind {
            SmoothingKind::Ema => "ema".to_string(),
            SmoothingKind::Sma => "sma".to_string(),
        },
        smoothing_kind,
    })
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if value.is_finite() {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn warmup_prefix(validated: &ValidatedDisparityIndexParams) -> usize {
    validated
        .ema_period
        .saturating_add(validated.lookback_period)
        .saturating_add(validated.smoothing_period)
        .saturating_sub(3)
}

#[inline(always)]
fn needed_valid_bars(validated: &ValidatedDisparityIndexParams) -> usize {
    warmup_prefix(validated).saturating_add(1)
}

#[inline(always)]
fn validate_common(
    data: &[f64],
    validated: &ValidatedDisparityIndexParams,
) -> Result<(), DisparityIndexError> {
    if data.is_empty() {
        return Err(DisparityIndexError::EmptyInputData);
    }
    let longest = longest_valid_run(data);
    if longest == 0 {
        return Err(DisparityIndexError::AllValuesNaN);
    }
    let needed = needed_valid_bars(validated);
    if longest < needed {
        return Err(DisparityIndexError::NotEnoughValidData {
            needed,
            valid: longest,
        });
    }
    Ok(())
}

#[inline(always)]
fn disparity_from_price(close: f64, ema: f64) -> Option<f64> {
    if !close.is_finite() || !ema.is_finite() {
        return None;
    }
    if ema.abs() <= f64::EPSILON {
        if close.abs() <= f64::EPSILON {
            Some(0.0)
        } else {
            None
        }
    } else {
        Some((close - ema) / ema * 100.0)
    }
}

#[derive(Debug, Clone)]
pub struct DisparityIndexStream {
    validated: ValidatedDisparityIndexParams,
    ema_alpha: f64,
    ema_beta: f64,
    smoothing_alpha: f64,
    smoothing_beta: f64,
    ema_seed_count: usize,
    ema_seed_sum: f64,
    ema: f64,
    ema_ready: bool,
    disparity_window: Vec<f64>,
    disparity_count: usize,
    disparity_index: usize,
    smoothing_seed_count: usize,
    smoothing_seed_sum: f64,
    smoothed: f64,
    smoothed_ready: bool,
    sma_window: Vec<f64>,
    sma_count: usize,
    sma_index: usize,
    sma_sum: f64,
}

impl DisparityIndexStream {
    #[inline(always)]
    pub fn try_new(params: DisparityIndexParams) -> Result<Self, DisparityIndexError> {
        let validated = validate_params_raw(
            params.ema_period.unwrap_or(14),
            params.lookback_period.unwrap_or(14),
            params.smoothing_period.unwrap_or(9),
            params.smoothing_type.as_deref().unwrap_or("ema"),
        )?;
        let ema_alpha = 2.0 / (validated.ema_period as f64 + 1.0);
        let smoothing_alpha = 2.0 / (validated.smoothing_period as f64 + 1.0);
        Ok(Self {
            ema_alpha,
            ema_beta: 1.0 - ema_alpha,
            smoothing_alpha,
            smoothing_beta: 1.0 - smoothing_alpha,
            disparity_window: vec![f64::NAN; validated.lookback_period],
            sma_window: vec![f64::NAN; validated.smoothing_period],
            validated,
            ema_seed_count: 0,
            ema_seed_sum: 0.0,
            ema: f64::NAN,
            ema_ready: false,
            disparity_count: 0,
            disparity_index: 0,
            smoothing_seed_count: 0,
            smoothing_seed_sum: 0.0,
            smoothed: f64::NAN,
            smoothed_ready: false,
            sma_count: 0,
            sma_index: 0,
            sma_sum: 0.0,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.ema_seed_count = 0;
        self.ema_seed_sum = 0.0;
        self.ema = f64::NAN;
        self.ema_ready = false;
        self.disparity_count = 0;
        self.disparity_index = 0;
        self.smoothing_seed_count = 0;
        self.smoothing_seed_sum = 0.0;
        self.smoothed = f64::NAN;
        self.smoothed_ready = false;
        self.sma_count = 0;
        self.sma_index = 0;
        self.sma_sum = 0.0;
        self.disparity_window.fill(f64::NAN);
        self.sma_window.fill(f64::NAN);
    }

    #[inline(always)]
    fn push_disparity(&mut self, value: f64) {
        self.disparity_window[self.disparity_index] = value;
        self.disparity_index += 1;
        if self.disparity_index == self.validated.lookback_period {
            self.disparity_index = 0;
        }
        if self.disparity_count < self.validated.lookback_period {
            self.disparity_count += 1;
        }
    }

    #[inline(always)]
    fn scaled_from_disparity_window(&self, disparity: f64) -> Option<f64> {
        if self.disparity_count < self.validated.lookback_period {
            return None;
        }
        let mut high = f64::NEG_INFINITY;
        let mut low = f64::INFINITY;
        for &value in &self.disparity_window {
            high = high.max(value);
            low = low.min(value);
        }
        if !(high > low) {
            Some(50.0)
        } else {
            Some((disparity - low) / (high - low) * 100.0)
        }
    }

    #[inline(always)]
    fn smooth_scaled(&mut self, scaled: f64) -> Option<f64> {
        match self.validated.smoothing_kind {
            SmoothingKind::Ema => {
                if !self.smoothed_ready {
                    self.smoothing_seed_sum += scaled;
                    self.smoothing_seed_count += 1;
                    if self.smoothing_seed_count < self.validated.smoothing_period {
                        return None;
                    }
                    self.smoothed =
                        self.smoothing_seed_sum / self.validated.smoothing_period as f64;
                    self.smoothed_ready = true;
                    Some(self.smoothed)
                } else {
                    self.smoothed = self
                        .smoothed
                        .mul_add(self.smoothing_beta, self.smoothing_alpha * scaled);
                    Some(self.smoothed)
                }
            }
            SmoothingKind::Sma => {
                if self.sma_count < self.validated.smoothing_period {
                    self.sma_window[self.sma_count] = scaled;
                    self.sma_sum += scaled;
                    self.sma_count += 1;
                    if self.sma_count < self.validated.smoothing_period {
                        None
                    } else {
                        Some(self.sma_sum / self.validated.smoothing_period as f64)
                    }
                } else {
                    let old = self.sma_window[self.sma_index];
                    self.sma_window[self.sma_index] = scaled;
                    self.sma_index += 1;
                    if self.sma_index == self.validated.smoothing_period {
                        self.sma_index = 0;
                    }
                    self.sma_sum += scaled - old;
                    Some(self.sma_sum / self.validated.smoothing_period as f64)
                }
            }
        }
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if !self.ema_ready {
            self.ema_seed_sum += value;
            self.ema_seed_count += 1;
            if self.ema_seed_count < self.validated.ema_period {
                return None;
            }
            self.ema = self.ema_seed_sum / self.validated.ema_period as f64;
            self.ema_ready = true;
        } else {
            self.ema = self.ema.mul_add(self.ema_beta, self.ema_alpha * value);
        }
        let disparity = disparity_from_price(value, self.ema)?;
        self.push_disparity(disparity);
        let scaled = self.scaled_from_disparity_window(disparity)?;
        self.smooth_scaled(scaled)
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        warmup_prefix(&self.validated)
    }
}

#[inline(always)]
fn compute_row(data: &[f64], validated: &ValidatedDisparityIndexParams, out: &mut [f64]) {
    let mut stream = DisparityIndexStream::try_new(DisparityIndexParams {
        ema_period: Some(validated.ema_period),
        lookback_period: Some(validated.lookback_period),
        smoothing_period: Some(validated.smoothing_period),
        smoothing_type: Some(validated.smoothing_type.clone()),
    })
    .expect("validated disparity index params");
    for (dst, &value) in out.iter_mut().zip(data.iter()) {
        *dst = stream.update(value).unwrap_or(f64::NAN);
    }
}

#[inline]
pub fn disparity_index(
    input: &DisparityIndexInput,
) -> Result<DisparityIndexOutput, DisparityIndexError> {
    disparity_index_with_kernel(input, Kernel::Auto)
}

pub fn disparity_index_with_kernel(
    input: &DisparityIndexInput,
    kernel: Kernel,
) -> Result<DisparityIndexOutput, DisparityIndexError> {
    let data = input_slice(input);
    let validated = validate_params_raw(
        input.get_ema_period(),
        input.get_lookback_period(),
        input.get_smoothing_period(),
        &input.get_smoothing_type(),
    )?;
    validate_common(data, &validated)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = alloc_with_nan_prefix(data.len(), 0);
    out.fill(f64::NAN);
    compute_row(data, &validated, &mut out);
    Ok(DisparityIndexOutput { values: out })
}

pub fn disparity_index_into_slice(
    dst: &mut [f64],
    input: &DisparityIndexInput,
    kernel: Kernel,
) -> Result<(), DisparityIndexError> {
    let data = input_slice(input);
    let validated = validate_params_raw(
        input.get_ema_period(),
        input.get_lookback_period(),
        input.get_smoothing_period(),
        &input.get_smoothing_type(),
    )?;
    validate_common(data, &validated)?;
    if dst.len() != data.len() {
        return Err(DisparityIndexError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    dst.fill(f64::NAN);
    compute_row(data, &validated, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn disparity_index_into(
    input: &DisparityIndexInput,
    out: &mut [f64],
) -> Result<(), DisparityIndexError> {
    disparity_index_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct DisparityIndexBatchRange {
    pub ema_period: (usize, usize, usize),
    pub lookback_period: (usize, usize, usize),
    pub smoothing_period: (usize, usize, usize),
    pub smoothing_types: Vec<String>,
}

impl Default for DisparityIndexBatchRange {
    fn default() -> Self {
        Self {
            ema_period: (14, 14, 0),
            lookback_period: (14, 14, 0),
            smoothing_period: (9, 9, 0),
            smoothing_types: vec!["ema".to_string()],
        }
    }
}

#[derive(Debug, Clone)]
pub struct DisparityIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DisparityIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone)]
pub struct DisparityIndexBatchBuilder {
    range: DisparityIndexBatchRange,
    kernel: Kernel,
}

impl Default for DisparityIndexBatchBuilder {
    fn default() -> Self {
        Self {
            range: DisparityIndexBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl DisparityIndexBatchBuilder {
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
    pub fn ema_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ema_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn lookback_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn smoothing_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn smoothing_types<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.range.smoothing_types = values
            .into_iter()
            .map(|value| value.as_ref().to_string())
            .collect();
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<DisparityIndexBatchOutput, DisparityIndexError> {
        disparity_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<DisparityIndexBatchOutput, DisparityIndexError> {
        disparity_index_batch_with_kernel(candles.close.as_slice(), &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_axis(start: usize, end: usize, step: usize) -> Result<Vec<usize>, DisparityIndexError> {
    if start == 0 || end == 0 {
        return Err(DisparityIndexError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(DisparityIndexError::InvalidRange { start, end, step });
    }

    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(cur);
        if cur >= end {
            break;
        }
        let next = cur.saturating_add(step);
        if next <= cur {
            return Err(DisparityIndexError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_checked(
    range: &DisparityIndexBatchRange,
) -> Result<Vec<DisparityIndexParams>, DisparityIndexError> {
    let ema_periods = expand_axis(range.ema_period.0, range.ema_period.1, range.ema_period.2)?;
    let lookbacks = expand_axis(
        range.lookback_period.0,
        range.lookback_period.1,
        range.lookback_period.2,
    )?;
    let smoothing_periods = expand_axis(
        range.smoothing_period.0,
        range.smoothing_period.1,
        range.smoothing_period.2,
    )?;
    let smoothing_types = if range.smoothing_types.is_empty() {
        vec!["ema".to_string()]
    } else {
        range.smoothing_types.clone()
    };

    let total = ema_periods
        .len()
        .checked_mul(lookbacks.len())
        .and_then(|v| v.checked_mul(smoothing_periods.len()))
        .and_then(|v| v.checked_mul(smoothing_types.len()))
        .ok_or_else(|| DisparityIndexError::InvalidInput {
            msg: "disparity_index: parameter grid size overflow".to_string(),
        })?;
    let mut out = Vec::with_capacity(total);
    for &ema_period in &ema_periods {
        for &lookback_period in &lookbacks {
            for &smoothing_period in &smoothing_periods {
                for smoothing_type in &smoothing_types {
                    validate_params_raw(
                        ema_period,
                        lookback_period,
                        smoothing_period,
                        smoothing_type,
                    )?;
                    out.push(DisparityIndexParams {
                        ema_period: Some(ema_period),
                        lookback_period: Some(lookback_period),
                        smoothing_period: Some(smoothing_period),
                        smoothing_type: Some(smoothing_type.clone()),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_disparity_index(range: &DisparityIndexBatchRange) -> Vec<DisparityIndexParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn disparity_index_batch_with_kernel(
    data: &[f64],
    sweep: &DisparityIndexBatchRange,
    kernel: Kernel,
) -> Result<DisparityIndexBatchOutput, DisparityIndexError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(DisparityIndexError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    if data.is_empty() {
        return Err(DisparityIndexError::EmptyInputData);
    }
    if longest_valid_run(data) == 0 {
        return Err(DisparityIndexError::AllValuesNaN);
    }

    let mut max_needed = 0usize;
    let mut warmups = Vec::with_capacity(combos.len());
    for params in &combos {
        let validated = validate_params_raw(
            params.ema_period.unwrap_or(14),
            params.lookback_period.unwrap_or(14),
            params.smoothing_period.unwrap_or(9),
            params.smoothing_type.as_deref().unwrap_or("ema"),
        )?;
        max_needed = max_needed.max(needed_valid_bars(&validated));
        warmups.push(warmup_prefix(&validated));
    }
    let longest = longest_valid_run(data);
    if longest < max_needed {
        return Err(DisparityIndexError::NotEnoughValidData {
            needed: max_needed,
            valid: longest,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut values_mu, cols, &warmups);
    let mut values = unsafe {
        Vec::from_raw_parts(
            values_mu.as_mut_ptr() as *mut f64,
            values_mu.len(),
            values_mu.capacity(),
        )
    };
    std::mem::forget(values_mu);

    disparity_index_batch_inner_into(data, sweep, kernel, true, &mut values)?;

    Ok(DisparityIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn disparity_index_batch_slice(
    data: &[f64],
    sweep: &DisparityIndexBatchRange,
    kernel: Kernel,
) -> Result<DisparityIndexBatchOutput, DisparityIndexError> {
    disparity_index_batch_inner(data, sweep, kernel, false)
}

pub fn disparity_index_batch_par_slice(
    data: &[f64],
    sweep: &DisparityIndexBatchRange,
    kernel: Kernel,
) -> Result<DisparityIndexBatchOutput, DisparityIndexError> {
    disparity_index_batch_inner(data, sweep, kernel, true)
}

fn disparity_index_batch_inner(
    data: &[f64],
    sweep: &DisparityIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<DisparityIndexBatchOutput, DisparityIndexError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| DisparityIndexError::InvalidInput {
            msg: "disparity_index: rows*cols overflow in batch".to_string(),
        })?;

    let mut warmups = Vec::with_capacity(combos.len());
    for params in &combos {
        let validated = validate_params_raw(
            params.ema_period.unwrap_or(14),
            params.lookback_period.unwrap_or(14),
            params.smoothing_period.unwrap_or(9),
            params.smoothing_type.as_deref().unwrap_or("ema"),
        )?;
        warmups.push(warmup_prefix(&validated));
    }

    let mut values_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut values_mu, cols, &warmups);
    let mut values = unsafe {
        Vec::from_raw_parts(
            values_mu.as_mut_ptr() as *mut f64,
            values_mu.len(),
            values_mu.capacity(),
        )
    };
    std::mem::forget(values_mu);
    debug_assert_eq!(values.len(), total);

    disparity_index_batch_inner_into(data, sweep, kernel, parallel, &mut values)?;

    Ok(DisparityIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn disparity_index_batch_inner_into(
    data: &[f64],
    sweep: &DisparityIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DisparityIndexParams>, DisparityIndexError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(DisparityIndexError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(DisparityIndexError::EmptyInputData);
    }
    let longest = longest_valid_run(data);
    if longest == 0 {
        return Err(DisparityIndexError::AllValuesNaN);
    }

    let total = combos
        .len()
        .checked_mul(len)
        .ok_or_else(|| DisparityIndexError::InvalidInput {
            msg: "disparity_index: rows*cols overflow in batch_into".to_string(),
        })?;
    if out.len() != total {
        return Err(DisparityIndexError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len: total,
        });
    }

    let mut max_needed = 0usize;
    let validated_params: Vec<ValidatedDisparityIndexParams> = combos
        .iter()
        .map(|params| {
            validate_params_raw(
                params.ema_period.unwrap_or(14),
                params.lookback_period.unwrap_or(14),
                params.smoothing_period.unwrap_or(9),
                params.smoothing_type.as_deref().unwrap_or("ema"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for validated in &validated_params {
        max_needed = max_needed.max(needed_valid_bars(validated));
    }
    if longest < max_needed {
        return Err(DisparityIndexError::NotEnoughValidData {
            needed: max_needed,
            valid: longest,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst: &mut [f64]| {
        dst.fill(f64::NAN);
        compute_row(data, &validated_params[row], dst);
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        out.par_chunks_mut(len)
            .enumerate()
            .for_each(|(row, dst)| worker(row, dst));
    } else {
        for (row, dst) in out.chunks_mut(len).enumerate() {
            worker(row, dst);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, dst) in out.chunks_mut(len).enumerate() {
            worker(row, dst);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "disparity_index")]
#[pyo3(signature = (
    data,
    ema_period=14,
    lookback_period=14,
    smoothing_period=9,
    smoothing_type="ema",
    kernel=None
))]
pub fn disparity_index_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    ema_period: usize,
    lookback_period: usize,
    smoothing_period: usize,
    smoothing_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = DisparityIndexInput::from_slice(
        data,
        DisparityIndexParams {
            ema_period: Some(ema_period),
            lookback_period: Some(lookback_period),
            smoothing_period: Some(smoothing_period),
            smoothing_type: Some(smoothing_type.to_string()),
        },
    );
    let out = py
        .allow_threads(|| disparity_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "DisparityIndexStream")]
pub struct DisparityIndexStreamPy {
    stream: DisparityIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DisparityIndexStreamPy {
    #[new]
    #[pyo3(signature = (
        ema_period=14,
        lookback_period=14,
        smoothing_period=9,
        smoothing_type="ema"
    ))]
    fn new(
        ema_period: usize,
        lookback_period: usize,
        smoothing_period: usize,
        smoothing_type: &str,
    ) -> PyResult<Self> {
        let stream = DisparityIndexStream::try_new(DisparityIndexParams {
            ema_period: Some(ema_period),
            lookback_period: Some(lookback_period),
            smoothing_period: Some(smoothing_period),
            smoothing_type: Some(smoothing_type.to_string()),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn reset(&mut self) {
        self.stream.reset();
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "disparity_index_batch")]
#[pyo3(signature = (
    data,
    ema_period_range=(14, 14, 0),
    lookback_period_range=(14, 14, 0),
    smoothing_period_range=(9, 9, 0),
    smoothing_types=None,
    kernel=None
))]
pub fn disparity_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    ema_period_range: (usize, usize, usize),
    lookback_period_range: (usize, usize, usize),
    smoothing_period_range: (usize, usize, usize),
    smoothing_types: Option<Vec<String>>,
    kernel: Option<&str>,
) -> PyResult<PyObject> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = DisparityIndexBatchRange {
        ema_period: ema_period_range,
        lookback_period: lookback_period_range,
        smoothing_period: smoothing_period_range,
        smoothing_types: smoothing_types.unwrap_or_else(|| vec!["ema".to_string()]),
    };
    let out = py
        .allow_threads(|| disparity_index_batch_with_kernel(data, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let values = out
        .values
        .into_pyarray(py)
        .reshape([out.rows, out.cols])?
        .into_pyobject(py)?;
    let ema_periods: Vec<u64> = out
        .combos
        .iter()
        .map(|p| p.ema_period.unwrap_or(14) as u64)
        .collect();
    let lookback_periods: Vec<u64> = out
        .combos
        .iter()
        .map(|p| p.lookback_period.unwrap_or(14) as u64)
        .collect();
    let smoothing_periods: Vec<u64> = out
        .combos
        .iter()
        .map(|p| p.smoothing_period.unwrap_or(9) as u64)
        .collect();
    let smoothing_types: Vec<String> = out
        .combos
        .iter()
        .map(|p| {
            p.smoothing_type
                .clone()
                .unwrap_or_else(|| "ema".to_string())
        })
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("values", values)?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    dict.set_item("ema_periods", ema_periods.into_pyarray(py))?;
    dict.set_item("lookback_periods", lookback_periods.into_pyarray(py))?;
    dict.set_item("smoothing_periods", smoothing_periods.into_pyarray(py))?;
    dict.set_item("smoothing_types", smoothing_types)?;
    Ok(dict.into_any().unbind())
}

#[cfg(feature = "python")]
pub fn register_disparity_index_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(disparity_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(disparity_index_batch_py, m)?)?;
    m.add_class::<DisparityIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisparityIndexBatchConfig {
    pub ema_period_range: Vec<usize>,
    pub lookback_period_range: Vec<usize>,
    pub smoothing_period_range: Vec<usize>,
    #[serde(default)]
    pub smoothing_types: Vec<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = disparity_index_js)]
pub fn disparity_index_js(
    data: &[f64],
    ema_period: usize,
    lookback_period: usize,
    smoothing_period: usize,
    smoothing_type: &str,
) -> Result<JsValue, JsValue> {
    let input = DisparityIndexInput::from_slice(
        data,
        DisparityIndexParams {
            ema_period: Some(ema_period),
            lookback_period: Some(lookback_period),
            smoothing_period: Some(smoothing_period),
            smoothing_type: Some(smoothing_type.to_string()),
        },
    );
    let out = disparity_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out.values).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = disparity_index_batch_js)]
pub fn disparity_index_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: DisparityIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.ema_period_range.len() != 3
        || config.lookback_period_range.len() != 3
        || config.smoothing_period_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: every numeric range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = DisparityIndexBatchRange {
        ema_period: (
            config.ema_period_range[0],
            config.ema_period_range[1],
            config.ema_period_range[2],
        ),
        lookback_period: (
            config.lookback_period_range[0],
            config.lookback_period_range[1],
            config.lookback_period_range[2],
        ),
        smoothing_period: (
            config.smoothing_period_range[0],
            config.smoothing_period_range[1],
            config.smoothing_period_range[2],
        ),
        smoothing_types: if config.smoothing_types.is_empty() {
            vec!["ema".to_string()]
        } else {
            config.smoothing_types
        },
    };
    let out = disparity_index_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("values"),
        &serde_wasm_bindgen::to_value(&out.values).unwrap(),
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
pub fn disparity_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn disparity_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn smoothing_type_from_code(code: usize) -> Result<String, JsValue> {
    match code {
        0 => Ok("ema".to_string()),
        1 => Ok("sma".to_string()),
        _ => Err(JsValue::from_str(
            "invalid smoothing type code: use 0 for ema or 1 for sma",
        )),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn smoothing_types_from_code_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<String>, JsValue> {
    if step == 0 {
        return Ok(vec![smoothing_type_from_code(start)?]);
    }
    if start > end {
        return Err(JsValue::from_str(
            "invalid smoothing type code range: start must be <= end",
        ));
    }
    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(smoothing_type_from_code(cur)?);
        if cur >= end {
            break;
        }
        let next = cur.saturating_add(step);
        if next <= cur {
            return Err(JsValue::from_str(
                "invalid smoothing type code range: step overflow",
            ));
        }
        cur = next.min(end);
    }
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn disparity_index_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    ema_period: usize,
    lookback_period: usize,
    smoothing_period: usize,
    smoothing_type_code: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to disparity_index_into",
        ));
    }
    let smoothing_type = smoothing_type_from_code(smoothing_type_code)?;
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = DisparityIndexInput::from_slice(
            data,
            DisparityIndexParams {
                ema_period: Some(ema_period),
                lookback_period: Some(lookback_period),
                smoothing_period: Some(smoothing_period),
                smoothing_type: Some(smoothing_type),
            },
        );
        disparity_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn disparity_index_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    ema_period_start: usize,
    ema_period_end: usize,
    ema_period_step: usize,
    lookback_period_start: usize,
    lookback_period_end: usize,
    lookback_period_step: usize,
    smoothing_period_start: usize,
    smoothing_period_end: usize,
    smoothing_period_step: usize,
    smoothing_type_start: usize,
    smoothing_type_end: usize,
    smoothing_type_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to disparity_index_batch_into",
        ));
    }
    let sweep = DisparityIndexBatchRange {
        ema_period: (ema_period_start, ema_period_end, ema_period_step),
        lookback_period: (
            lookback_period_start,
            lookback_period_end,
            lookback_period_step,
        ),
        smoothing_period: (
            smoothing_period_start,
            smoothing_period_end,
            smoothing_period_step,
        ),
        smoothing_types: smoothing_types_from_code_range(
            smoothing_type_start,
            smoothing_type_end,
            smoothing_type_step,
        )?,
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in disparity_index_batch_into"))?;

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        disparity_index_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn disparity_index_output_into_js(
    data: &[f64],
    ema_period: usize,
    lookback_period: usize,
    smoothing_period: usize,
    smoothing_type: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = disparity_index_js(
        data,
        ema_period,
        lookback_period,
        smoothing_period,
        smoothing_type,
    )?;
    crate::write_wasm_object_f64_outputs("disparity_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn disparity_index_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = disparity_index_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "disparity_index_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, ParamKV, ParamValue,
    };

    fn sample_close(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                100.0
                    + ((i as f64) * 0.11).sin() * 2.5
                    + ((i as f64) * 0.037).cos() * 0.9
                    + (i as f64) * 0.02
            })
            .collect()
    }

    fn naive_disparity_index(
        data: &[f64],
        ema_period: usize,
        lookback_period: usize,
        smoothing_period: usize,
        smoothing_type: &str,
    ) -> Vec<f64> {
        let validated = validate_params_raw(
            ema_period,
            lookback_period,
            smoothing_period,
            smoothing_type,
        )
        .unwrap();
        let mut out = vec![f64::NAN; data.len()];
        compute_row(data, &validated, &mut out);
        out
    }

    #[test]
    fn disparity_index_matches_naive() -> Result<(), Box<dyn Error>> {
        let close = sample_close(256);
        let input = DisparityIndexInput::from_slice(&close, DisparityIndexParams::default());
        let out = disparity_index(&input)?;
        let expected = naive_disparity_index(&close, 14, 14, 9, "ema");
        for (a, b) in out.values.iter().zip(expected.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn disparity_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let close = sample_close(220);
        let input = DisparityIndexInput::from_slice(
            &close,
            DisparityIndexParams {
                ema_period: Some(10),
                lookback_period: Some(12),
                smoothing_period: Some(5),
                smoothing_type: Some("sma".to_string()),
            },
        );
        let base = disparity_index(&input)?;
        let mut out = vec![0.0; close.len()];
        disparity_index_into_slice(&mut out, &input, Kernel::Auto)?;
        for (a, b) in out.iter().zip(base.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn disparity_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let close = sample_close(240);
        let params = DisparityIndexParams {
            ema_period: Some(14),
            lookback_period: Some(14),
            smoothing_period: Some(9),
            smoothing_type: Some("ema".to_string()),
        };
        let batch = disparity_index(&DisparityIndexInput::from_slice(&close, params.clone()))?;
        let mut stream = DisparityIndexStream::try_new(params)?;
        let mut got = Vec::with_capacity(close.len());
        for &value in &close {
            got.push(stream.update(value).unwrap_or(f64::NAN));
        }
        for (a, b) in got.iter().zip(batch.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn disparity_index_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let close = sample_close(180);
        let single = disparity_index(&DisparityIndexInput::from_slice(
            &close,
            DisparityIndexParams::default(),
        ))?;
        let batch = disparity_index_batch_with_kernel(
            &close,
            &DisparityIndexBatchRange::default(),
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        for (a, b) in batch.values.iter().zip(single.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn disparity_index_rejects_invalid_params() {
        let close = sample_close(64);
        let err = disparity_index(&DisparityIndexInput::from_slice(
            &close,
            DisparityIndexParams {
                ema_period: Some(0),
                ..DisparityIndexParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(err, DisparityIndexError::InvalidEmaPeriod { .. }));

        let err = disparity_index(&DisparityIndexInput::from_slice(
            &close,
            DisparityIndexParams {
                smoothing_type: Some("bad".to_string()),
                ..DisparityIndexParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            DisparityIndexError::InvalidSmoothingType { .. }
        ));
    }

    #[test]
    fn disparity_index_dispatch_compute_returns_value() -> Result<(), Box<dyn Error>> {
        let close = sample_close(160);
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "disparity_index",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &close },
            params: &[
                ParamKV {
                    key: "ema_period",
                    value: ParamValue::Int(14),
                },
                ParamKV {
                    key: "lookback_period",
                    value: ParamValue::Int(14),
                },
                ParamKV {
                    key: "smoothing_period",
                    value: ParamValue::Int(9),
                },
                ParamKV {
                    key: "smoothing_type",
                    value: ParamValue::EnumString("ema"),
                },
            ],
            kernel: Kernel::Auto,
        })?;
        let values = match out.series {
            crate::indicators::dispatch::IndicatorSeries::F64(values) => values,
            _ => panic!("expected F64 output"),
        };
        assert_eq!(values.len(), close.len());
        assert!(values.iter().any(|v| v.is_finite()));
        Ok(())
    }
}
