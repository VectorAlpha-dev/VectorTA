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
use std::collections::VecDeque;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ProjectionOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct ProjectionOscillatorOutput {
    pub pbo: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct ProjectionOscillatorParams {
    pub length: Option<usize>,
    pub smooth_length: Option<usize>,
}

impl Default for ProjectionOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(14),
            smooth_length: Some(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectionOscillatorInput<'a> {
    pub data: ProjectionOscillatorData<'a>,
    pub params: ProjectionOscillatorParams,
}

impl<'a> ProjectionOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: ProjectionOscillatorParams,
    ) -> Self {
        Self {
            data: ProjectionOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
        params: ProjectionOscillatorParams,
    ) -> Self {
        Self {
            data: ProjectionOscillatorData::Slices { high, low, source },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", ProjectionOscillatorParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }

    #[inline]
    pub fn get_smooth_length(&self) -> usize {
        self.params.smooth_length.unwrap_or(4)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ProjectionOscillatorBuilder {
    length: Option<usize>,
    smooth_length: Option<usize>,
    kernel: Kernel,
}

impl Default for ProjectionOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            smooth_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ProjectionOscillatorBuilder {
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
    pub fn smooth_length(mut self, value: usize) -> Self {
        self.smooth_length = Some(value);
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
    ) -> Result<ProjectionOscillatorOutput, ProjectionOscillatorError> {
        let params = ProjectionOscillatorParams {
            length: self.length,
            smooth_length: self.smooth_length,
        };
        projection_oscillator_with_kernel(
            &ProjectionOscillatorInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        source: &[f64],
    ) -> Result<ProjectionOscillatorOutput, ProjectionOscillatorError> {
        let params = ProjectionOscillatorParams {
            length: self.length,
            smooth_length: self.smooth_length,
        };
        projection_oscillator_with_kernel(
            &ProjectionOscillatorInput::from_slices(high, low, source, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<ProjectionOscillatorStream, ProjectionOscillatorError> {
        ProjectionOscillatorStream::try_new(ProjectionOscillatorParams {
            length: self.length,
            smooth_length: self.smooth_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum ProjectionOscillatorError {
    #[error("projection_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "projection_oscillator: Input length mismatch: high = {high_len}, low = {low_len}, source = {source_len}"
    )]
    InputLengthMismatch {
        high_len: usize,
        low_len: usize,
        source_len: usize,
    },
    #[error("projection_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("projection_oscillator: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("projection_oscillator: Invalid smooth_length: {smooth_length}")]
    InvalidSmoothLength { smooth_length: usize },
    #[error("projection_oscillator: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("projection_oscillator: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("projection_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("projection_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "projection_oscillator: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("projection_oscillator: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
struct WmaState {
    period: usize,
    denom: f64,
    window: VecDeque<f64>,
    sum: f64,
    weighted_sum: f64,
}

impl WmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            denom: (period * (period + 1) / 2) as f64,
            window: VecDeque::with_capacity(period.max(1)),
            sum: 0.0,
            weighted_sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.weighted_sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if self.period == 1 {
            return Some(value);
        }
        if self.window.len() < self.period {
            let weight = self.window.len() + 1;
            self.window.push_back(value);
            self.sum += value;
            self.weighted_sum += value * weight as f64;
            if self.window.len() == self.period {
                Some(self.weighted_sum / self.denom)
            } else {
                None
            }
        } else {
            let oldest = self.window.pop_front().unwrap_or(0.0);
            let old_sum = self.sum;
            self.window.push_back(value);
            self.sum = old_sum - oldest + value;
            self.weighted_sum = self.weighted_sum - old_sum + self.period as f64 * value;
            Some(self.weighted_sum / self.denom)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectionOscillatorStream {
    length: usize,
    smooth_length: usize,
    high_window: VecDeque<f64>,
    low_window: VecDeque<f64>,
    high_slopes: VecDeque<f64>,
    low_slopes: VecDeque<f64>,
    pbo_wma: WmaState,
    signal_wma: WmaState,
}

impl ProjectionOscillatorStream {
    #[inline(always)]
    pub fn try_new(params: ProjectionOscillatorParams) -> Result<Self, ProjectionOscillatorError> {
        let length = params.length.unwrap_or(14);
        let smooth_length = params.smooth_length.unwrap_or(4);
        validate_params(length, smooth_length)?;
        Ok(Self {
            length,
            smooth_length,
            high_window: VecDeque::with_capacity(length.max(1)),
            low_window: VecDeque::with_capacity(length.max(1)),
            high_slopes: VecDeque::with_capacity(length.max(1)),
            low_slopes: VecDeque::with_capacity(length.max(1)),
            pbo_wma: WmaState::new(smooth_length),
            signal_wma: WmaState::new(smooth_length),
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.high_window.clear();
        self.low_window.clear();
        self.high_slopes.clear();
        self.low_slopes.clear();
        self.pbo_wma.reset();
        self.signal_wma.reset();
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, source: f64) -> Option<(f64, f64)> {
        if !is_valid_triple(high, low, source) {
            self.reset();
            return None;
        }

        push_ring(&mut self.high_window, high, self.length);
        push_ring(&mut self.low_window, low, self.length);

        let high_slope = if self.high_window.len() == self.length {
            linreg_slope_from_window(&self.high_window)
        } else {
            f64::NAN
        };
        let low_slope = if self.low_window.len() == self.length {
            linreg_slope_from_window(&self.low_window)
        } else {
            f64::NAN
        };
        push_ring(&mut self.high_slopes, high_slope, self.length);
        push_ring(&mut self.low_slopes, low_slope, self.length);

        if self.high_window.len() != self.length
            || self.low_window.len() != self.length
            || self.high_slopes.len() != self.length
            || self.low_slopes.len() != self.length
            || self.high_slopes.iter().any(|v| !v.is_finite())
            || self.low_slopes.iter().any(|v| !v.is_finite())
        {
            return None;
        }

        let mut upper = f64::NEG_INFINITY;
        let mut lower = f64::INFINITY;
        let last = self.length - 1;
        for age in 0..self.length {
            let idx = last - age;
            let projected_high = self.high_window[idx] + self.high_slopes[idx] * age as f64;
            let projected_low = self.low_window[idx] + self.low_slopes[idx] * age as f64;
            if projected_high > upper {
                upper = projected_high;
            }
            if projected_low < lower {
                lower = projected_low;
            }
        }

        let range = upper - lower;
        let raw = if range.abs() <= f64::EPSILON {
            0.0
        } else {
            100.0 * (source - lower) / range
        };

        let pbo = self.pbo_wma.update(raw)?;
        let signal = self.signal_wma.update(pbo).unwrap_or(f64::NAN);
        Some((pbo, signal))
    }

    #[inline(always)]
    pub fn get_pbo_warmup_period(&self) -> usize {
        pbo_warmup_prefix(self.length, self.smooth_length)
    }

    #[inline(always)]
    pub fn get_signal_warmup_period(&self) -> usize {
        signal_warmup_prefix(self.length, self.smooth_length)
    }
}

#[inline(always)]
fn push_ring(buf: &mut VecDeque<f64>, value: f64, cap: usize) {
    if cap == 0 {
        return;
    }
    if buf.len() == cap {
        buf.pop_front();
    }
    buf.push_back(value);
}

#[inline(always)]
fn linreg_slope_from_window(window: &VecDeque<f64>) -> f64 {
    let n = window.len();
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    let sum_x = (n * (n - 1) / 2) as f64;
    let sum_x2 = ((n - 1) * n * (2 * n - 1) / 6) as f64;
    let denom = nf * sum_x2 - sum_x * sum_x;
    if denom.abs() <= f64::EPSILON {
        return 0.0;
    }
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    for (idx, &value) in window.iter().enumerate() {
        let x = idx as f64;
        sum_y += value;
        sum_xy += x * value;
    }
    (nf * sum_xy - sum_x * sum_y) / denom
}

#[inline(always)]
fn is_valid_triple(high: f64, low: f64, source: f64) -> bool {
    high.is_finite() && low.is_finite() && source.is_finite()
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a ProjectionOscillatorInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), ProjectionOscillatorError> {
    match &input.data {
        ProjectionOscillatorData::Candles { candles, source } => Ok((
            candles.high.as_slice(),
            candles.low.as_slice(),
            candle_source(candles, source),
        )),
        ProjectionOscillatorData::Slices { high, low, source } => Ok((high, low, source)),
    }
}

#[inline(always)]
fn candle_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "open" => candles.open.as_slice(),
        "high" => candles.high.as_slice(),
        "low" => candles.low.as_slice(),
        "close" => candles.close.as_slice(),
        "volume" => candles.volume.as_slice(),
        "hl2" => candles.hl2.as_slice(),
        "hlc3" => candles.hlc3.as_slice(),
        "ohlc4" => candles.ohlc4.as_slice(),
        "hlcc4" | "hlcc" => candles.hlcc4.as_slice(),
        _ => source_type(candles, source),
    }
}

#[inline(always)]
fn validate_params(length: usize, smooth_length: usize) -> Result<(), ProjectionOscillatorError> {
    if length == 0 {
        return Err(ProjectionOscillatorError::InvalidLength { length });
    }
    if smooth_length == 0 {
        return Err(ProjectionOscillatorError::InvalidSmoothLength { smooth_length });
    }
    Ok(())
}

#[inline(always)]
fn signal_needed_bars(
    length: usize,
    smooth_length: usize,
) -> Result<usize, ProjectionOscillatorError> {
    length
        .checked_mul(2)
        .and_then(|v| smooth_length.checked_mul(2).and_then(|s| v.checked_add(s)))
        .and_then(|v| v.checked_sub(3))
        .ok_or_else(|| ProjectionOscillatorError::InvalidInput {
            msg: "projection_oscillator: warmup overflow".to_string(),
        })
}

#[inline(always)]
fn pbo_warmup_prefix(length: usize, smooth_length: usize) -> usize {
    length
        .saturating_mul(2)
        .saturating_add(smooth_length)
        .saturating_sub(3)
}

#[inline(always)]
fn signal_warmup_prefix(length: usize, smooth_length: usize) -> usize {
    length
        .saturating_mul(2)
        .saturating_add(smooth_length.saturating_mul(2))
        .saturating_sub(4)
}

#[inline(always)]
fn longest_valid_run(high: &[f64], low: &[f64], source: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for ((&h, &l), &s) in high.iter().zip(low.iter()).zip(source.iter()) {
        if is_valid_triple(h, l, s) {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

fn validate_common(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    smooth_length: usize,
) -> Result<(), ProjectionOscillatorError> {
    validate_params(length, smooth_length)?;
    if high.is_empty() || low.is_empty() || source.is_empty() {
        return Err(ProjectionOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != source.len() {
        return Err(ProjectionOscillatorError::InputLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            source_len: source.len(),
        });
    }
    let longest = longest_valid_run(high, low, source);
    if longest == 0 {
        return Err(ProjectionOscillatorError::AllValuesNaN);
    }
    let needed = signal_needed_bars(length, smooth_length)?;
    if longest < needed {
        return Err(ProjectionOscillatorError::NotEnoughValidData {
            needed,
            valid: longest,
        });
    }
    Ok(())
}

#[inline(always)]
fn compute_row(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    smooth_length: usize,
    out_pbo: &mut [f64],
    out_signal: &mut [f64],
) {
    if length == 14 && smooth_length == 4 {
        compute_row_default_14_4(high, low, source, out_pbo, out_signal);
        return;
    }

    let mut stream = ProjectionOscillatorStream::try_new(ProjectionOscillatorParams {
        length: Some(length),
        smooth_length: Some(smooth_length),
    })
    .expect("validated params");

    for i in 0..high.len() {
        if let Some((pbo, signal)) = stream.update(high[i], low[i], source[i]) {
            out_pbo[i] = pbo;
            if signal.is_finite() {
                out_signal[i] = signal;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Wma4State {
    values: [f64; 4],
    pos: usize,
    len: usize,
    sum: f64,
    weighted_sum: f64,
}

impl Wma4State {
    #[inline(always)]
    fn new() -> Self {
        Self {
            values: [0.0; 4],
            pos: 0,
            len: 0,
            sum: 0.0,
            weighted_sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.pos = 0;
        self.len = 0;
        self.sum = 0.0;
        self.weighted_sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if self.len < 4 {
            let weight = self.len + 1;
            self.values[self.len] = value;
            self.len += 1;
            self.sum += value;
            self.weighted_sum += value * weight as f64;
            if self.len == 4 {
                Some(self.weighted_sum / 10.0)
            } else {
                None
            }
        } else {
            let oldest = self.values[self.pos];
            let old_sum = self.sum;
            self.values[self.pos] = value;
            self.pos = (self.pos + 1) & 3;
            self.sum = old_sum - oldest + value;
            self.weighted_sum = self.weighted_sum - old_sum + 4.0 * value;
            Some(self.weighted_sum / 10.0)
        }
    }
}

#[inline(always)]
fn push_fixed_14(values: &mut [f64; 14], len: &mut usize, value: f64) {
    if *len < 14 {
        values[*len] = value;
        *len += 1;
    } else {
        values.copy_within(1.., 0);
        values[13] = value;
    }
}

#[inline(always)]
fn push_slope_fixed_14(
    values: &mut [f64; 14],
    len: &mut usize,
    finite_count: &mut usize,
    value: f64,
) {
    if *len < 14 {
        values[*len] = value;
        *len += 1;
    } else {
        if values[0].is_finite() {
            *finite_count -= 1;
        }
        values.copy_within(1.., 0);
        values[13] = value;
    }
    if value.is_finite() {
        *finite_count += 1;
    }
}

#[inline(always)]
fn linreg_slope_14(values: &[f64; 14]) -> f64 {
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    for (idx, &value) in values.iter().enumerate() {
        let x = idx as f64;
        sum_y += value;
        sum_xy += x * value;
    }
    (14.0 * sum_xy - 91.0 * sum_y) / 3185.0
}

fn compute_row_default_14_4(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    out_pbo: &mut [f64],
    out_signal: &mut [f64],
) {
    let mut high_window = [0.0; 14];
    let mut low_window = [0.0; 14];
    let mut high_slopes = [f64::NAN; 14];
    let mut low_slopes = [f64::NAN; 14];
    let mut high_len = 0usize;
    let mut low_len = 0usize;
    let mut high_slope_len = 0usize;
    let mut low_slope_len = 0usize;
    let mut high_slope_finite = 0usize;
    let mut low_slope_finite = 0usize;
    let mut pbo_wma = Wma4State::new();
    let mut signal_wma = Wma4State::new();

    for i in 0..high.len() {
        let h = high[i];
        let l = low[i];
        let s = source[i];
        if !is_valid_triple(h, l, s) {
            high_len = 0;
            low_len = 0;
            high_slope_len = 0;
            low_slope_len = 0;
            high_slope_finite = 0;
            low_slope_finite = 0;
            pbo_wma.reset();
            signal_wma.reset();
            continue;
        }

        push_fixed_14(&mut high_window, &mut high_len, h);
        push_fixed_14(&mut low_window, &mut low_len, l);

        let high_slope = if high_len == 14 {
            linreg_slope_14(&high_window)
        } else {
            f64::NAN
        };
        let low_slope = if low_len == 14 {
            linreg_slope_14(&low_window)
        } else {
            f64::NAN
        };
        push_slope_fixed_14(
            &mut high_slopes,
            &mut high_slope_len,
            &mut high_slope_finite,
            high_slope,
        );
        push_slope_fixed_14(
            &mut low_slopes,
            &mut low_slope_len,
            &mut low_slope_finite,
            low_slope,
        );

        if high_len != 14 || low_len != 14 || high_slope_len != 14 || low_slope_len != 14 {
            continue;
        }
        if high_slope_finite != 14 || low_slope_finite != 14 {
            continue;
        }

        let mut upper = f64::NEG_INFINITY;
        let mut lower = f64::INFINITY;
        for age in 0..14 {
            let idx = 13 - age;
            let age_f = age as f64;
            let projected_high = high_window[idx] + high_slopes[idx] * age_f;
            let projected_low = low_window[idx] + low_slopes[idx] * age_f;
            if projected_high > upper {
                upper = projected_high;
            }
            if projected_low < lower {
                lower = projected_low;
            }
        }

        let range = upper - lower;
        let raw = if range.abs() <= f64::EPSILON {
            0.0
        } else {
            100.0 * (s - lower) / range
        };

        if let Some(pbo) = pbo_wma.update(raw) {
            out_pbo[i] = pbo;
            if let Some(signal) = signal_wma.update(pbo) {
                out_signal[i] = signal;
            }
        }
    }
}

pub fn projection_oscillator(
    input: &ProjectionOscillatorInput,
) -> Result<ProjectionOscillatorOutput, ProjectionOscillatorError> {
    projection_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn projection_oscillator_with_kernel(
    input: &ProjectionOscillatorInput,
    kernel: Kernel,
) -> Result<ProjectionOscillatorOutput, ProjectionOscillatorError> {
    let (high, low, source) = input_slices(input)?;
    let length = input.get_length();
    let smooth_length = input.get_smooth_length();
    validate_common(high, low, source, length, smooth_length)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut pbo = alloc_with_nan_prefix(high.len(), pbo_warmup_prefix(length, smooth_length));
    let mut signal = alloc_with_nan_prefix(high.len(), signal_warmup_prefix(length, smooth_length));
    compute_row(
        high,
        low,
        source,
        length,
        smooth_length,
        &mut pbo,
        &mut signal,
    );
    Ok(ProjectionOscillatorOutput { pbo, signal })
}

pub fn projection_oscillator_into_slice(
    out_pbo: &mut [f64],
    out_signal: &mut [f64],
    input: &ProjectionOscillatorInput,
    kernel: Kernel,
) -> Result<(), ProjectionOscillatorError> {
    let (high, low, source) = input_slices(input)?;
    if out_pbo.len() != high.len() {
        return Err(ProjectionOscillatorError::OutputLengthMismatch {
            expected: high.len(),
            got: out_pbo.len(),
        });
    }
    if out_signal.len() != high.len() {
        return Err(ProjectionOscillatorError::OutputLengthMismatch {
            expected: high.len(),
            got: out_signal.len(),
        });
    }
    let length = input.get_length();
    let smooth_length = input.get_smooth_length();
    validate_common(high, low, source, length, smooth_length)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    out_pbo.fill(f64::NAN);
    out_signal.fill(f64::NAN);
    compute_row(
        high,
        low,
        source,
        length,
        smooth_length,
        out_pbo,
        out_signal,
    );
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn projection_oscillator_into(
    input: &ProjectionOscillatorInput,
    out_pbo: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), ProjectionOscillatorError> {
    projection_oscillator_into_slice(out_pbo, out_signal, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct ProjectionOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub smooth_length: (usize, usize, usize),
}

impl Default for ProjectionOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 14, 0),
            smooth_length: (4, 4, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectionOscillatorBatchOutput {
    pub pbo: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<ProjectionOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct ProjectionOscillatorBatchBuilder {
    range: ProjectionOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for ProjectionOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: ProjectionOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl ProjectionOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.length = value;
        self
    }

    #[inline(always)]
    pub fn smooth_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.smooth_length = value;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        source: &[f64],
    ) -> Result<ProjectionOscillatorBatchOutput, ProjectionOscillatorError> {
        projection_oscillator_batch_with_kernel(high, low, source, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<ProjectionOscillatorBatchOutput, ProjectionOscillatorError> {
        projection_oscillator_batch_with_kernel(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

fn expand_axis(range: (usize, usize, usize)) -> Result<Vec<usize>, ProjectionOscillatorError> {
    let (start, end, step) = range;
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(ProjectionOscillatorError::InvalidRange { start, end, step });
    }
    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(cur);
        if cur >= end {
            break;
        }
        let next =
            cur.checked_add(step)
                .ok_or_else(|| ProjectionOscillatorError::InvalidInput {
                    msg: "projection_oscillator: range step overflow".to_string(),
                })?;
        if next <= cur {
            return Err(ProjectionOscillatorError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
    }
    Ok(out)
}

fn expand_grid_checked(
    sweep: &ProjectionOscillatorBatchRange,
) -> Result<Vec<ProjectionOscillatorParams>, ProjectionOscillatorError> {
    let lengths = expand_axis(sweep.length)?;
    let smooth_lengths = expand_axis(sweep.smooth_length)?;
    let total = lengths
        .len()
        .checked_mul(smooth_lengths.len())
        .ok_or_else(|| ProjectionOscillatorError::InvalidInput {
            msg: "projection_oscillator: parameter grid size overflow".to_string(),
        })?;
    let mut out = Vec::with_capacity(total);
    for &length in &lengths {
        for &smooth_length in &smooth_lengths {
            out.push(ProjectionOscillatorParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            });
        }
    }
    Ok(out)
}

pub fn expand_grid_projection_oscillator(
    sweep: &ProjectionOscillatorBatchRange,
) -> Result<Vec<ProjectionOscillatorParams>, ProjectionOscillatorError> {
    expand_grid_checked(sweep)
}

pub fn projection_oscillator_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &ProjectionOscillatorBatchRange,
    kernel: Kernel,
) -> Result<ProjectionOscillatorBatchOutput, ProjectionOscillatorError> {
    projection_oscillator_batch_inner(high, low, source, sweep, kernel, true)
}

pub fn projection_oscillator_batch_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &ProjectionOscillatorBatchRange,
    kernel: Kernel,
) -> Result<ProjectionOscillatorBatchOutput, ProjectionOscillatorError> {
    projection_oscillator_batch_inner(high, low, source, sweep, kernel, false)
}

pub fn projection_oscillator_batch_par_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &ProjectionOscillatorBatchRange,
    kernel: Kernel,
) -> Result<ProjectionOscillatorBatchOutput, ProjectionOscillatorError> {
    projection_oscillator_batch_inner(high, low, source, sweep, kernel, true)
}

fn projection_oscillator_batch_inner(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &ProjectionOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<ProjectionOscillatorBatchOutput, ProjectionOscillatorError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| ProjectionOscillatorError::InvalidInput {
            msg: "projection_oscillator: rows*cols overflow in batch".to_string(),
        })?;

    if high.is_empty() || low.is_empty() || source.is_empty() {
        return Err(ProjectionOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != source.len() {
        return Err(ProjectionOscillatorError::InputLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            source_len: source.len(),
        });
    }

    let mut pbo_warmups = Vec::with_capacity(rows);
    let mut signal_warmups = Vec::with_capacity(rows);
    let mut max_needed = 0usize;
    for combo in &combos {
        let length = combo.length.unwrap_or(14);
        let smooth_length = combo.smooth_length.unwrap_or(4);
        validate_params(length, smooth_length)?;
        pbo_warmups.push(pbo_warmup_prefix(length, smooth_length));
        signal_warmups.push(signal_warmup_prefix(length, smooth_length));
        max_needed = max_needed.max(signal_needed_bars(length, smooth_length)?);
    }

    let longest = longest_valid_run(high, low, source);
    if longest == 0 {
        return Err(ProjectionOscillatorError::AllValuesNaN);
    }
    if longest < max_needed {
        return Err(ProjectionOscillatorError::NotEnoughValidData {
            needed: max_needed,
            valid: longest,
        });
    }

    let mut pbo_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut pbo_mu, cols, &pbo_warmups);
    init_matrix_prefixes(&mut signal_mu, cols, &signal_warmups);

    let mut pbo = unsafe {
        Vec::from_raw_parts(
            pbo_mu.as_mut_ptr() as *mut f64,
            pbo_mu.len(),
            pbo_mu.capacity(),
        )
    };
    let mut signal = unsafe {
        Vec::from_raw_parts(
            signal_mu.as_mut_ptr() as *mut f64,
            signal_mu.len(),
            signal_mu.capacity(),
        )
    };
    std::mem::forget(pbo_mu);
    std::mem::forget(signal_mu);
    debug_assert_eq!(pbo.len(), total);
    debug_assert_eq!(signal.len(), total);

    projection_oscillator_batch_inner_into(
        high,
        low,
        source,
        sweep,
        kernel,
        parallel,
        &mut pbo,
        &mut signal,
    )?;

    Ok(ProjectionOscillatorBatchOutput {
        pbo,
        signal,
        combos,
        rows,
        cols,
    })
}

fn projection_oscillator_batch_inner_into(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &ProjectionOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_pbo: &mut [f64],
    out_signal: &mut [f64],
) -> Result<Vec<ProjectionOscillatorParams>, ProjectionOscillatorError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(ProjectionOscillatorError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let len = high.len();
    if len == 0 || low.is_empty() || source.is_empty() {
        return Err(ProjectionOscillatorError::EmptyInputData);
    }
    if len != low.len() || len != source.len() {
        return Err(ProjectionOscillatorError::InputLengthMismatch {
            high_len: len,
            low_len: low.len(),
            source_len: source.len(),
        });
    }

    let total =
        combos
            .len()
            .checked_mul(len)
            .ok_or_else(|| ProjectionOscillatorError::InvalidInput {
                msg: "projection_oscillator: rows*cols overflow in batch_into".to_string(),
            })?;
    if out_pbo.len() != total {
        return Err(ProjectionOscillatorError::MismatchedOutputLen {
            dst_len: out_pbo.len(),
            expected_len: total,
        });
    }
    if out_signal.len() != total {
        return Err(ProjectionOscillatorError::MismatchedOutputLen {
            dst_len: out_signal.len(),
            expected_len: total,
        });
    }

    let longest = longest_valid_run(high, low, source);
    if longest == 0 {
        return Err(ProjectionOscillatorError::AllValuesNaN);
    }
    let mut max_needed = 0usize;
    for combo in &combos {
        let length = combo.length.unwrap_or(14);
        let smooth_length = combo.smooth_length.unwrap_or(4);
        validate_params(length, smooth_length)?;
        max_needed = max_needed.max(signal_needed_bars(length, smooth_length)?);
    }
    if longest < max_needed {
        return Err(ProjectionOscillatorError::NotEnoughValidData {
            needed: max_needed,
            valid: longest,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst_pbo: &mut [f64], dst_signal: &mut [f64]| {
        let combo = &combos[row];
        dst_pbo.fill(f64::NAN);
        dst_signal.fill(f64::NAN);
        compute_row(
            high,
            low,
            source,
            combo.length.unwrap_or(14),
            combo.smooth_length.unwrap_or(4),
            dst_pbo,
            dst_signal,
        );
    };

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_pbo
                .par_chunks_mut(len)
                .zip(out_signal.par_chunks_mut(len))
                .enumerate()
                .for_each(|(row, (dst_pbo, dst_signal))| worker(row, dst_pbo, dst_signal));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (dst_pbo, dst_signal)) in out_pbo
                .chunks_mut(len)
                .zip(out_signal.chunks_mut(len))
                .enumerate()
            {
                worker(row, dst_pbo, dst_signal);
            }
        }
    } else {
        for (row, (dst_pbo, dst_signal)) in out_pbo
            .chunks_mut(len)
            .zip(out_signal.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_pbo, dst_signal);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "projection_oscillator", signature = (high, low, source, length=14, smooth_length=4, kernel=None))]
pub fn projection_oscillator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    source: PyReadonlyArray1<'py, f64>,
    length: usize,
    smooth_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let source = source.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = ProjectionOscillatorInput::from_slices(
        high,
        low,
        source,
        ProjectionOscillatorParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let out = py
        .allow_threads(|| projection_oscillator_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.pbo.into_pyarray(py), out.signal.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "ProjectionOscillatorStream")]
pub struct ProjectionOscillatorStreamPy {
    inner: ProjectionOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ProjectionOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=14, smooth_length=4))]
    fn new(length: usize, smooth_length: usize) -> PyResult<Self> {
        let inner = ProjectionOscillatorStream::try_new(ProjectionOscillatorParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, high: f64, low: f64, source: f64) -> Option<(f64, f64)> {
        self.inner.update(high, low, source)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "projection_oscillator_batch", signature = (high, low, source, length_range=(14, 14, 0), smooth_length_range=(4, 4, 0), kernel=None))]
pub fn projection_oscillator_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    source: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let source = source.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let output = py
        .allow_threads(|| {
            projection_oscillator_batch_with_kernel(
                high,
                low,
                source,
                &ProjectionOscillatorBatchRange {
                    length: length_range,
                    smooth_length: smooth_length_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "pbo",
        output
            .pbo
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "signal",
        output
            .signal
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_lengths",
        output
            .combos
            .iter()
            .map(|params| params.smooth_length.unwrap_or(4) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_projection_oscillator_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(projection_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(projection_oscillator_batch_py, m)?)?;
    m.add_class::<ProjectionOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionOscillatorBatchConfig {
    pub length_range: Vec<usize>,
    pub smooth_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = projection_oscillator_js)]
pub fn projection_oscillator_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    smooth_length: usize,
) -> Result<JsValue, JsValue> {
    let input = ProjectionOscillatorInput::from_slices(
        high,
        low,
        source,
        ProjectionOscillatorParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let out = projection_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("pbo"),
        &serde_wasm_bindgen::to_value(&out.pbo).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = projection_oscillator_batch_js)]
pub fn projection_oscillator_batch_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: ProjectionOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.smooth_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = projection_oscillator_batch_with_kernel(
        high,
        low,
        source,
        &ProjectionOscillatorBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
            smooth_length: (
                config.smooth_length_range[0],
                config.smooth_length_range[1],
                config.smooth_length_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("pbo"),
        &serde_wasm_bindgen::to_value(&out.pbo).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
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
pub fn projection_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(2 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn projection_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn projection_oscillator_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    source_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || source_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to projection_oscillator_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (dst_pbo, dst_signal) = out.split_at_mut(len);
        let input = ProjectionOscillatorInput::from_slices(
            high,
            low,
            source,
            ProjectionOscillatorParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            },
        );
        projection_oscillator_into_slice(dst_pbo, dst_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn projection_oscillator_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    source_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    smooth_length_start: usize,
    smooth_length_end: usize,
    smooth_length_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || source_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to projection_oscillator_batch_into",
        ));
    }
    let sweep = ProjectionOscillatorBatchRange {
        length: (length_start, length_end, length_step),
        smooth_length: (smooth_length_start, smooth_length_end, smooth_length_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in projection_oscillator_batch_into")
        })?;
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_pbo, dst_signal) = out.split_at_mut(split);
        projection_oscillator_batch_inner_into(
            high,
            low,
            source,
            &sweep,
            Kernel::Auto,
            false,
            dst_pbo,
            dst_signal,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn projection_oscillator_output_into_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    smooth_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = projection_oscillator_js(high, low, source, length, smooth_length)?;
    crate::write_wasm_object_f64_outputs("projection_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn projection_oscillator_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = projection_oscillator_batch_js(high, low, source, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "projection_oscillator_batch_output_into_js",
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

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..len)
            .map(|i| {
                let x = i as f64;
                101.0 + x * 0.04 + (x * 0.11).sin() * 1.4 + (x * 0.017).cos() * 0.5
            })
            .collect();
        let low: Vec<f64> = high
            .iter()
            .enumerate()
            .map(|(i, &h)| h - 1.2 - ((i as f64) * 0.07).cos().abs() * 0.35)
            .collect();
        let close: Vec<f64> = high
            .iter()
            .zip(low.iter())
            .enumerate()
            .map(|(i, (&h, &l))| l + (h - l) * (0.35 + 0.25 * ((i as f64) * 0.09).sin().abs()))
            .collect();
        (high, low, close)
    }

    fn naive_projection_oscillator(
        high: &[f64],
        low: &[f64],
        source: &[f64],
        length: usize,
        smooth_length: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut pbo = vec![f64::NAN; high.len()];
        let mut signal = vec![f64::NAN; high.len()];
        compute_row(
            high,
            low,
            source,
            length,
            smooth_length,
            &mut pbo,
            &mut signal,
        );
        (pbo, signal)
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn projection_oscillator_matches_naive() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = sample_ohlc(256);
        let input = ProjectionOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            ProjectionOscillatorParams::default(),
        );
        let out = projection_oscillator_with_kernel(&input, Kernel::Scalar)?;
        let (expected_pbo, expected_signal) =
            naive_projection_oscillator(&high, &low, &close, 14, 4);
        assert_series_close(&out.pbo, &expected_pbo, 1e-12);
        assert_series_close(&out.signal, &expected_signal, 1e-12);
        Ok(())
    }

    #[test]
    fn projection_oscillator_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = sample_ohlc(200);
        let input = ProjectionOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            ProjectionOscillatorParams {
                length: Some(10),
                smooth_length: Some(3),
            },
        );
        let base = projection_oscillator(&input)?;
        let mut pbo = vec![0.0; close.len()];
        let mut signal = vec![0.0; close.len()];
        projection_oscillator_into_slice(&mut pbo, &mut signal, &input, Kernel::Auto)?;
        assert_series_close(&base.pbo, &pbo, 1e-12);
        assert_series_close(&base.signal, &signal, 1e-12);
        Ok(())
    }

    #[test]
    fn projection_oscillator_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = sample_ohlc(220);
        let params = ProjectionOscillatorParams {
            length: Some(14),
            smooth_length: Some(4),
        };
        let batch = projection_oscillator(&ProjectionOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            params.clone(),
        ))?;
        let mut stream = ProjectionOscillatorStream::try_new(params)?;
        let mut pbo = Vec::with_capacity(close.len());
        let mut signal = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            if let Some((p, s)) = stream.update(high[i], low[i], close[i]) {
                pbo.push(p);
                signal.push(s);
            } else {
                pbo.push(f64::NAN);
                signal.push(f64::NAN);
            }
        }
        assert_series_close(&batch.pbo, &pbo, 1e-12);
        assert_series_close(&batch.signal, &signal, 1e-12);
        Ok(())
    }

    #[test]
    fn projection_oscillator_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = sample_ohlc(180);
        let single = projection_oscillator(&ProjectionOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            ProjectionOscillatorParams::default(),
        ))?;
        let batch = projection_oscillator_batch_with_kernel(
            &high,
            &low,
            &close,
            &ProjectionOscillatorBatchRange::default(),
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_close(&batch.pbo, &single.pbo, 1e-12);
        assert_series_close(&batch.signal, &single.signal, 1e-12);
        Ok(())
    }

    #[test]
    fn projection_oscillator_rejects_invalid_params() {
        let (high, low, close) = sample_ohlc(64);
        let err = projection_oscillator(&ProjectionOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            ProjectionOscillatorParams {
                length: Some(0),
                ..ProjectionOscillatorParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            ProjectionOscillatorError::InvalidLength { .. }
        ));
    }

    #[test]
    fn projection_oscillator_dispatch_compute_returns_pbo() -> Result<(), Box<dyn Error>> {
        let (high, low, close) = sample_ohlc(160);
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "projection_oscillator",
            output_id: Some("pbo"),
            data: IndicatorDataRef::Ohlc {
                open: &close,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &[
                ParamKV {
                    key: "length",
                    value: ParamValue::Int(14),
                },
                ParamKV {
                    key: "smooth_length",
                    value: ParamValue::Int(4),
                },
            ],
            kernel: Kernel::Auto,
        })?;
        assert_eq!(out.output_id, "pbo");
        assert_eq!(out.cols, close.len());
        Ok(())
    }
}
