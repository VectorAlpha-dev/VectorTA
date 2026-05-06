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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_ADAPTIVE_LENGTH: usize = 55;
const DEFAULT_STC_LENGTH: usize = 12;
const DEFAULT_SMOOTHING_FACTOR: f64 = 0.45;
const DEFAULT_FAST_LENGTH: usize = 26;
const DEFAULT_SLOW_LENGTH: usize = 50;
const HISTOGRAM_EMA_PERIOD: usize = 9;
const SCALE_100: f64 = 100.0;
const CENTER: f64 = 50.0;
const EPS: f64 = 1.0e-12;

#[derive(Debug, Clone)]
pub enum AdaptiveSchaffTrendCycleData<'a> {
    Candles(&'a Candles),
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AdaptiveSchaffTrendCycleOutput {
    pub stc: Vec<f64>,
    pub histogram: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveSchaffTrendCycleOutputField {
    Stc,
    Histogram,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AdaptiveSchaffTrendCycleParams {
    pub adaptive_length: Option<usize>,
    pub stc_length: Option<usize>,
    pub smoothing_factor: Option<f64>,
    pub fast_length: Option<usize>,
    pub slow_length: Option<usize>,
}

impl Default for AdaptiveSchaffTrendCycleParams {
    fn default() -> Self {
        Self {
            adaptive_length: Some(DEFAULT_ADAPTIVE_LENGTH),
            stc_length: Some(DEFAULT_STC_LENGTH),
            smoothing_factor: Some(DEFAULT_SMOOTHING_FACTOR),
            fast_length: Some(DEFAULT_FAST_LENGTH),
            slow_length: Some(DEFAULT_SLOW_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveSchaffTrendCycleInput<'a> {
    pub data: AdaptiveSchaffTrendCycleData<'a>,
    pub params: AdaptiveSchaffTrendCycleParams,
}

impl<'a> AdaptiveSchaffTrendCycleInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AdaptiveSchaffTrendCycleParams) -> Self {
        Self {
            data: AdaptiveSchaffTrendCycleData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: AdaptiveSchaffTrendCycleParams,
    ) -> Self {
        Self {
            data: AdaptiveSchaffTrendCycleData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AdaptiveSchaffTrendCycleParams::default())
    }

    #[inline]
    pub fn get_adaptive_length(&self) -> usize {
        self.params
            .adaptive_length
            .unwrap_or(DEFAULT_ADAPTIVE_LENGTH)
    }

    #[inline]
    pub fn get_stc_length(&self) -> usize {
        self.params.stc_length.unwrap_or(DEFAULT_STC_LENGTH)
    }

    #[inline]
    pub fn get_smoothing_factor(&self) -> f64 {
        self.params
            .smoothing_factor
            .unwrap_or(DEFAULT_SMOOTHING_FACTOR)
    }

    #[inline]
    pub fn get_fast_length(&self) -> usize {
        self.params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH)
    }

    #[inline]
    pub fn get_slow_length(&self) -> usize {
        self.params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            AdaptiveSchaffTrendCycleData::Candles(candles) => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
            ),
            AdaptiveSchaffTrendCycleData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveSchaffTrendCycleBuilder {
    adaptive_length: Option<usize>,
    stc_length: Option<usize>,
    smoothing_factor: Option<f64>,
    fast_length: Option<usize>,
    slow_length: Option<usize>,
    kernel: Kernel,
}

impl Default for AdaptiveSchaffTrendCycleBuilder {
    fn default() -> Self {
        Self {
            adaptive_length: None,
            stc_length: None,
            smoothing_factor: None,
            fast_length: None,
            slow_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdaptiveSchaffTrendCycleBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn adaptive_length(mut self, value: usize) -> Self {
        self.adaptive_length = Some(value);
        self
    }

    #[inline]
    pub fn stc_length(mut self, value: usize) -> Self {
        self.stc_length = Some(value);
        self
    }

    #[inline]
    pub fn smoothing_factor(mut self, value: f64) -> Self {
        self.smoothing_factor = Some(value);
        self
    }

    #[inline]
    pub fn fast_length(mut self, value: usize) -> Self {
        self.fast_length = Some(value);
        self
    }

    #[inline]
    pub fn slow_length(mut self, value: usize) -> Self {
        self.slow_length = Some(value);
        self
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<AdaptiveSchaffTrendCycleOutput, AdaptiveSchaffTrendCycleError> {
        let input = AdaptiveSchaffTrendCycleInput::from_candles(
            candles,
            AdaptiveSchaffTrendCycleParams {
                adaptive_length: self.adaptive_length,
                stc_length: self.stc_length,
                smoothing_factor: self.smoothing_factor,
                fast_length: self.fast_length,
                slow_length: self.slow_length,
            },
        );
        adaptive_schaff_trend_cycle_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AdaptiveSchaffTrendCycleOutput, AdaptiveSchaffTrendCycleError> {
        let input = AdaptiveSchaffTrendCycleInput::from_slices(
            high,
            low,
            close,
            AdaptiveSchaffTrendCycleParams {
                adaptive_length: self.adaptive_length,
                stc_length: self.stc_length,
                smoothing_factor: self.smoothing_factor,
                fast_length: self.fast_length,
                slow_length: self.slow_length,
            },
        );
        adaptive_schaff_trend_cycle_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<AdaptiveSchaffTrendCycleStream, AdaptiveSchaffTrendCycleError> {
        AdaptiveSchaffTrendCycleStream::try_new(AdaptiveSchaffTrendCycleParams {
            adaptive_length: self.adaptive_length,
            stc_length: self.stc_length,
            smoothing_factor: self.smoothing_factor,
            fast_length: self.fast_length,
            slow_length: self.slow_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum AdaptiveSchaffTrendCycleError {
    #[error("adaptive_schaff_trend_cycle: Empty input data.")]
    EmptyInputData,
    #[error(
        "adaptive_schaff_trend_cycle: Input length mismatch: high={high}, low={low}, close={close}"
    )]
    DataLengthMismatch {
        high: usize,
        low: usize,
        close: usize,
    },
    #[error("adaptive_schaff_trend_cycle: All input values are invalid.")]
    AllValuesNaN,
    #[error(
        "adaptive_schaff_trend_cycle: Invalid adaptive_length: adaptive_length = {adaptive_length}, data length = {data_len}"
    )]
    InvalidAdaptiveLength {
        adaptive_length: usize,
        data_len: usize,
    },
    #[error(
        "adaptive_schaff_trend_cycle: Invalid stc_length: stc_length = {stc_length}, data length = {data_len}"
    )]
    InvalidStcLength { stc_length: usize, data_len: usize },
    #[error("adaptive_schaff_trend_cycle: Invalid smoothing_factor: {smoothing_factor}")]
    InvalidSmoothingFactor { smoothing_factor: f64 },
    #[error("adaptive_schaff_trend_cycle: Invalid fast_length: {fast_length}")]
    InvalidFastLength { fast_length: usize },
    #[error("adaptive_schaff_trend_cycle: Invalid slow_length: {slow_length}")]
    InvalidSlowLength { slow_length: usize },
    #[error(
        "adaptive_schaff_trend_cycle: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("adaptive_schaff_trend_cycle: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("adaptive_schaff_trend_cycle: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error(
        "adaptive_schaff_trend_cycle: Invalid float range: start={start}, end={end}, step={step}"
    )]
    InvalidFloatRange { start: f64, end: f64, step: f64 },
    #[error("adaptive_schaff_trend_cycle: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn valid_bar(high: f64, low: f64, close: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && high >= low
}

#[inline(always)]
fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..close.len()).find(|&i| valid_bar(high[i], low[i], close[i]))
}

#[inline(always)]
fn normalize_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => detect_best_kernel(),
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    }
}

#[inline(always)]
fn validate_lengths(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(), AdaptiveSchaffTrendCycleError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(AdaptiveSchaffTrendCycleError::EmptyInputData);
    }
    if high.len() != low.len() || low.len() != close.len() {
        return Err(AdaptiveSchaffTrendCycleError::DataLengthMismatch {
            high: high.len(),
            low: low.len(),
            close: close.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn validate_params(
    adaptive_length: usize,
    stc_length: usize,
    smoothing_factor: f64,
    fast_length: usize,
    slow_length: usize,
    len: usize,
) -> Result<(), AdaptiveSchaffTrendCycleError> {
    if adaptive_length == 0 || adaptive_length > len {
        return Err(AdaptiveSchaffTrendCycleError::InvalidAdaptiveLength {
            adaptive_length,
            data_len: len,
        });
    }
    if stc_length == 0 || stc_length > len {
        return Err(AdaptiveSchaffTrendCycleError::InvalidStcLength {
            stc_length,
            data_len: len,
        });
    }
    if !smoothing_factor.is_finite()
        || !(0.0..=1.0).contains(&smoothing_factor)
        || smoothing_factor <= 0.0
    {
        return Err(AdaptiveSchaffTrendCycleError::InvalidSmoothingFactor { smoothing_factor });
    }
    if fast_length == 0 {
        return Err(AdaptiveSchaffTrendCycleError::InvalidFastLength { fast_length });
    }
    if slow_length == 0 {
        return Err(AdaptiveSchaffTrendCycleError::InvalidSlowLength { slow_length });
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct EmaState {
    alpha: f64,
    initialized: bool,
    value: f64,
}

impl EmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            initialized: false,
            value: f64::NAN,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.initialized = false;
        self.value = f64::NAN;
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        if !self.initialized {
            self.value = value;
            self.initialized = true;
        } else {
            self.value += self.alpha * (value - self.value);
        }
        self.value
    }
}

#[derive(Clone, Debug)]
struct RollingCorrelationTime {
    period: usize,
    values: VecDeque<f64>,
    sum_x: f64,
    sum_x2: f64,
    sum_xy: f64,
    sum_y: f64,
    n_sum_y2_minus_sum_y_sq: f64,
}

impl RollingCorrelationTime {
    #[inline]
    fn new(period: usize) -> Self {
        let n = period as f64;
        let sum_y = n * (n - 1.0) * 0.5;
        let sum_y2 = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
        Self {
            period,
            values: VecDeque::with_capacity(period),
            sum_x: 0.0,
            sum_x2: 0.0,
            sum_xy: 0.0,
            sum_y,
            n_sum_y2_minus_sum_y_sq: n * sum_y2 - sum_y * sum_y,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.values.clear();
        self.sum_x = 0.0;
        self.sum_x2 = 0.0;
        self.sum_xy = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.values.len() < self.period {
            let idx = self.values.len() as f64;
            self.values.push_back(value);
            self.sum_x += value;
            self.sum_x2 += value * value;
            self.sum_xy += idx * value;
            if self.values.len() == self.period {
                return Some(self.compute());
            }
            return None;
        }

        let old_sum_x = self.sum_x;
        let old_first = self.values.pop_front().unwrap_or(0.0);
        self.values.push_back(value);
        self.sum_x = old_sum_x - old_first + value;
        self.sum_x2 = self.sum_x2 - old_first * old_first + value * value;
        self.sum_xy = self.sum_xy - (old_sum_x - old_first) + (self.period as f64 - 1.0) * value;
        Some(self.compute())
    }

    #[inline]
    fn compute(&self) -> f64 {
        if self.period <= 1 {
            return 0.0;
        }

        let n = self.period as f64;
        let numerator = n * self.sum_xy - self.sum_x * self.sum_y;
        let denom_x = n * self.sum_x2 - self.sum_x * self.sum_x;
        if denom_x <= EPS || self.n_sum_y2_minus_sum_y_sq <= EPS {
            return 0.0;
        }

        let corr = numerator / (denom_x * self.n_sum_y2_minus_sum_y_sq).sqrt();
        corr.clamp(-1.0, 1.0)
    }
}

#[derive(Clone, Debug)]
struct RollingMinMax {
    period: usize,
    next_index: usize,
    min_q: VecDeque<(usize, f64)>,
    max_q: VecDeque<(usize, f64)>,
}

impl RollingMinMax {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            next_index: 0,
            min_q: VecDeque::with_capacity(period),
            max_q: VecDeque::with_capacity(period),
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.next_index = 0;
        self.min_q.clear();
        self.max_q.clear();
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let idx = self.next_index;
        self.next_index += 1;

        while let Some((_, back)) = self.min_q.back() {
            if *back <= value {
                break;
            }
            self.min_q.pop_back();
        }
        self.min_q.push_back((idx, value));

        while let Some((_, back)) = self.max_q.back() {
            if *back >= value {
                break;
            }
            self.max_q.pop_back();
        }
        self.max_q.push_back((idx, value));

        let window_start = idx.saturating_add(1).saturating_sub(self.period);
        while let Some((front_idx, _)) = self.min_q.front() {
            if *front_idx >= window_start {
                break;
            }
            self.min_q.pop_front();
        }
        while let Some((front_idx, _)) = self.max_q.front() {
            if *front_idx >= window_start {
                break;
            }
            self.max_q.pop_front();
        }

        if idx + 1 < self.period {
            return None;
        }

        Some((
            self.min_q.front().map(|(_, value)| *value).unwrap_or(value),
            self.max_q.front().map(|(_, value)| *value).unwrap_or(value),
        ))
    }
}

#[derive(Clone, Debug)]
struct AdaptiveSchaffTrendCycleCore {
    smoothing_factor: f64,
    fast_alpha: f64,
    slow_alpha: f64,
    correlation: RollingCorrelationTime,
    macd_window: RollingMinMax,
    smoothed_window: RollingMinMax,
    range_ema: EmaState,
    histogram_ema: EmaState,
    prev_close: f64,
    macd_prev1: f64,
    macd_prev2: f64,
    normalized_prev: f64,
    smoothed_macd_prev: f64,
    smoothed_macd_initialized: bool,
    smoothed_normalized_prev: f64,
    stc_prev: f64,
    stc_initialized: bool,
}

impl AdaptiveSchaffTrendCycleCore {
    #[inline]
    fn new(
        adaptive_length: usize,
        stc_length: usize,
        smoothing_factor: f64,
        fast_length: usize,
        slow_length: usize,
    ) -> Self {
        Self {
            smoothing_factor,
            fast_alpha: 2.0 / (fast_length as f64 + 1.0),
            slow_alpha: 2.0 / (slow_length as f64 + 1.0),
            correlation: RollingCorrelationTime::new(adaptive_length),
            macd_window: RollingMinMax::new(stc_length),
            smoothed_window: RollingMinMax::new(stc_length),
            range_ema: EmaState::new(slow_length),
            histogram_ema: EmaState::new(HISTOGRAM_EMA_PERIOD),
            prev_close: f64::NAN,
            macd_prev1: 0.0,
            macd_prev2: 0.0,
            normalized_prev: 0.0,
            smoothed_macd_prev: 0.0,
            smoothed_macd_initialized: false,
            smoothed_normalized_prev: 0.0,
            stc_prev: 0.0,
            stc_initialized: false,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.correlation.reset();
        self.macd_window.reset();
        self.smoothed_window.reset();
        self.range_ema.reset();
        self.histogram_ema.reset();
        self.prev_close = f64::NAN;
        self.macd_prev1 = 0.0;
        self.macd_prev2 = 0.0;
        self.normalized_prev = 0.0;
        self.smoothed_macd_prev = 0.0;
        self.smoothed_macd_initialized = false;
        self.smoothed_normalized_prev = 0.0;
        self.stc_prev = 0.0;
        self.stc_initialized = false;
    }

    #[inline]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        if !valid_bar(high, low, close) {
            self.reset();
            return None;
        }

        let range_ema = self.range_ema.update(high - low);
        let correlation = self.correlation.update(close);
        let prev_close = self.prev_close;
        self.prev_close = close;

        let Some(corr) = correlation else {
            return Some((f64::NAN, f64::NAN));
        };

        let delta = if prev_close.is_finite() {
            close - prev_close
        } else {
            0.0
        };
        let r2 = 0.5 * corr * corr + 0.5;
        let k = r2 * ((1.0 - self.fast_alpha) * (1.0 - self.slow_alpha))
            + (1.0 - r2) * ((1.0 - self.fast_alpha) / (1.0 - self.slow_alpha));
        let macd = delta * (self.fast_alpha - self.slow_alpha)
            + (2.0 - self.fast_alpha - self.slow_alpha) * self.macd_prev1
            - k * self.macd_prev2;
        self.macd_prev2 = self.macd_prev1;
        self.macd_prev1 = macd;

        let histogram = if range_ema.abs() > EPS {
            let normalized_macd = macd / range_ema * SCALE_100;
            let histogram_ema = self.histogram_ema.update(normalized_macd);
            (normalized_macd - histogram_ema) * 0.5
        } else {
            f64::NAN
        };

        let Some((macd_min, macd_max)) = self.macd_window.update(macd) else {
            return Some((f64::NAN, histogram));
        };
        let macd_span = macd_max - macd_min;
        let normalized = if macd_span > EPS {
            (macd - macd_min) / macd_span * SCALE_100
        } else {
            self.normalized_prev
        };
        self.normalized_prev = normalized;

        let smoothed_macd = if !self.smoothed_macd_initialized {
            self.smoothed_macd_initialized = true;
            normalized
        } else {
            self.smoothed_macd_prev + self.smoothing_factor * (normalized - self.smoothed_macd_prev)
        };
        self.smoothed_macd_prev = smoothed_macd;

        let Some((smoothed_min, smoothed_max)) = self.smoothed_window.update(smoothed_macd) else {
            return Some((f64::NAN, histogram));
        };
        let smoothed_span = smoothed_max - smoothed_min;
        let smoothed_normalized = if smoothed_span > EPS {
            (smoothed_macd - smoothed_min) / smoothed_span * SCALE_100
        } else {
            self.smoothed_normalized_prev
        };
        self.smoothed_normalized_prev = smoothed_normalized;

        let stc_raw = if !self.stc_initialized {
            self.stc_initialized = true;
            smoothed_normalized
        } else {
            self.stc_prev + self.smoothing_factor * (smoothed_normalized - self.stc_prev)
        };
        self.stc_prev = stc_raw;

        Some((stc_raw - CENTER, histogram))
    }
}

#[inline]
fn adaptive_schaff_trend_cycle_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    adaptive_length: usize,
    stc_length: usize,
    smoothing_factor: f64,
    fast_length: usize,
    slow_length: usize,
    out_stc: &mut [f64],
    out_histogram: &mut [f64],
) {
    let mut core = AdaptiveSchaffTrendCycleCore::new(
        adaptive_length,
        stc_length,
        smoothing_factor,
        fast_length,
        slow_length,
    );

    for i in 0..close.len() {
        match core.update(high[i], low[i], close[i]) {
            Some((stc, histogram)) => {
                out_stc[i] = stc;
                out_histogram[i] = histogram;
            }
            None => {
                out_stc[i] = f64::NAN;
                out_histogram[i] = f64::NAN;
            }
        }
    }
}

#[inline]
fn adaptive_schaff_trend_cycle_output_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    adaptive_length: usize,
    stc_length: usize,
    smoothing_factor: f64,
    fast_length: usize,
    slow_length: usize,
    field: AdaptiveSchaffTrendCycleOutputField,
    out: &mut [f64],
) {
    let mut core = AdaptiveSchaffTrendCycleCore::new(
        adaptive_length,
        stc_length,
        smoothing_factor,
        fast_length,
        slow_length,
    );

    for i in 0..close.len() {
        out[i] = match core.update(high[i], low[i], close[i]) {
            Some((stc, histogram)) => match field {
                AdaptiveSchaffTrendCycleOutputField::Stc => stc,
                AdaptiveSchaffTrendCycleOutputField::Histogram => histogram,
            },
            None => f64::NAN,
        };
    }
}

#[inline]
pub fn adaptive_schaff_trend_cycle(
    input: &AdaptiveSchaffTrendCycleInput,
) -> Result<AdaptiveSchaffTrendCycleOutput, AdaptiveSchaffTrendCycleError> {
    adaptive_schaff_trend_cycle_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn adaptive_schaff_trend_cycle_with_kernel(
    input: &AdaptiveSchaffTrendCycleInput,
    kernel: Kernel,
) -> Result<AdaptiveSchaffTrendCycleOutput, AdaptiveSchaffTrendCycleError> {
    let (high, low, close) = input.as_refs();
    validate_lengths(high, low, close)?;

    let adaptive_length = input.get_adaptive_length();
    let stc_length = input.get_stc_length();
    let smoothing_factor = input.get_smoothing_factor();
    let fast_length = input.get_fast_length();
    let slow_length = input.get_slow_length();
    validate_params(
        adaptive_length,
        stc_length,
        smoothing_factor,
        fast_length,
        slow_length,
        close.len(),
    )?;

    let first_valid =
        first_valid_bar(high, low, close).ok_or(AdaptiveSchaffTrendCycleError::AllValuesNaN)?;
    let valid = close.len().saturating_sub(first_valid);
    let needed = adaptive_length.max(stc_length);
    if valid < needed {
        return Err(AdaptiveSchaffTrendCycleError::NotEnoughValidData { needed, valid });
    }

    let _kernel = normalize_kernel(kernel);
    let len = close.len();
    let mut stc = alloc_with_nan_prefix(len, first_valid);
    let mut histogram = alloc_with_nan_prefix(len, first_valid);

    adaptive_schaff_trend_cycle_row_scalar(
        high,
        low,
        close,
        adaptive_length,
        stc_length,
        smoothing_factor,
        fast_length,
        slow_length,
        &mut stc,
        &mut histogram,
    );

    Ok(AdaptiveSchaffTrendCycleOutput { stc, histogram })
}

#[inline]
pub fn adaptive_schaff_trend_cycle_into_slice(
    out_stc: &mut [f64],
    out_histogram: &mut [f64],
    input: &AdaptiveSchaffTrendCycleInput,
    kernel: Kernel,
) -> Result<(), AdaptiveSchaffTrendCycleError> {
    let (high, low, close) = input.as_refs();
    validate_lengths(high, low, close)?;
    let len = close.len();
    if out_stc.len() != len || out_histogram.len() != len {
        return Err(AdaptiveSchaffTrendCycleError::OutputLengthMismatch {
            expected: len,
            got: out_stc.len().max(out_histogram.len()),
        });
    }

    let adaptive_length = input.get_adaptive_length();
    let stc_length = input.get_stc_length();
    let smoothing_factor = input.get_smoothing_factor();
    let fast_length = input.get_fast_length();
    let slow_length = input.get_slow_length();
    validate_params(
        adaptive_length,
        stc_length,
        smoothing_factor,
        fast_length,
        slow_length,
        len,
    )?;

    let _kernel = normalize_kernel(kernel);
    adaptive_schaff_trend_cycle_row_scalar(
        high,
        low,
        close,
        adaptive_length,
        stc_length,
        smoothing_factor,
        fast_length,
        slow_length,
        out_stc,
        out_histogram,
    );
    Ok(())
}

pub fn adaptive_schaff_trend_cycle_output_into_slice(
    out: &mut [f64],
    input: &AdaptiveSchaffTrendCycleInput,
    kernel: Kernel,
    field: AdaptiveSchaffTrendCycleOutputField,
) -> Result<(), AdaptiveSchaffTrendCycleError> {
    let (high, low, close) = input.as_refs();
    validate_lengths(high, low, close)?;
    let len = close.len();
    if out.len() != len {
        return Err(AdaptiveSchaffTrendCycleError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    let adaptive_length = input.get_adaptive_length();
    let stc_length = input.get_stc_length();
    let smoothing_factor = input.get_smoothing_factor();
    let fast_length = input.get_fast_length();
    let slow_length = input.get_slow_length();
    validate_params(
        adaptive_length,
        stc_length,
        smoothing_factor,
        fast_length,
        slow_length,
        len,
    )?;

    let _kernel = normalize_kernel(kernel);
    adaptive_schaff_trend_cycle_output_row_scalar(
        high,
        low,
        close,
        adaptive_length,
        stc_length,
        smoothing_factor,
        fast_length,
        slow_length,
        field,
        out,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn adaptive_schaff_trend_cycle_into(
    input: &AdaptiveSchaffTrendCycleInput,
    out_stc: &mut [f64],
    out_histogram: &mut [f64],
) -> Result<(), AdaptiveSchaffTrendCycleError> {
    adaptive_schaff_trend_cycle_into_slice(out_stc, out_histogram, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct AdaptiveSchaffTrendCycleStream {
    core: AdaptiveSchaffTrendCycleCore,
}

impl AdaptiveSchaffTrendCycleStream {
    #[inline]
    pub fn try_new(
        params: AdaptiveSchaffTrendCycleParams,
    ) -> Result<Self, AdaptiveSchaffTrendCycleError> {
        let adaptive_length = params.adaptive_length.unwrap_or(DEFAULT_ADAPTIVE_LENGTH);
        let stc_length = params.stc_length.unwrap_or(DEFAULT_STC_LENGTH);
        let smoothing_factor = params.smoothing_factor.unwrap_or(DEFAULT_SMOOTHING_FACTOR);
        let fast_length = params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH);
        let slow_length = params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH);
        validate_params(
            adaptive_length,
            stc_length,
            smoothing_factor,
            fast_length,
            slow_length,
            usize::MAX,
        )?;
        Ok(Self {
            core: AdaptiveSchaffTrendCycleCore::new(
                adaptive_length,
                stc_length,
                smoothing_factor,
                fast_length,
                slow_length,
            ),
        })
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.core.update(high, low, close)
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveSchaffTrendCycleBatchRange {
    pub adaptive_length: (usize, usize, usize),
    pub stc_length: (usize, usize, usize),
    pub smoothing_factor: (f64, f64, f64),
    pub fast_length: (usize, usize, usize),
    pub slow_length: (usize, usize, usize),
}

impl Default for AdaptiveSchaffTrendCycleBatchRange {
    fn default() -> Self {
        Self {
            adaptive_length: (DEFAULT_ADAPTIVE_LENGTH, DEFAULT_ADAPTIVE_LENGTH, 0),
            stc_length: (DEFAULT_STC_LENGTH, DEFAULT_STC_LENGTH, 0),
            smoothing_factor: (DEFAULT_SMOOTHING_FACTOR, DEFAULT_SMOOTHING_FACTOR, 0.0),
            fast_length: (DEFAULT_FAST_LENGTH, DEFAULT_FAST_LENGTH, 0),
            slow_length: (DEFAULT_SLOW_LENGTH, DEFAULT_SLOW_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveSchaffTrendCycleBatchOutput {
    pub stc: Vec<f64>,
    pub histogram: Vec<f64>,
    pub combos: Vec<AdaptiveSchaffTrendCycleParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct AdaptiveSchaffTrendCycleBatchBuilder {
    range: AdaptiveSchaffTrendCycleBatchRange,
    kernel: Kernel,
}

impl Default for AdaptiveSchaffTrendCycleBatchBuilder {
    fn default() -> Self {
        Self {
            range: AdaptiveSchaffTrendCycleBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl AdaptiveSchaffTrendCycleBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn adaptive_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.adaptive_length = value;
        self
    }

    #[inline]
    pub fn stc_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.stc_length = value;
        self
    }

    #[inline]
    pub fn smoothing_factor_range(mut self, value: (f64, f64, f64)) -> Self {
        self.range.smoothing_factor = value;
        self
    }

    #[inline]
    pub fn fast_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.fast_length = value;
        self
    }

    #[inline]
    pub fn slow_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.slow_length = value;
        self
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AdaptiveSchaffTrendCycleBatchOutput, AdaptiveSchaffTrendCycleError> {
        adaptive_schaff_trend_cycle_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<AdaptiveSchaffTrendCycleBatchOutput, AdaptiveSchaffTrendCycleError> {
        adaptive_schaff_trend_cycle_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            &self.range,
            self.kernel,
        )
    }
}

pub fn expand_grid_adaptive_schaff_trend_cycle(
    range: &AdaptiveSchaffTrendCycleBatchRange,
) -> Result<Vec<AdaptiveSchaffTrendCycleParams>, AdaptiveSchaffTrendCycleError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, AdaptiveSchaffTrendCycleError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start <= end {
            let mut x = start;
            while x <= end {
                out.push(x);
                x = x.saturating_add(step);
                if step == 0 {
                    break;
                }
            }
        } else {
            let mut x = start;
            while x >= end {
                out.push(x);
                let next = x.saturating_sub(step);
                if next == x {
                    break;
                }
                x = next;
                if x < end {
                    break;
                }
            }
        }

        if out.is_empty() {
            return Err(AdaptiveSchaffTrendCycleError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, AdaptiveSchaffTrendCycleError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(AdaptiveSchaffTrendCycleError::InvalidFloatRange { start, end, step });
        }
        if step.abs() < EPS || (start - end).abs() < EPS {
            return Ok(vec![start]);
        }

        let step = step.abs();
        let mut out = Vec::new();
        if start <= end {
            let mut x = start;
            while x <= end + EPS {
                out.push(x);
                x += step;
            }
        } else {
            let mut x = start;
            while x + EPS >= end {
                out.push(x);
                x -= step;
            }
        }

        if out.is_empty() {
            return Err(AdaptiveSchaffTrendCycleError::InvalidFloatRange { start, end, step });
        }
        Ok(out)
    }

    let adaptive_lengths = axis_usize(range.adaptive_length)?;
    let stc_lengths = axis_usize(range.stc_length)?;
    let smoothing_factors = axis_f64(range.smoothing_factor)?;
    let fast_lengths = axis_usize(range.fast_length)?;
    let slow_lengths = axis_usize(range.slow_length)?;

    let cap = adaptive_lengths
        .len()
        .checked_mul(stc_lengths.len())
        .and_then(|value| value.checked_mul(smoothing_factors.len()))
        .and_then(|value| value.checked_mul(fast_lengths.len()))
        .and_then(|value| value.checked_mul(slow_lengths.len()))
        .ok_or(AdaptiveSchaffTrendCycleError::InvalidRange {
            start: range.adaptive_length.0.to_string(),
            end: range.adaptive_length.1.to_string(),
            step: range.adaptive_length.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &adaptive_length in &adaptive_lengths {
        for &stc_length in &stc_lengths {
            for &smoothing_factor in &smoothing_factors {
                for &fast_length in &fast_lengths {
                    for &slow_length in &slow_lengths {
                        out.push(AdaptiveSchaffTrendCycleParams {
                            adaptive_length: Some(adaptive_length),
                            stc_length: Some(stc_length),
                            smoothing_factor: Some(smoothing_factor),
                            fast_length: Some(fast_length),
                            slow_length: Some(slow_length),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn adaptive_schaff_trend_cycle_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdaptiveSchaffTrendCycleBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveSchaffTrendCycleBatchOutput, AdaptiveSchaffTrendCycleError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AdaptiveSchaffTrendCycleError::InvalidKernelForBatch(other)),
    };
    adaptive_schaff_trend_cycle_batch_par_slice(
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline]
pub fn adaptive_schaff_trend_cycle_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdaptiveSchaffTrendCycleBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveSchaffTrendCycleBatchOutput, AdaptiveSchaffTrendCycleError> {
    adaptive_schaff_trend_cycle_batch_inner(high, low, close, sweep, kernel, false)
}

#[inline]
pub fn adaptive_schaff_trend_cycle_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdaptiveSchaffTrendCycleBatchRange,
    kernel: Kernel,
) -> Result<AdaptiveSchaffTrendCycleBatchOutput, AdaptiveSchaffTrendCycleError> {
    adaptive_schaff_trend_cycle_batch_inner(high, low, close, sweep, kernel, true)
}

fn adaptive_schaff_trend_cycle_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdaptiveSchaffTrendCycleBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<AdaptiveSchaffTrendCycleBatchOutput, AdaptiveSchaffTrendCycleError> {
    validate_lengths(high, low, close)?;
    let combos = expand_grid_adaptive_schaff_trend_cycle(sweep)?;
    for params in &combos {
        validate_params(
            params.adaptive_length.unwrap_or(DEFAULT_ADAPTIVE_LENGTH),
            params.stc_length.unwrap_or(DEFAULT_STC_LENGTH),
            params.smoothing_factor.unwrap_or(DEFAULT_SMOOTHING_FACTOR),
            params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH),
            params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH),
            close.len(),
        )?;
    }

    let first_valid =
        first_valid_bar(high, low, close).ok_or(AdaptiveSchaffTrendCycleError::AllValuesNaN)?;
    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(AdaptiveSchaffTrendCycleError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;

    let mut stc_matrix = make_uninit_matrix(rows, cols);
    let mut histogram_matrix = make_uninit_matrix(rows, cols);
    let warmups = vec![first_valid; rows];
    init_matrix_prefixes(&mut stc_matrix, cols, &warmups);
    init_matrix_prefixes(&mut histogram_matrix, cols, &warmups);

    let mut stc_guard = ManuallyDrop::new(stc_matrix);
    let mut histogram_guard = ManuallyDrop::new(histogram_matrix);

    let stc_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(stc_guard.as_mut_ptr(), stc_guard.len()) };
    let histogram_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(histogram_guard.as_mut_ptr(), histogram_guard.len())
    };

    let do_row = |row: usize,
                  row_stc: &mut [MaybeUninit<f64>],
                  row_histogram: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let dst_stc =
            unsafe { std::slice::from_raw_parts_mut(row_stc.as_mut_ptr() as *mut f64, cols) };
        let dst_histogram =
            unsafe { std::slice::from_raw_parts_mut(row_histogram.as_mut_ptr() as *mut f64, cols) };
        adaptive_schaff_trend_cycle_row_scalar(
            high,
            low,
            close,
            params.adaptive_length.unwrap_or(DEFAULT_ADAPTIVE_LENGTH),
            params.stc_length.unwrap_or(DEFAULT_STC_LENGTH),
            params.smoothing_factor.unwrap_or(DEFAULT_SMOOTHING_FACTOR),
            params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH),
            params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH),
            dst_stc,
            dst_histogram,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        stc_mu
            .par_chunks_mut(cols)
            .zip(histogram_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_stc, row_histogram))| do_row(row, row_stc, row_histogram));

        #[cfg(target_arch = "wasm32")]
        for (row, (row_stc, row_histogram)) in stc_mu
            .chunks_mut(cols)
            .zip(histogram_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_stc, row_histogram);
        }
    } else {
        for (row, (row_stc, row_histogram)) in stc_mu
            .chunks_mut(cols)
            .zip(histogram_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_stc, row_histogram);
        }
    }

    let stc = unsafe {
        Vec::from_raw_parts(
            stc_guard.as_mut_ptr() as *mut f64,
            total,
            stc_guard.capacity(),
        )
    };
    let histogram = unsafe {
        Vec::from_raw_parts(
            histogram_guard.as_mut_ptr() as *mut f64,
            total,
            histogram_guard.capacity(),
        )
    };

    Ok(AdaptiveSchaffTrendCycleBatchOutput {
        stc,
        histogram,
        combos,
        rows,
        cols,
    })
}

fn adaptive_schaff_trend_cycle_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AdaptiveSchaffTrendCycleBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_stc: &mut [f64],
    out_histogram: &mut [f64],
) -> Result<Vec<AdaptiveSchaffTrendCycleParams>, AdaptiveSchaffTrendCycleError> {
    validate_lengths(high, low, close)?;
    let combos = expand_grid_adaptive_schaff_trend_cycle(sweep)?;
    for params in &combos {
        validate_params(
            params.adaptive_length.unwrap_or(DEFAULT_ADAPTIVE_LENGTH),
            params.stc_length.unwrap_or(DEFAULT_STC_LENGTH),
            params.smoothing_factor.unwrap_or(DEFAULT_SMOOTHING_FACTOR),
            params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH),
            params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH),
            close.len(),
        )?;
    }

    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(AdaptiveSchaffTrendCycleError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    if out_stc.len() != total || out_histogram.len() != total {
        return Err(AdaptiveSchaffTrendCycleError::OutputLengthMismatch {
            expected: total,
            got: out_stc.len().max(out_histogram.len()),
        });
    }

    let _kernel = kernel;
    let do_row = |row: usize, dst_stc: &mut [f64], dst_histogram: &mut [f64]| {
        let params = &combos[row];
        adaptive_schaff_trend_cycle_row_scalar(
            high,
            low,
            close,
            params.adaptive_length.unwrap_or(DEFAULT_ADAPTIVE_LENGTH),
            params.stc_length.unwrap_or(DEFAULT_STC_LENGTH),
            params.smoothing_factor.unwrap_or(DEFAULT_SMOOTHING_FACTOR),
            params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH),
            params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH),
            dst_stc,
            dst_histogram,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_stc
            .par_chunks_mut(cols)
            .zip(out_histogram.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (dst_stc, dst_histogram))| do_row(row, dst_stc, dst_histogram));

        #[cfg(target_arch = "wasm32")]
        for (row, (dst_stc, dst_histogram)) in out_stc
            .chunks_mut(cols)
            .zip(out_histogram.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_stc, dst_histogram);
        }
    } else {
        for (row, (dst_stc, dst_histogram)) in out_stc
            .chunks_mut(cols)
            .zip(out_histogram.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_stc, dst_histogram);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "adaptive_schaff_trend_cycle")]
#[pyo3(signature = (high, low, close, adaptive_length=DEFAULT_ADAPTIVE_LENGTH, stc_length=DEFAULT_STC_LENGTH, smoothing_factor=DEFAULT_SMOOTHING_FACTOR, fast_length=DEFAULT_FAST_LENGTH, slow_length=DEFAULT_SLOW_LENGTH, kernel=None))]
pub fn adaptive_schaff_trend_cycle_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    adaptive_length: usize,
    stc_length: usize,
    smoothing_factor: f64,
    fast_length: usize,
    slow_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let input = AdaptiveSchaffTrendCycleInput::from_slices(
        high,
        low,
        close,
        AdaptiveSchaffTrendCycleParams {
            adaptive_length: Some(adaptive_length),
            stc_length: Some(stc_length),
            smoothing_factor: Some(smoothing_factor),
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| adaptive_schaff_trend_cycle_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.stc.into_pyarray(py), out.histogram.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "AdaptiveSchaffTrendCycleStream")]
pub struct AdaptiveSchaffTrendCycleStreamPy {
    stream: AdaptiveSchaffTrendCycleStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdaptiveSchaffTrendCycleStreamPy {
    #[new]
    #[pyo3(signature = (adaptive_length=DEFAULT_ADAPTIVE_LENGTH, stc_length=DEFAULT_STC_LENGTH, smoothing_factor=DEFAULT_SMOOTHING_FACTOR, fast_length=DEFAULT_FAST_LENGTH, slow_length=DEFAULT_SLOW_LENGTH))]
    fn new(
        adaptive_length: usize,
        stc_length: usize,
        smoothing_factor: f64,
        fast_length: usize,
        slow_length: usize,
    ) -> PyResult<Self> {
        let stream = AdaptiveSchaffTrendCycleStream::try_new(AdaptiveSchaffTrendCycleParams {
            adaptive_length: Some(adaptive_length),
            stc_length: Some(stc_length),
            smoothing_factor: Some(smoothing_factor),
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "adaptive_schaff_trend_cycle_batch")]
#[pyo3(signature = (high, low, close, adaptive_length_range, stc_length_range, smoothing_factor_range, fast_length_range, slow_length_range, kernel=None))]
pub fn adaptive_schaff_trend_cycle_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    adaptive_length_range: (usize, usize, usize),
    stc_length_range: (usize, usize, usize),
    smoothing_factor_range: (f64, f64, f64),
    fast_length_range: (usize, usize, usize),
    slow_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = AdaptiveSchaffTrendCycleBatchRange {
        adaptive_length: adaptive_length_range,
        stc_length: stc_length_range,
        smoothing_factor: smoothing_factor_range,
        fast_length: fast_length_range,
        slow_length: slow_length_range,
    };
    let combos = expand_grid_adaptive_schaff_trend_cycle(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let stc_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let histogram_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_stc = unsafe { stc_arr.as_slice_mut()? };
    let out_histogram = unsafe { histogram_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        adaptive_schaff_trend_cycle_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_stc,
            out_histogram,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let adaptive_lengths: Vec<u64> = combos
        .iter()
        .map(|params| params.adaptive_length.unwrap_or(DEFAULT_ADAPTIVE_LENGTH) as u64)
        .collect();
    let stc_lengths: Vec<u64> = combos
        .iter()
        .map(|params| params.stc_length.unwrap_or(DEFAULT_STC_LENGTH) as u64)
        .collect();
    let smoothing_factors: Vec<f64> = combos
        .iter()
        .map(|params| params.smoothing_factor.unwrap_or(DEFAULT_SMOOTHING_FACTOR))
        .collect();
    let fast_lengths: Vec<u64> = combos
        .iter()
        .map(|params| params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH) as u64)
        .collect();
    let slow_lengths: Vec<u64> = combos
        .iter()
        .map(|params| params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH) as u64)
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("stc", stc_arr.reshape((rows, cols))?)?;
    dict.set_item("histogram", histogram_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("adaptive_lengths", adaptive_lengths.into_pyarray(py))?;
    dict.set_item("stc_lengths", stc_lengths.into_pyarray(py))?;
    dict.set_item("smoothing_factors", smoothing_factors.into_pyarray(py))?;
    dict.set_item("fast_lengths", fast_lengths.into_pyarray(py))?;
    dict.set_item("slow_lengths", slow_lengths.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_adaptive_schaff_trend_cycle_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adaptive_schaff_trend_cycle_py, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_schaff_trend_cycle_batch_py, m)?)?;
    m.add_class::<AdaptiveSchaffTrendCycleStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdaptiveSchaffTrendCycleJsOutput {
    stc: Vec<f64>,
    histogram: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdaptiveSchaffTrendCycleBatchConfig {
    adaptive_length_range: Vec<usize>,
    stc_length_range: Vec<usize>,
    smoothing_factor_range: Vec<f64>,
    fast_length_range: Vec<usize>,
    slow_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdaptiveSchaffTrendCycleBatchJsOutput {
    stc: Vec<f64>,
    histogram: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<AdaptiveSchaffTrendCycleParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "adaptive_schaff_trend_cycle")]
pub fn adaptive_schaff_trend_cycle_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    adaptive_length: usize,
    stc_length: usize,
    smoothing_factor: f64,
    fast_length: usize,
    slow_length: usize,
) -> Result<JsValue, JsValue> {
    let input = AdaptiveSchaffTrendCycleInput::from_slices(
        high,
        low,
        close,
        AdaptiveSchaffTrendCycleParams {
            adaptive_length: Some(adaptive_length),
            stc_length: Some(stc_length),
            smoothing_factor: Some(smoothing_factor),
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
        },
    );
    let out = adaptive_schaff_trend_cycle(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&AdaptiveSchaffTrendCycleJsOutput {
        stc: out.stc,
        histogram: out.histogram,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_schaff_trend_cycle_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    adaptive_length: usize,
    stc_length: usize,
    smoothing_factor: f64,
    fast_length: usize,
    slow_length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to adaptive_schaff_trend_cycle_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 2);
        let (out_stc, out_histogram) = out.split_at_mut(len);
        let input = AdaptiveSchaffTrendCycleInput::from_slices(
            high,
            low,
            close,
            AdaptiveSchaffTrendCycleParams {
                adaptive_length: Some(adaptive_length),
                stc_length: Some(stc_length),
                smoothing_factor: Some(smoothing_factor),
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
            },
        );
        adaptive_schaff_trend_cycle_into_slice(out_stc, out_histogram, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "adaptive_schaff_trend_cycle_into_host")]
pub fn adaptive_schaff_trend_cycle_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out_ptr: *mut f64,
    adaptive_length: usize,
    stc_length: usize,
    smoothing_factor: f64,
    fast_length: usize,
    slow_length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to adaptive_schaff_trend_cycle_into_host",
        ));
    }

    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, close.len() * 2);
        let (out_stc, out_histogram) = out.split_at_mut(close.len());
        let input = AdaptiveSchaffTrendCycleInput::from_slices(
            high,
            low,
            close,
            AdaptiveSchaffTrendCycleParams {
                adaptive_length: Some(adaptive_length),
                stc_length: Some(stc_length),
                smoothing_factor: Some(smoothing_factor),
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
            },
        );
        adaptive_schaff_trend_cycle_into_slice(out_stc, out_histogram, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_schaff_trend_cycle_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len * 2];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_schaff_trend_cycle_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 2);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "adaptive_schaff_trend_cycle_batch")]
pub fn adaptive_schaff_trend_cycle_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AdaptiveSchaffTrendCycleBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.adaptive_length_range.len() != 3
        || config.stc_length_range.len() != 3
        || config.smoothing_factor_range.len() != 3
        || config.fast_length_range.len() != 3
        || config.slow_length_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = AdaptiveSchaffTrendCycleBatchRange {
        adaptive_length: (
            config.adaptive_length_range[0],
            config.adaptive_length_range[1],
            config.adaptive_length_range[2],
        ),
        stc_length: (
            config.stc_length_range[0],
            config.stc_length_range[1],
            config.stc_length_range[2],
        ),
        smoothing_factor: (
            config.smoothing_factor_range[0],
            config.smoothing_factor_range[1],
            config.smoothing_factor_range[2],
        ),
        fast_length: (
            config.fast_length_range[0],
            config.fast_length_range[1],
            config.fast_length_range[2],
        ),
        slow_length: (
            config.slow_length_range[0],
            config.slow_length_range[1],
            config.slow_length_range[2],
        ),
    };
    let batch = adaptive_schaff_trend_cycle_batch_slice(high, low, close, &sweep, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&AdaptiveSchaffTrendCycleBatchJsOutput {
        stc: batch.stc,
        histogram: batch.histogram,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_schaff_trend_cycle_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    stc_ptr: *mut f64,
    histogram_ptr: *mut f64,
    len: usize,
    adaptive_length_start: usize,
    adaptive_length_end: usize,
    adaptive_length_step: usize,
    stc_length_start: usize,
    stc_length_end: usize,
    stc_length_step: usize,
    smoothing_factor_start: f64,
    smoothing_factor_end: f64,
    smoothing_factor_step: f64,
    fast_length_start: usize,
    fast_length_end: usize,
    fast_length_step: usize,
    slow_length_start: usize,
    slow_length_end: usize,
    slow_length_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || stc_ptr.is_null()
        || histogram_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to adaptive_schaff_trend_cycle_batch_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = AdaptiveSchaffTrendCycleBatchRange {
            adaptive_length: (
                adaptive_length_start,
                adaptive_length_end,
                adaptive_length_step,
            ),
            stc_length: (stc_length_start, stc_length_end, stc_length_step),
            smoothing_factor: (
                smoothing_factor_start,
                smoothing_factor_end,
                smoothing_factor_step,
            ),
            fast_length: (fast_length_start, fast_length_end, fast_length_step),
            slow_length: (slow_length_start, slow_length_end, slow_length_step),
        };
        let combos = expand_grid_adaptive_schaff_trend_cycle(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out_stc = std::slice::from_raw_parts_mut(stc_ptr, total);
        let out_histogram = std::slice::from_raw_parts_mut(histogram_ptr, total);
        adaptive_schaff_trend_cycle_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            Kernel::Scalar,
            false,
            out_stc,
            out_histogram,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_schaff_trend_cycle_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    adaptive_length: usize,
    stc_length: usize,
    smoothing_factor: f64,
    fast_length: usize,
    slow_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_schaff_trend_cycle_js(
        high,
        low,
        close,
        adaptive_length,
        stc_length,
        smoothing_factor,
        fast_length,
        slow_length,
    )?;
    crate::write_wasm_object_f64_outputs("adaptive_schaff_trend_cycle_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn adaptive_schaff_trend_cycle_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = adaptive_schaff_trend_cycle_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "adaptive_schaff_trend_cycle_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
        ParamValue,
    };

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&lhs, &rhs)) in a.iter().zip(b.iter()).enumerate() {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(
                    lhs.is_nan() && rhs.is_nan(),
                    "nan mismatch at {i}: {lhs} vs {rhs}"
                );
            } else {
                assert!(
                    (lhs - rhs).abs() <= tol,
                    "mismatch at {i}: {lhs} vs {rhs} with tol {tol}"
                );
            }
        }
    }

    fn sample_hlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + i as f64 * 0.17 + (i as f64 * 0.031).sin() * 2.2;
            let spread = 1.1 + (i as f64 * 0.07).cos().abs() * 1.4;
            let c = base + (i as f64 * 0.11).sin() * 0.75;
            high.push(base + spread);
            low.push(base - spread);
            close.push(c);
        }
        (high, low, close)
    }

    fn check_output_contract(kernel: Kernel) {
        let (high, low, close) = sample_hlc(320);
        let input = AdaptiveSchaffTrendCycleInput::from_slices(
            &high,
            &low,
            &close,
            AdaptiveSchaffTrendCycleParams::default(),
        );
        let out = adaptive_schaff_trend_cycle_with_kernel(&input, kernel).expect("indicator");
        assert_eq!(out.stc.len(), close.len());
        assert_eq!(out.histogram.len(), close.len());
        assert!(out.stc.iter().any(|v| v.is_finite()));
        assert!(out.histogram.iter().any(|v| v.is_finite()));
    }

    fn check_into_matches_api(kernel: Kernel) {
        let (high, low, close) = sample_hlc(240);
        let input = AdaptiveSchaffTrendCycleInput::from_slices(
            &high,
            &low,
            &close,
            AdaptiveSchaffTrendCycleParams {
                adaptive_length: Some(40),
                stc_length: Some(10),
                smoothing_factor: Some(0.38),
                fast_length: Some(20),
                slow_length: Some(42),
            },
        );
        let baseline = adaptive_schaff_trend_cycle_with_kernel(&input, kernel).expect("baseline");
        let mut stc = vec![0.0; close.len()];
        let mut histogram = vec![0.0; close.len()];
        adaptive_schaff_trend_cycle_into_slice(&mut stc, &mut histogram, &input, kernel)
            .expect("into");
        assert_close(&baseline.stc, &stc, 1e-12);
        assert_close(&baseline.histogram, &histogram, 1e-12);
    }

    fn check_stream_matches_batch() {
        let (high, low, close) = sample_hlc(260);
        let params = AdaptiveSchaffTrendCycleParams {
            adaptive_length: Some(34),
            stc_length: Some(9),
            smoothing_factor: Some(0.5),
            fast_length: Some(18),
            slow_length: Some(40),
        };
        let input = AdaptiveSchaffTrendCycleInput::from_slices(&high, &low, &close, params.clone());
        let batch = adaptive_schaff_trend_cycle(&input).expect("batch");
        let mut stream = AdaptiveSchaffTrendCycleStream::try_new(params).expect("stream");
        let mut stc = vec![f64::NAN; close.len()];
        let mut histogram = vec![f64::NAN; close.len()];
        for i in 0..close.len() {
            if let Some((s, h)) = stream.update(high[i], low[i], close[i]) {
                stc[i] = s;
                histogram[i] = h;
            }
        }
        assert_close(&batch.stc, &stc, 1e-12);
        assert_close(&batch.histogram, &histogram, 1e-12);
    }

    fn check_batch_single_matches_single(kernel: Kernel) {
        let (high, low, close) = sample_hlc(180);
        let batch = adaptive_schaff_trend_cycle_batch_with_kernel(
            &high,
            &low,
            &close,
            &AdaptiveSchaffTrendCycleBatchRange {
                adaptive_length: (55, 55, 0),
                stc_length: (12, 12, 0),
                smoothing_factor: (0.45, 0.45, 0.0),
                fast_length: (26, 26, 0),
                slow_length: (50, 50, 0),
            },
            kernel,
        )
        .expect("batch");
        let single = adaptive_schaff_trend_cycle(&AdaptiveSchaffTrendCycleInput::from_slices(
            &high,
            &low,
            &close,
            AdaptiveSchaffTrendCycleParams::default(),
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_close(&batch.stc[..close.len()], &single.stc, 1e-12);
        assert_close(&batch.histogram[..close.len()], &single.histogram, 1e-12);
    }

    #[test]
    fn adaptive_schaff_trend_cycle_invalid_params() {
        let (high, low, close) = sample_hlc(64);

        let err = adaptive_schaff_trend_cycle(&AdaptiveSchaffTrendCycleInput::from_slices(
            &high,
            &low,
            &close,
            AdaptiveSchaffTrendCycleParams {
                adaptive_length: Some(0),
                ..AdaptiveSchaffTrendCycleParams::default()
            },
        ))
        .expect_err("invalid adaptive length");
        assert!(matches!(
            err,
            AdaptiveSchaffTrendCycleError::InvalidAdaptiveLength { .. }
        ));

        let err = adaptive_schaff_trend_cycle(&AdaptiveSchaffTrendCycleInput::from_slices(
            &high,
            &low,
            &close,
            AdaptiveSchaffTrendCycleParams {
                smoothing_factor: Some(0.0),
                ..AdaptiveSchaffTrendCycleParams::default()
            },
        ))
        .expect_err("invalid smoothing");
        assert!(matches!(
            err,
            AdaptiveSchaffTrendCycleError::InvalidSmoothingFactor { .. }
        ));
    }

    #[test]
    fn adaptive_schaff_trend_cycle_output_contract() {
        check_output_contract(Kernel::Auto);
        check_output_contract(Kernel::Scalar);
    }

    #[test]
    fn adaptive_schaff_trend_cycle_into_matches_api() {
        check_into_matches_api(Kernel::Auto);
        check_into_matches_api(Kernel::Scalar);
    }

    #[test]
    fn adaptive_schaff_trend_cycle_stream_matches_batch() {
        check_stream_matches_batch();
    }

    #[test]
    fn adaptive_schaff_trend_cycle_batch_single_matches_single() {
        check_batch_single_matches_single(Kernel::Auto);
    }

    #[test]
    fn adaptive_schaff_trend_cycle_dispatch_matches_direct() {
        let (high, low, close) = sample_hlc(160);
        let combo = [
            ParamKV {
                key: "adaptive_length",
                value: ParamValue::Int(55),
            },
            ParamKV {
                key: "stc_length",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "smoothing_factor",
                value: ParamValue::Float(0.45),
            },
            ParamKV {
                key: "fast_length",
                value: ParamValue::Int(26),
            },
            ParamKV {
                key: "slow_length",
                value: ParamValue::Int(50),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "adaptive_schaff_trend_cycle",
            output_id: Some("stc"),
            data: IndicatorDataRef::Ohlc {
                open: &close,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };

        let batch = compute_cpu_batch(req).expect("dispatch");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());

        let direct = adaptive_schaff_trend_cycle(&AdaptiveSchaffTrendCycleInput::from_slices(
            &high,
            &low,
            &close,
            AdaptiveSchaffTrendCycleParams::default(),
        ))
        .expect("direct");
        let row = &batch.values_f64.as_ref().expect("f64 output")[0..close.len()];
        assert_close(row, &direct.stc, 1e-12);
    }
}
