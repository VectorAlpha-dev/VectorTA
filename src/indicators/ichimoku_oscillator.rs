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
use crate::utilities::helpers::detect_best_batch_kernel;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";
const DEFAULT_CONVERSION_PERIODS: usize = 9;
const DEFAULT_BASE_PERIODS: usize = 26;
const DEFAULT_LAGGING_SPAN_PERIODS: usize = 52;
const DEFAULT_DISPLACEMENT: usize = 26;
const DEFAULT_MA_LENGTH: usize = 12;
const DEFAULT_SMOOTHING_LENGTH: usize = 3;
const DEFAULT_EXTRA_SMOOTHING: bool = true;
const DEFAULT_WINDOW_SIZE: usize = 20;
const DEFAULT_CLAMP: bool = true;
const DEFAULT_TOP_BAND: f64 = 2.0;
const DEFAULT_MID_BAND: f64 = 1.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub enum IchimokuOscillatorNormalizeMode {
    All,
    Window,
    Disabled,
}

impl Default for IchimokuOscillatorNormalizeMode {
    fn default() -> Self {
        Self::Window
    }
}

impl std::str::FromStr for IchimokuOscillatorNormalizeMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "all" => Ok(Self::All),
            "window" => Ok(Self::Window),
            "disabled" => Ok(Self::Disabled),
            other => Err(format!("Unknown normalize mode: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum IchimokuOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        source: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct IchimokuOscillatorOutput {
    pub signal: Vec<f64>,
    pub ma: Vec<f64>,
    pub conversion: Vec<f64>,
    pub base: Vec<f64>,
    pub chikou: Vec<f64>,
    pub current_kumo_a: Vec<f64>,
    pub current_kumo_b: Vec<f64>,
    pub future_kumo_a: Vec<f64>,
    pub future_kumo_b: Vec<f64>,
    pub max_level: Vec<f64>,
    pub high_level: Vec<f64>,
    pub low_level: Vec<f64>,
    pub min_level: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct IchimokuOscillatorParams {
    pub conversion_periods: Option<usize>,
    pub base_periods: Option<usize>,
    pub lagging_span_periods: Option<usize>,
    pub displacement: Option<usize>,
    pub ma_length: Option<usize>,
    pub smoothing_length: Option<usize>,
    pub extra_smoothing: Option<bool>,
    pub normalize: Option<IchimokuOscillatorNormalizeMode>,
    pub window_size: Option<usize>,
    pub clamp: Option<bool>,
    pub top_band: Option<f64>,
    pub mid_band: Option<f64>,
}

impl Default for IchimokuOscillatorParams {
    fn default() -> Self {
        Self {
            conversion_periods: Some(DEFAULT_CONVERSION_PERIODS),
            base_periods: Some(DEFAULT_BASE_PERIODS),
            lagging_span_periods: Some(DEFAULT_LAGGING_SPAN_PERIODS),
            displacement: Some(DEFAULT_DISPLACEMENT),
            ma_length: Some(DEFAULT_MA_LENGTH),
            smoothing_length: Some(DEFAULT_SMOOTHING_LENGTH),
            extra_smoothing: Some(DEFAULT_EXTRA_SMOOTHING),
            normalize: Some(IchimokuOscillatorNormalizeMode::Window),
            window_size: Some(DEFAULT_WINDOW_SIZE),
            clamp: Some(DEFAULT_CLAMP),
            top_band: Some(DEFAULT_TOP_BAND),
            mid_band: Some(DEFAULT_MID_BAND),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IchimokuOscillatorInput<'a> {
    pub data: IchimokuOscillatorData<'a>,
    pub params: IchimokuOscillatorParams,
}

impl<'a> IchimokuOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: IchimokuOscillatorParams,
    ) -> Self {
        Self {
            data: IchimokuOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        source: &'a [f64],
        params: IchimokuOscillatorParams,
    ) -> Self {
        Self {
            data: IchimokuOscillatorData::Slices {
                high,
                low,
                close,
                source,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DEFAULT_SOURCE, IchimokuOscillatorParams::default())
    }
}

#[derive(Debug, Clone)]
struct ValidatedParams {
    conversion_periods: usize,
    base_periods: usize,
    lagging_span_periods: usize,
    displacement: usize,
    ma_length: usize,
    smoothing_length: usize,
    extra_smoothing: bool,
    normalize: IchimokuOscillatorNormalizeMode,
    window_size: usize,
    clamp: bool,
    top_band: f64,
    mid_band: f64,
}

impl ValidatedParams {
    fn from_params(params: &IchimokuOscillatorParams) -> Result<Self, IchimokuOscillatorError> {
        let conversion_periods = params
            .conversion_periods
            .unwrap_or(DEFAULT_CONVERSION_PERIODS);
        let base_periods = params.base_periods.unwrap_or(DEFAULT_BASE_PERIODS);
        let lagging_span_periods = params
            .lagging_span_periods
            .unwrap_or(DEFAULT_LAGGING_SPAN_PERIODS);
        let displacement = params.displacement.unwrap_or(DEFAULT_DISPLACEMENT);
        let ma_length = params.ma_length.unwrap_or(DEFAULT_MA_LENGTH);
        let smoothing_length = params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH);
        let extra_smoothing = params.extra_smoothing.unwrap_or(DEFAULT_EXTRA_SMOOTHING);
        let normalize = params.normalize.unwrap_or_default();
        let window_size = params.window_size.unwrap_or(DEFAULT_WINDOW_SIZE);
        let clamp = params.clamp.unwrap_or(DEFAULT_CLAMP);
        let top_band = params.top_band.unwrap_or(DEFAULT_TOP_BAND);
        let mid_band = params.mid_band.unwrap_or(DEFAULT_MID_BAND);

        for (name, value) in [
            ("conversion_periods", conversion_periods),
            ("base_periods", base_periods),
            ("lagging_span_periods", lagging_span_periods),
            ("displacement", displacement),
            ("ma_length", ma_length),
            ("smoothing_length", smoothing_length),
        ] {
            if value == 0 {
                return Err(IchimokuOscillatorError::InvalidPeriod {
                    name: name.to_string(),
                    value,
                });
            }
        }
        if matches!(normalize, IchimokuOscillatorNormalizeMode::Window) && window_size < 5 {
            return Err(IchimokuOscillatorError::InvalidWindowSize { window_size });
        }
        if !top_band.is_finite() || top_band < 0.0 {
            return Err(IchimokuOscillatorError::InvalidBand {
                name: "top_band".to_string(),
                value: top_band,
            });
        }
        if !mid_band.is_finite() || mid_band < 0.0 {
            return Err(IchimokuOscillatorError::InvalidBand {
                name: "mid_band".to_string(),
                value: mid_band,
            });
        }

        Ok(Self {
            conversion_periods,
            base_periods,
            lagging_span_periods,
            displacement,
            ma_length,
            smoothing_length,
            extra_smoothing,
            normalize,
            window_size,
            clamp,
            top_band,
            mid_band,
        })
    }

    fn into_params(self) -> IchimokuOscillatorParams {
        IchimokuOscillatorParams {
            conversion_periods: Some(self.conversion_periods),
            base_periods: Some(self.base_periods),
            lagging_span_periods: Some(self.lagging_span_periods),
            displacement: Some(self.displacement),
            ma_length: Some(self.ma_length),
            smoothing_length: Some(self.smoothing_length),
            extra_smoothing: Some(self.extra_smoothing),
            normalize: Some(self.normalize),
            window_size: Some(self.window_size),
            clamp: Some(self.clamp),
            top_band: Some(self.top_band),
            mid_band: Some(self.mid_band),
        }
    }

    fn min_required_history(&self) -> usize {
        self.lagging_span_periods
            .saturating_add(self.displacement)
            .saturating_sub(1)
            .max(self.base_periods)
            .max(self.conversion_periods)
            .max(self.ma_length)
    }
}

#[derive(Clone, Debug)]
pub struct IchimokuOscillatorBuilder {
    source: Option<String>,
    conversion_periods: Option<usize>,
    base_periods: Option<usize>,
    lagging_span_periods: Option<usize>,
    displacement: Option<usize>,
    ma_length: Option<usize>,
    smoothing_length: Option<usize>,
    extra_smoothing: Option<bool>,
    normalize: Option<IchimokuOscillatorNormalizeMode>,
    window_size: Option<usize>,
    clamp: Option<bool>,
    top_band: Option<f64>,
    mid_band: Option<f64>,
    kernel: Kernel,
}

impl Default for IchimokuOscillatorBuilder {
    fn default() -> Self {
        Self {
            source: None,
            conversion_periods: None,
            base_periods: None,
            lagging_span_periods: None,
            displacement: None,
            ma_length: None,
            smoothing_length: None,
            extra_smoothing: None,
            normalize: None,
            window_size: None,
            clamp: None,
            top_band: None,
            mid_band: None,
            kernel: Kernel::Auto,
        }
    }
}

impl IchimokuOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: impl Into<String>) -> Self {
        self.source = Some(value.into());
        self
    }

    #[inline(always)]
    pub fn conversion_periods(mut self, value: usize) -> Self {
        self.conversion_periods = Some(value);
        self
    }

    #[inline(always)]
    pub fn base_periods(mut self, value: usize) -> Self {
        self.base_periods = Some(value);
        self
    }

    #[inline(always)]
    pub fn lagging_span_periods(mut self, value: usize) -> Self {
        self.lagging_span_periods = Some(value);
        self
    }

    #[inline(always)]
    pub fn displacement(mut self, value: usize) -> Self {
        self.displacement = Some(value);
        self
    }

    #[inline(always)]
    pub fn ma_length(mut self, value: usize) -> Self {
        self.ma_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn smoothing_length(mut self, value: usize) -> Self {
        self.smoothing_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn extra_smoothing(mut self, value: bool) -> Self {
        self.extra_smoothing = Some(value);
        self
    }

    #[inline(always)]
    pub fn normalize(mut self, value: IchimokuOscillatorNormalizeMode) -> Self {
        self.normalize = Some(value);
        self
    }

    #[inline(always)]
    pub fn window_size(mut self, value: usize) -> Self {
        self.window_size = Some(value);
        self
    }

    #[inline(always)]
    pub fn clamp(mut self, value: bool) -> Self {
        self.clamp = Some(value);
        self
    }

    #[inline(always)]
    pub fn top_band(mut self, value: f64) -> Self {
        self.top_band = Some(value);
        self
    }

    #[inline(always)]
    pub fn mid_band(mut self, value: f64) -> Self {
        self.mid_band = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    fn params(&self) -> IchimokuOscillatorParams {
        IchimokuOscillatorParams {
            conversion_periods: self.conversion_periods,
            base_periods: self.base_periods,
            lagging_span_periods: self.lagging_span_periods,
            displacement: self.displacement,
            ma_length: self.ma_length,
            smoothing_length: self.smoothing_length,
            extra_smoothing: self.extra_smoothing,
            normalize: self.normalize,
            window_size: self.window_size,
            clamp: self.clamp,
            top_band: self.top_band,
            mid_band: self.mid_band,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<IchimokuOscillatorOutput, IchimokuOscillatorError> {
        let input = IchimokuOscillatorInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or(DEFAULT_SOURCE),
            self.params(),
        );
        ichimoku_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
    ) -> Result<IchimokuOscillatorOutput, IchimokuOscillatorError> {
        let input = IchimokuOscillatorInput::from_slices(high, low, close, source, self.params());
        ichimoku_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<IchimokuOscillatorStream, IchimokuOscillatorError> {
        IchimokuOscillatorStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum IchimokuOscillatorError {
    #[error("ichimoku_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("ichimoku_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("ichimoku_oscillator: Inconsistent slice lengths: high={high_len}, low={low_len}, close={close_len}, source={source_len}")]
    InconsistentSliceLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
        source_len: usize,
    },
    #[error("ichimoku_oscillator: Invalid period `{name}`: {value}")]
    InvalidPeriod { name: String, value: usize },
    #[error("ichimoku_oscillator: Invalid window_size: {window_size}")]
    InvalidWindowSize { window_size: usize },
    #[error("ichimoku_oscillator: Invalid band `{name}`: {value}")]
    InvalidBand { name: String, value: f64 },
    #[error("ichimoku_oscillator: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ichimoku_oscillator: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ichimoku_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ichimoku_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    source: &'a [f64],
    first: usize,
    params: ValidatedParams,
}

fn first_valid_hlcs(high: &[f64], low: &[f64], close: &[f64], source: &[f64]) -> Option<usize> {
    (0..close.len()).find(|&i| {
        high[i].is_finite() && low[i].is_finite() && close[i].is_finite() && source[i].is_finite()
    })
}

fn extract_input<'a>(
    input: &'a IchimokuOscillatorInput<'a>,
) -> Result<PreparedInput<'a>, IchimokuOscillatorError> {
    let (high, low, close, source) = match &input.data {
        IchimokuOscillatorData::Candles { candles, source } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            source_type(candles, source),
        ),
        IchimokuOscillatorData::Slices {
            high,
            low,
            close,
            source,
        } => (*high, *low, *close, *source),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() || source.is_empty() {
        return Err(IchimokuOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() || high.len() != source.len() {
        return Err(IchimokuOscillatorError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            source_len: source.len(),
        });
    }

    let params = ValidatedParams::from_params(&input.params)?;
    let first =
        first_valid_hlcs(high, low, close, source).ok_or(IchimokuOscillatorError::AllValuesNaN)?;
    let valid = close.len().saturating_sub(first);
    let needed = params.min_required_history();
    if valid < needed {
        return Err(IchimokuOscillatorError::NotEnoughValidData { needed, valid });
    }

    Ok(PreparedInput {
        high,
        low,
        close,
        source,
        first,
        params,
    })
}

#[inline(always)]
fn avg_if_finite(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() {
        0.5 * (a + b)
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn diff_if_finite(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() {
        a - b
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn gaussian_value(x: f64, bandwidth: f64) -> f64 {
    (-((x / bandwidth).powi(2)) * 0.5).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn rolling_midpoint(high: &[f64], low: &[f64], length: usize, first: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; high.len()];
    let mut maxq: VecDeque<usize> = VecDeque::with_capacity(length + 1);
    let mut minq: VecDeque<usize> = VecDeque::with_capacity(length + 1);
    let warm = first + length - 1;

    for i in first..high.len() {
        if !high[i].is_finite() || !low[i].is_finite() {
            continue;
        }
        let start = i.saturating_add(1).saturating_sub(length).max(first);
        while let Some(&front) = maxq.front() {
            if front < start {
                maxq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&front) = minq.front() {
            if front < start {
                minq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&back) = maxq.back() {
            if high[back] <= high[i] {
                maxq.pop_back();
            } else {
                break;
            }
        }
        while let Some(&back) = minq.back() {
            if low[back] >= low[i] {
                minq.pop_back();
            } else {
                break;
            }
        }
        maxq.push_back(i);
        minq.push_back(i);
        if i >= warm {
            out[i] = 0.5 * (high[*maxq.front().unwrap()] + low[*minq.front().unwrap()]);
        }
    }

    out
}

fn chebyshev_series(data: &[f64], len: usize, ripple: f64) -> Vec<f64> {
    let a = ((1.0 / len as f64) * ((1.0 / (1.0 - ripple)).acosh())).cosh();
    let b = ((1.0 / len as f64) * ((1.0 / ripple).asinh())).sinh();
    let c = (a - b) / (a + b);
    let one_minus_c = 1.0 - c;
    let mut out = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        if data[i].is_finite() {
            let prev = if i > 0 && out[i - 1].is_finite() {
                out[i - 1]
            } else {
                0.0
            };
            out[i] = one_minus_c.mul_add(data[i], c * prev);
        }
    }
    out
}

fn gaussian_kernel_series(data: &[f64], size: usize, h: f64, r: f64) -> Vec<f64> {
    let weights: Vec<f64> = (0..=size)
        .map(|i| gaussian_value((i * i) as f64 / (h * h * r), r))
        .collect();
    let mut out = vec![f64::NAN; data.len()];
    for i in size..data.len() {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        let mut ok = true;
        for j in 0..=size {
            let value = data[i - j];
            if !value.is_finite() {
                ok = false;
                break;
            }
            sum += value * weights[j];
            weight_sum += weights[j];
        }
        if ok && weight_sum != 0.0 {
            out[i] = sum / weight_sum;
        }
    }
    out
}

fn smooth_series(data: &[f64], length: usize, extra: bool) -> Vec<f64> {
    let cheb = chebyshev_series(data, length, 0.5);
    if extra {
        gaussian_kernel_series(&cheb, 4, 2.0, 1.0)
    } else {
        cheb
    }
}

fn wma_series(data: &[f64], length: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    let denom = (length * (length + 1) / 2) as f64;
    for i in (length - 1)..data.len() {
        let mut sum = 0.0;
        let mut ok = true;
        for j in 0..length {
            let value = data[i + 1 - length + j];
            if !value.is_finite() {
                ok = false;
                break;
            }
            sum += value * (j + 1) as f64;
        }
        if ok {
            out[i] = sum / denom;
        }
    }
    out
}

fn shift_back(data: &[f64], shift: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    if shift == 0 {
        out.copy_from_slice(data);
        return out;
    }
    for i in shift..data.len() {
        out[i] = data[i - shift];
    }
    out
}

fn rolling_rms_window(data: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    let mut queue = VecDeque::with_capacity(window + 1);
    let mut sum_sq = 0.0;
    for i in 0..data.len() {
        if !data[i].is_finite() {
            queue.clear();
            sum_sq = 0.0;
            continue;
        }
        let sq = data[i] * data[i];
        queue.push_back(sq);
        sum_sq += sq;
        if queue.len() > window {
            sum_sq -= queue.pop_front().unwrap_or(0.0);
        }
        if queue.len() == window && window > 1 {
            out[i] = (sum_sq / (window - 1) as f64).sqrt();
        }
    }
    out
}

fn rms_all(signal: &[f64], gate: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; signal.len()];
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for i in 0..signal.len() {
        if signal[i].is_finite() {
            sum_sq += signal[i] * signal[i];
            count += 1;
        }
        if gate[i].is_finite() && count != 0 {
            out[i] = (sum_sq / count as f64).sqrt();
        }
    }
    out
}

fn normalize_value(
    value: f64,
    min_level: f64,
    max_level: f64,
    mode: IchimokuOscillatorNormalizeMode,
    clamp: bool,
) -> f64 {
    if matches!(mode, IchimokuOscillatorNormalizeMode::Disabled) {
        return value;
    }
    if !value.is_finite()
        || !min_level.is_finite()
        || !max_level.is_finite()
        || min_level == max_level
    {
        return f64::NAN;
    }
    let mut scaled = (value - min_level) / (max_level - min_level);
    if clamp {
        scaled = scaled.clamp(0.0, 1.0);
    }
    (scaled - 0.5) * 200.0
}

fn compute_core(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    first: usize,
    params: &ValidatedParams,
) -> IchimokuOscillatorOutput {
    let conversion_raw = rolling_midpoint(high, low, params.conversion_periods, first);
    let base_raw = rolling_midpoint(high, low, params.base_periods, first);
    let span_b_raw = rolling_midpoint(high, low, params.lagging_span_periods, first);

    let kumo_a_input: Vec<f64> = conversion_raw
        .iter()
        .zip(base_raw.iter())
        .map(|(&a, &b)| avg_if_finite(a, b))
        .collect();
    let kumo_a = smooth_series(
        &kumo_a_input,
        params.smoothing_length,
        params.extra_smoothing,
    );
    let kumo_b = smooth_series(&span_b_raw, params.smoothing_length, params.extra_smoothing);
    let kumo_center: Vec<f64> = kumo_a
        .iter()
        .zip(kumo_b.iter())
        .map(|(&a, &b)| avg_if_finite(a, b))
        .collect();
    let kumo_a_centered: Vec<f64> = kumo_a
        .iter()
        .zip(kumo_center.iter())
        .map(|(&a, &c)| diff_if_finite(a, c))
        .collect();
    let kumo_b_centered: Vec<f64> = kumo_b
        .iter()
        .zip(kumo_center.iter())
        .map(|(&a, &c)| diff_if_finite(a, c))
        .collect();

    let shift = params.displacement.saturating_sub(1);
    let kumo_center_offset = shift_back(&kumo_center, shift);
    let kumo_a_offset = shift_back(&kumo_a, shift);
    let kumo_b_offset = shift_back(&kumo_b, shift);
    let kumo_a_offset_centered: Vec<f64> = kumo_a_offset
        .iter()
        .zip(kumo_center_offset.iter())
        .map(|(&a, &c)| diff_if_finite(a, c))
        .collect();
    let kumo_b_offset_centered: Vec<f64> = kumo_b_offset
        .iter()
        .zip(kumo_center_offset.iter())
        .map(|(&a, &c)| diff_if_finite(a, c))
        .collect();

    let chikou_raw = shift_back(source, params.displacement + 1);
    let chikou_input: Vec<f64> = source
        .iter()
        .zip(chikou_raw.iter())
        .map(|(&a, &b)| diff_if_finite(a, b))
        .collect();
    let signal_input: Vec<f64> = source
        .iter()
        .zip(kumo_center_offset.iter())
        .map(|(&a, &b)| diff_if_finite(a, b))
        .collect();
    let conversion_input: Vec<f64> = conversion_raw
        .iter()
        .zip(kumo_center_offset.iter())
        .map(|(&a, &b)| diff_if_finite(a, b))
        .collect();
    let base_input: Vec<f64> = base_raw
        .iter()
        .zip(kumo_center_offset.iter())
        .map(|(&a, &b)| diff_if_finite(a, b))
        .collect();

    let chikou = smooth_series(
        &chikou_input,
        params.smoothing_length,
        params.extra_smoothing,
    );
    let signal = smooth_series(
        &signal_input,
        params.smoothing_length,
        params.extra_smoothing,
    );
    let conversion = smooth_series(
        &conversion_input,
        params.smoothing_length,
        params.extra_smoothing,
    );
    let base = smooth_series(&base_input, params.smoothing_length, params.extra_smoothing);
    let ma = wma_series(&signal, params.ma_length);

    let dev = match params.normalize {
        IchimokuOscillatorNormalizeMode::All => rms_all(&signal, &kumo_a_offset),
        IchimokuOscillatorNormalizeMode::Window => rolling_rms_window(&signal, params.window_size),
        IchimokuOscillatorNormalizeMode::Disabled => vec![0.0; close.len()],
    };

    let max_level: Vec<f64> = dev
        .iter()
        .map(|&d| {
            if d.is_finite() {
                d * params.top_band
            } else {
                f64::NAN
            }
        })
        .collect();
    let min_level: Vec<f64> = dev
        .iter()
        .map(|&d| {
            if d.is_finite() {
                -d * params.top_band
            } else {
                f64::NAN
            }
        })
        .collect();
    let high_level: Vec<f64> = dev
        .iter()
        .map(|&d| {
            if d.is_finite() {
                d * params.mid_band
            } else {
                f64::NAN
            }
        })
        .collect();
    let low_level: Vec<f64> = dev
        .iter()
        .map(|&d| {
            if d.is_finite() {
                -d * params.mid_band
            } else {
                f64::NAN
            }
        })
        .collect();

    let mut out = IchimokuOscillatorOutput {
        signal: vec![f64::NAN; close.len()],
        ma: vec![f64::NAN; close.len()],
        conversion: vec![f64::NAN; close.len()],
        base: vec![f64::NAN; close.len()],
        chikou: vec![f64::NAN; close.len()],
        current_kumo_a: vec![f64::NAN; close.len()],
        current_kumo_b: vec![f64::NAN; close.len()],
        future_kumo_a: vec![f64::NAN; close.len()],
        future_kumo_b: vec![f64::NAN; close.len()],
        max_level: vec![f64::NAN; close.len()],
        high_level: vec![f64::NAN; close.len()],
        low_level: vec![f64::NAN; close.len()],
        min_level: vec![f64::NAN; close.len()],
    };

    for i in 0..close.len() {
        out.signal[i] = normalize_value(
            signal[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        out.ma[i] = normalize_value(
            ma[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        out.conversion[i] = normalize_value(
            conversion[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        out.base[i] = normalize_value(
            base[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        out.chikou[i] = normalize_value(
            chikou[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        out.future_kumo_a[i] = normalize_value(
            kumo_a_centered[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        out.future_kumo_b[i] = normalize_value(
            kumo_b_centered[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        if i >= shift {
            out.current_kumo_a[i] = normalize_value(
                kumo_a_offset_centered[i],
                min_level[i - shift],
                max_level[i - shift],
                params.normalize,
                params.clamp,
            );
            out.current_kumo_b[i] = normalize_value(
                kumo_b_offset_centered[i],
                min_level[i - shift],
                max_level[i - shift],
                params.normalize,
                params.clamp,
            );
        }
        out.max_level[i] = normalize_value(
            max_level[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        out.high_level[i] = normalize_value(
            high_level[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        out.low_level[i] = normalize_value(
            low_level[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
        out.min_level[i] = normalize_value(
            min_level[i],
            min_level[i],
            max_level[i],
            params.normalize,
            params.clamp,
        );
    }

    out
}

pub fn ichimoku_oscillator(
    input: &IchimokuOscillatorInput,
) -> Result<IchimokuOscillatorOutput, IchimokuOscillatorError> {
    ichimoku_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn ichimoku_oscillator_with_kernel(
    input: &IchimokuOscillatorInput,
    kernel: Kernel,
) -> Result<IchimokuOscillatorOutput, IchimokuOscillatorError> {
    let prepared = extract_input(input)?;
    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    Ok(compute_core(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.source,
        prepared.first,
        &prepared.params,
    ))
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[allow(clippy::too_many_arguments)]
pub fn ichimoku_oscillator_into(
    input: &IchimokuOscillatorInput,
    out_signal: &mut [f64],
    out_ma: &mut [f64],
    out_conversion: &mut [f64],
    out_base: &mut [f64],
    out_chikou: &mut [f64],
    out_current_kumo_a: &mut [f64],
    out_current_kumo_b: &mut [f64],
    out_future_kumo_a: &mut [f64],
    out_future_kumo_b: &mut [f64],
    out_max_level: &mut [f64],
    out_high_level: &mut [f64],
    out_low_level: &mut [f64],
    out_min_level: &mut [f64],
) -> Result<(), IchimokuOscillatorError> {
    ichimoku_oscillator_into_slice(
        out_signal,
        out_ma,
        out_conversion,
        out_base,
        out_chikou,
        out_current_kumo_a,
        out_current_kumo_b,
        out_future_kumo_a,
        out_future_kumo_b,
        out_max_level,
        out_high_level,
        out_low_level,
        out_min_level,
        input,
        Kernel::Auto,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ichimoku_oscillator_into_slice(
    out_signal: &mut [f64],
    out_ma: &mut [f64],
    out_conversion: &mut [f64],
    out_base: &mut [f64],
    out_chikou: &mut [f64],
    out_current_kumo_a: &mut [f64],
    out_current_kumo_b: &mut [f64],
    out_future_kumo_a: &mut [f64],
    out_future_kumo_b: &mut [f64],
    out_max_level: &mut [f64],
    out_high_level: &mut [f64],
    out_low_level: &mut [f64],
    out_min_level: &mut [f64],
    input: &IchimokuOscillatorInput,
    kernel: Kernel,
) -> Result<(), IchimokuOscillatorError> {
    let prepared = extract_input(input)?;
    let len = prepared.close.len();
    for out in [
        &*out_signal,
        &*out_ma,
        &*out_conversion,
        &*out_base,
        &*out_chikou,
        &*out_current_kumo_a,
        &*out_current_kumo_b,
        &*out_future_kumo_a,
        &*out_future_kumo_b,
        &*out_max_level,
        &*out_high_level,
        &*out_low_level,
        &*out_min_level,
    ] {
        if out.len() != len {
            return Err(IchimokuOscillatorError::OutputLengthMismatch {
                expected: len,
                got: out.len(),
            });
        }
    }
    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let out = compute_core(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.source,
        prepared.first,
        &prepared.params,
    );
    out_signal.copy_from_slice(&out.signal);
    out_ma.copy_from_slice(&out.ma);
    out_conversion.copy_from_slice(&out.conversion);
    out_base.copy_from_slice(&out.base);
    out_chikou.copy_from_slice(&out.chikou);
    out_current_kumo_a.copy_from_slice(&out.current_kumo_a);
    out_current_kumo_b.copy_from_slice(&out.current_kumo_b);
    out_future_kumo_a.copy_from_slice(&out.future_kumo_a);
    out_future_kumo_b.copy_from_slice(&out.future_kumo_b);
    out_max_level.copy_from_slice(&out.max_level);
    out_high_level.copy_from_slice(&out.high_level);
    out_low_level.copy_from_slice(&out.low_level);
    out_min_level.copy_from_slice(&out.min_level);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct IchimokuOscillatorStream {
    params: IchimokuOscillatorParams,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    source: Vec<f64>,
}

impl IchimokuOscillatorStream {
    pub fn try_new(params: IchimokuOscillatorParams) -> Result<Self, IchimokuOscillatorError> {
        let _ = ValidatedParams::from_params(&params)?;
        Ok(Self {
            params,
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            source: Vec::new(),
        })
    }

    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        source: f64,
    ) -> (
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
    ) {
        self.high.push(high);
        self.low.push(low);
        self.close.push(close);
        self.source.push(source);
        let input = IchimokuOscillatorInput::from_slices(
            &self.high,
            &self.low,
            &self.close,
            &self.source,
            self.params.clone(),
        );
        match ichimoku_oscillator(&input) {
            Ok(out) => {
                let i = self.close.len() - 1;
                (
                    out.signal[i],
                    out.ma[i],
                    out.conversion[i],
                    out.base[i],
                    out.chikou[i],
                    out.current_kumo_a[i],
                    out.current_kumo_b[i],
                    out.future_kumo_a[i],
                    out.future_kumo_b[i],
                    out.max_level[i],
                    out.high_level[i],
                    out.low_level[i],
                    out.min_level[i],
                )
            }
            Err(_) => (
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct IchimokuOscillatorBatchRange {
    pub conversion_periods: (usize, usize, usize),
    pub base_periods: (usize, usize, usize),
    pub lagging_span_periods: (usize, usize, usize),
    pub displacement: (usize, usize, usize),
    pub ma_length: (usize, usize, usize),
    pub smoothing_length: (usize, usize, usize),
    pub window_size: (usize, usize, usize),
    pub top_band: (f64, f64, f64),
    pub mid_band: (f64, f64, f64),
    pub extra_smoothing: bool,
    pub normalize: IchimokuOscillatorNormalizeMode,
    pub clamp: bool,
}

impl Default for IchimokuOscillatorBatchRange {
    fn default() -> Self {
        Self {
            conversion_periods: (DEFAULT_CONVERSION_PERIODS, DEFAULT_CONVERSION_PERIODS, 0),
            base_periods: (DEFAULT_BASE_PERIODS, DEFAULT_BASE_PERIODS, 0),
            lagging_span_periods: (
                DEFAULT_LAGGING_SPAN_PERIODS,
                DEFAULT_LAGGING_SPAN_PERIODS,
                0,
            ),
            displacement: (DEFAULT_DISPLACEMENT, DEFAULT_DISPLACEMENT, 0),
            ma_length: (DEFAULT_MA_LENGTH, DEFAULT_MA_LENGTH, 0),
            smoothing_length: (DEFAULT_SMOOTHING_LENGTH, DEFAULT_SMOOTHING_LENGTH, 0),
            window_size: (DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_SIZE, 0),
            top_band: (DEFAULT_TOP_BAND, DEFAULT_TOP_BAND, 0.0),
            mid_band: (DEFAULT_MID_BAND, DEFAULT_MID_BAND, 0.0),
            extra_smoothing: DEFAULT_EXTRA_SMOOTHING,
            normalize: IchimokuOscillatorNormalizeMode::Window,
            clamp: DEFAULT_CLAMP,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IchimokuOscillatorBatchOutput {
    pub signal: Vec<f64>,
    pub ma: Vec<f64>,
    pub conversion: Vec<f64>,
    pub base: Vec<f64>,
    pub chikou: Vec<f64>,
    pub current_kumo_a: Vec<f64>,
    pub current_kumo_b: Vec<f64>,
    pub future_kumo_a: Vec<f64>,
    pub future_kumo_b: Vec<f64>,
    pub max_level: Vec<f64>,
    pub high_level: Vec<f64>,
    pub low_level: Vec<f64>,
    pub min_level: Vec<f64>,
    pub combos: Vec<IchimokuOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct IchimokuOscillatorBatchBuilder {
    range: IchimokuOscillatorBatchRange,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for IchimokuOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: IchimokuOscillatorBatchRange::default(),
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl IchimokuOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: impl Into<String>) -> Self {
        self.source = Some(value.into());
        self
    }

    #[inline(always)]
    pub fn extra_smoothing(mut self, value: bool) -> Self {
        self.range.extra_smoothing = value;
        self
    }

    #[inline(always)]
    pub fn normalize(mut self, value: IchimokuOscillatorNormalizeMode) -> Self {
        self.range.normalize = value;
        self
    }

    #[inline(always)]
    pub fn clamp(mut self, value: bool) -> Self {
        self.range.clamp = value;
        self
    }

    #[inline(always)]
    pub fn conversion_periods_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.conversion_periods = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn base_periods_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.base_periods = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn lagging_span_periods_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lagging_span_periods = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn displacement_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.displacement = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn ma_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ma_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn smoothing_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn window_size_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.window_size = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn top_band_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.top_band = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn mid_band_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mid_band = (start, end, step);
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
    ) -> Result<IchimokuOscillatorBatchOutput, IchimokuOscillatorError> {
        let source = source_type(candles, self.source.as_deref().unwrap_or(DEFAULT_SOURCE));
        ichimoku_oscillator_batch_with_kernel(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            source,
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
        source: &[f64],
    ) -> Result<IchimokuOscillatorBatchOutput, IchimokuOscillatorError> {
        ichimoku_oscillator_batch_with_kernel(high, low, close, source, &self.range, self.kernel)
    }
}

fn axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, IchimokuOscillatorError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(IchimokuOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
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
        let mut value = start;
        while value >= end {
            out.push(value);
            match value.checked_sub(step) {
                Some(next) => value = next,
                None => break,
            }
            if value > start {
                break;
            }
        }
    }
    Ok(out)
}

fn axis_f64(start: f64, end: f64, step: f64) -> Result<Vec<f64>, IchimokuOscillatorError> {
    if (start - end).abs() <= 1e-12 {
        return Ok(vec![start]);
    }
    if !step.is_finite() || step <= 0.0 {
        return Err(IchimokuOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end + 1e-12 {
            out.push(value);
            value += step;
        }
    } else {
        let mut value = start;
        while value >= end - 1e-12 {
            out.push(value);
            value -= step;
        }
    }
    Ok(out)
}

pub fn expand_grid(
    range: &IchimokuOscillatorBatchRange,
) -> Result<Vec<IchimokuOscillatorParams>, IchimokuOscillatorError> {
    let conversion_periods = axis_usize(
        range.conversion_periods.0,
        range.conversion_periods.1,
        range.conversion_periods.2,
    )?;
    let base_periods = axis_usize(
        range.base_periods.0,
        range.base_periods.1,
        range.base_periods.2,
    )?;
    let lagging_span_periods = axis_usize(
        range.lagging_span_periods.0,
        range.lagging_span_periods.1,
        range.lagging_span_periods.2,
    )?;
    let displacement = axis_usize(
        range.displacement.0,
        range.displacement.1,
        range.displacement.2,
    )?;
    let ma_length = axis_usize(range.ma_length.0, range.ma_length.1, range.ma_length.2)?;
    let smoothing_length = axis_usize(
        range.smoothing_length.0,
        range.smoothing_length.1,
        range.smoothing_length.2,
    )?;
    let window_size = axis_usize(
        range.window_size.0,
        range.window_size.1,
        range.window_size.2,
    )?;
    let top_band = axis_f64(range.top_band.0, range.top_band.1, range.top_band.2)?;
    let mid_band = axis_f64(range.mid_band.0, range.mid_band.1, range.mid_band.2)?;

    let mut out = Vec::new();
    for &conversion_periods in &conversion_periods {
        for &base_periods in &base_periods {
            for &lagging_span_periods in &lagging_span_periods {
                for &displacement in &displacement {
                    for &ma_length in &ma_length {
                        for &smoothing_length in &smoothing_length {
                            for &window_size in &window_size {
                                for &top_band in &top_band {
                                    for &mid_band in &mid_band {
                                        out.push(IchimokuOscillatorParams {
                                            conversion_periods: Some(conversion_periods),
                                            base_periods: Some(base_periods),
                                            lagging_span_periods: Some(lagging_span_periods),
                                            displacement: Some(displacement),
                                            ma_length: Some(ma_length),
                                            smoothing_length: Some(smoothing_length),
                                            extra_smoothing: Some(range.extra_smoothing),
                                            normalize: Some(range.normalize),
                                            window_size: Some(window_size),
                                            clamp: Some(range.clamp),
                                            top_band: Some(top_band),
                                            mid_band: Some(mid_band),
                                        });
                                    }
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

fn validate_raw_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
) -> Result<usize, IchimokuOscillatorError> {
    if high.is_empty() || low.is_empty() || close.is_empty() || source.is_empty() {
        return Err(IchimokuOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() || high.len() != source.len() {
        return Err(IchimokuOscillatorError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            source_len: source.len(),
        });
    }
    first_valid_hlcs(high, low, close, source).ok_or(IchimokuOscillatorError::AllValuesNaN)
}

pub fn ichimoku_oscillator_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &IchimokuOscillatorBatchRange,
    kernel: Kernel,
) -> Result<IchimokuOscillatorBatchOutput, IchimokuOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(IchimokuOscillatorError::InvalidKernelForBatch(kernel)),
    };
    ichimoku_oscillator_batch_par_slice(
        high,
        low,
        close,
        source,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

pub fn ichimoku_oscillator_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &IchimokuOscillatorBatchRange,
    kernel: Kernel,
) -> Result<IchimokuOscillatorBatchOutput, IchimokuOscillatorError> {
    ichimoku_oscillator_batch_inner(high, low, close, source, sweep, kernel, false)
}

pub fn ichimoku_oscillator_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &IchimokuOscillatorBatchRange,
    kernel: Kernel,
) -> Result<IchimokuOscillatorBatchOutput, IchimokuOscillatorError> {
    ichimoku_oscillator_batch_inner(high, low, close, source, sweep, kernel, true)
}

fn ichimoku_oscillator_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &IchimokuOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<IchimokuOscillatorBatchOutput, IchimokuOscillatorError> {
    let rows = expand_grid(sweep)?.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IchimokuOscillatorError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })?;
    let mut signal = vec![f64::NAN; total];
    let mut ma = vec![f64::NAN; total];
    let mut conversion = vec![f64::NAN; total];
    let mut base = vec![f64::NAN; total];
    let mut chikou = vec![f64::NAN; total];
    let mut current_kumo_a = vec![f64::NAN; total];
    let mut current_kumo_b = vec![f64::NAN; total];
    let mut future_kumo_a = vec![f64::NAN; total];
    let mut future_kumo_b = vec![f64::NAN; total];
    let mut max_level = vec![f64::NAN; total];
    let mut high_level = vec![f64::NAN; total];
    let mut low_level = vec![f64::NAN; total];
    let mut min_level = vec![f64::NAN; total];
    let combos = ichimoku_oscillator_batch_inner_into(
        high,
        low,
        close,
        source,
        sweep,
        kernel,
        parallel,
        &mut signal,
        &mut ma,
        &mut conversion,
        &mut base,
        &mut chikou,
        &mut current_kumo_a,
        &mut current_kumo_b,
        &mut future_kumo_a,
        &mut future_kumo_b,
        &mut max_level,
        &mut high_level,
        &mut low_level,
        &mut min_level,
    )?;
    Ok(IchimokuOscillatorBatchOutput {
        signal,
        ma,
        conversion,
        base,
        chikou,
        current_kumo_a,
        current_kumo_b,
        future_kumo_a,
        future_kumo_b,
        max_level,
        high_level,
        low_level,
        min_level,
        combos,
        rows,
        cols,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn ichimoku_oscillator_batch_into_slice(
    out_signal: &mut [f64],
    out_ma: &mut [f64],
    out_conversion: &mut [f64],
    out_base: &mut [f64],
    out_chikou: &mut [f64],
    out_current_kumo_a: &mut [f64],
    out_current_kumo_b: &mut [f64],
    out_future_kumo_a: &mut [f64],
    out_future_kumo_b: &mut [f64],
    out_max_level: &mut [f64],
    out_high_level: &mut [f64],
    out_low_level: &mut [f64],
    out_min_level: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &IchimokuOscillatorBatchRange,
    kernel: Kernel,
) -> Result<(), IchimokuOscillatorError> {
    ichimoku_oscillator_batch_inner_into(
        high,
        low,
        close,
        source,
        sweep,
        kernel,
        false,
        out_signal,
        out_ma,
        out_conversion,
        out_base,
        out_chikou,
        out_current_kumo_a,
        out_current_kumo_b,
        out_future_kumo_a,
        out_future_kumo_b,
        out_max_level,
        out_high_level,
        out_low_level,
        out_min_level,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ichimoku_oscillator_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &IchimokuOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_signal: &mut [f64],
    out_ma: &mut [f64],
    out_conversion: &mut [f64],
    out_base: &mut [f64],
    out_chikou: &mut [f64],
    out_current_kumo_a: &mut [f64],
    out_current_kumo_b: &mut [f64],
    out_future_kumo_a: &mut [f64],
    out_future_kumo_b: &mut [f64],
    out_max_level: &mut [f64],
    out_high_level: &mut [f64],
    out_low_level: &mut [f64],
    out_min_level: &mut [f64],
) -> Result<Vec<IchimokuOscillatorParams>, IchimokuOscillatorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(high, low, close, source)?;
    let validated = combos
        .iter()
        .map(ValidatedParams::from_params)
        .collect::<Result<Vec<_>, _>>()?;
    let rows = validated.len();
    let cols = close.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| IchimokuOscillatorError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })?;
    for out in [
        &*out_signal,
        &*out_ma,
        &*out_conversion,
        &*out_base,
        &*out_chikou,
        &*out_current_kumo_a,
        &*out_current_kumo_b,
        &*out_future_kumo_a,
        &*out_future_kumo_b,
        &*out_max_level,
        &*out_high_level,
        &*out_low_level,
        &*out_min_level,
    ] {
        if out.len() != expected {
            return Err(IchimokuOscillatorError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
    }
    let max_needed = validated
        .iter()
        .map(ValidatedParams::min_required_history)
        .max()
        .unwrap_or(0);
    let valid = cols.saturating_sub(first);
    if valid < max_needed {
        return Err(IchimokuOscillatorError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }
    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };

    let do_row = |row: usize,
                  dst_signal: &mut [f64],
                  dst_ma: &mut [f64],
                  dst_conversion: &mut [f64],
                  dst_base: &mut [f64],
                  dst_chikou: &mut [f64],
                  dst_current_kumo_a: &mut [f64],
                  dst_current_kumo_b: &mut [f64],
                  dst_future_kumo_a: &mut [f64],
                  dst_future_kumo_b: &mut [f64],
                  dst_max_level: &mut [f64],
                  dst_high_level: &mut [f64],
                  dst_low_level: &mut [f64],
                  dst_min_level: &mut [f64]| {
        let out = compute_core(high, low, close, source, first, &validated[row]);
        dst_signal.copy_from_slice(&out.signal);
        dst_ma.copy_from_slice(&out.ma);
        dst_conversion.copy_from_slice(&out.conversion);
        dst_base.copy_from_slice(&out.base);
        dst_chikou.copy_from_slice(&out.chikou);
        dst_current_kumo_a.copy_from_slice(&out.current_kumo_a);
        dst_current_kumo_b.copy_from_slice(&out.current_kumo_b);
        dst_future_kumo_a.copy_from_slice(&out.future_kumo_a);
        dst_future_kumo_b.copy_from_slice(&out.future_kumo_b);
        dst_max_level.copy_from_slice(&out.max_level);
        dst_high_level.copy_from_slice(&out.high_level);
        dst_low_level.copy_from_slice(&out.low_level);
        dst_min_level.copy_from_slice(&out.min_level);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            do_row(
                row,
                &mut out_signal[start..end],
                &mut out_ma[start..end],
                &mut out_conversion[start..end],
                &mut out_base[start..end],
                &mut out_chikou[start..end],
                &mut out_current_kumo_a[start..end],
                &mut out_current_kumo_b[start..end],
                &mut out_future_kumo_a[start..end],
                &mut out_future_kumo_b[start..end],
                &mut out_max_level[start..end],
                &mut out_high_level[start..end],
                &mut out_low_level[start..end],
                &mut out_min_level[start..end],
            );
        }
        #[cfg(target_arch = "wasm32")]
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            do_row(
                row,
                &mut out_signal[start..end],
                &mut out_ma[start..end],
                &mut out_conversion[start..end],
                &mut out_base[start..end],
                &mut out_chikou[start..end],
                &mut out_current_kumo_a[start..end],
                &mut out_current_kumo_b[start..end],
                &mut out_future_kumo_a[start..end],
                &mut out_future_kumo_b[start..end],
                &mut out_max_level[start..end],
                &mut out_high_level[start..end],
                &mut out_low_level[start..end],
                &mut out_min_level[start..end],
            );
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            do_row(
                row,
                &mut out_signal[start..end],
                &mut out_ma[start..end],
                &mut out_conversion[start..end],
                &mut out_base[start..end],
                &mut out_chikou[start..end],
                &mut out_current_kumo_a[start..end],
                &mut out_current_kumo_b[start..end],
                &mut out_future_kumo_a[start..end],
                &mut out_future_kumo_b[start..end],
                &mut out_max_level[start..end],
                &mut out_high_level[start..end],
                &mut out_low_level[start..end],
                &mut out_min_level[start..end],
            );
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
fn set_output_dict<'py>(
    py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    out: IchimokuOscillatorOutput,
) -> PyResult<()> {
    dict.set_item("signal", out.signal.into_pyarray(py))?;
    dict.set_item("ma", out.ma.into_pyarray(py))?;
    dict.set_item("conversion", out.conversion.into_pyarray(py))?;
    dict.set_item("base", out.base.into_pyarray(py))?;
    dict.set_item("chikou", out.chikou.into_pyarray(py))?;
    dict.set_item("current_kumo_a", out.current_kumo_a.into_pyarray(py))?;
    dict.set_item("current_kumo_b", out.current_kumo_b.into_pyarray(py))?;
    dict.set_item("future_kumo_a", out.future_kumo_a.into_pyarray(py))?;
    dict.set_item("future_kumo_b", out.future_kumo_b.into_pyarray(py))?;
    dict.set_item("max_level", out.max_level.into_pyarray(py))?;
    dict.set_item("high_level", out.high_level.into_pyarray(py))?;
    dict.set_item("low_level", out.low_level.into_pyarray(py))?;
    dict.set_item("min_level", out.min_level.into_pyarray(py))?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "ichimoku_oscillator")]
#[pyo3(signature = (high, low, close, source=None, conversion_periods=9, base_periods=26, lagging_span_periods=52, displacement=26, ma_length=12, smoothing_length=3, extra_smoothing=true, normalize="window", window_size=20, clamp=true, top_band=2.0, mid_band=1.5, kernel=None))]
pub fn ichimoku_oscillator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    source: Option<PyReadonlyArray1<'py, f64>>,
    conversion_periods: usize,
    base_periods: usize,
    lagging_span_periods: usize,
    displacement: usize,
    ma_length: usize,
    smoothing_length: usize,
    extra_smoothing: bool,
    normalize: &str,
    window_size: usize,
    clamp: bool,
    top_band: f64,
    mid_band: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let normalize = normalize
        .parse::<IchimokuOscillatorNormalizeMode>()
        .map_err(PyValueError::new_err)?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let source_holder = source;
    let source = if let Some(ref source) = source_holder {
        source.as_slice()?
    } else {
        close
    };
    let input = IchimokuOscillatorInput::from_slices(
        high,
        low,
        close,
        source,
        IchimokuOscillatorParams {
            conversion_periods: Some(conversion_periods),
            base_periods: Some(base_periods),
            lagging_span_periods: Some(lagging_span_periods),
            displacement: Some(displacement),
            ma_length: Some(ma_length),
            smoothing_length: Some(smoothing_length),
            extra_smoothing: Some(extra_smoothing),
            normalize: Some(normalize),
            window_size: Some(window_size),
            clamp: Some(clamp),
            top_band: Some(top_band),
            mid_band: Some(mid_band),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| ichimoku_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    set_output_dict(py, &dict, out)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "IchimokuOscillatorStream")]
pub struct IchimokuOscillatorStreamPy {
    stream: IchimokuOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl IchimokuOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (conversion_periods=9, base_periods=26, lagging_span_periods=52, displacement=26, ma_length=12, smoothing_length=3, extra_smoothing=true, normalize="window", window_size=20, clamp=true, top_band=2.0, mid_band=1.5))]
    fn new(
        conversion_periods: usize,
        base_periods: usize,
        lagging_span_periods: usize,
        displacement: usize,
        ma_length: usize,
        smoothing_length: usize,
        extra_smoothing: bool,
        normalize: &str,
        window_size: usize,
        clamp: bool,
        top_band: f64,
        mid_band: f64,
    ) -> PyResult<Self> {
        let normalize = normalize
            .parse::<IchimokuOscillatorNormalizeMode>()
            .map_err(PyValueError::new_err)?;
        let stream = IchimokuOscillatorStream::try_new(IchimokuOscillatorParams {
            conversion_periods: Some(conversion_periods),
            base_periods: Some(base_periods),
            lagging_span_periods: Some(lagging_span_periods),
            displacement: Some(displacement),
            ma_length: Some(ma_length),
            smoothing_length: Some(smoothing_length),
            extra_smoothing: Some(extra_smoothing),
            normalize: Some(normalize),
            window_size: Some(window_size),
            clamp: Some(clamp),
            top_band: Some(top_band),
            mid_band: Some(mid_band),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    #[pyo3(signature = (high, low, close, source=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        high: f64,
        low: f64,
        close: f64,
        source: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let got = self
            .stream
            .update(high, low, close, source.unwrap_or(close));
        let dict = PyDict::new(py);
        dict.set_item("signal", got.0)?;
        dict.set_item("ma", got.1)?;
        dict.set_item("conversion", got.2)?;
        dict.set_item("base", got.3)?;
        dict.set_item("chikou", got.4)?;
        dict.set_item("current_kumo_a", got.5)?;
        dict.set_item("current_kumo_b", got.6)?;
        dict.set_item("future_kumo_a", got.7)?;
        dict.set_item("future_kumo_b", got.8)?;
        dict.set_item("max_level", got.9)?;
        dict.set_item("high_level", got.10)?;
        dict.set_item("low_level", got.11)?;
        dict.set_item("min_level", got.12)?;
        Ok(dict.unbind())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ichimoku_oscillator_batch")]
#[pyo3(signature = (high, low, close, source=None, conversion_periods_range=(9,9,0), base_periods_range=(26,26,0), lagging_span_periods_range=(52,52,0), displacement_range=(26,26,0), ma_length_range=(12,12,0), smoothing_length_range=(3,3,0), window_size_range=(20,20,0), top_band_range=(2.0,2.0,0.0), mid_band_range=(1.5,1.5,0.0), extra_smoothing=true, normalize="window", clamp=true, kernel=None))]
pub fn ichimoku_oscillator_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    source: Option<PyReadonlyArray1<'py, f64>>,
    conversion_periods_range: (usize, usize, usize),
    base_periods_range: (usize, usize, usize),
    lagging_span_periods_range: (usize, usize, usize),
    displacement_range: (usize, usize, usize),
    ma_length_range: (usize, usize, usize),
    smoothing_length_range: (usize, usize, usize),
    window_size_range: (usize, usize, usize),
    top_band_range: (f64, f64, f64),
    mid_band_range: (f64, f64, f64),
    extra_smoothing: bool,
    normalize: &str,
    clamp: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let normalize = normalize
        .parse::<IchimokuOscillatorNormalizeMode>()
        .map_err(PyValueError::new_err)?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let source_holder = source;
    let source = if let Some(ref source) = source_holder {
        source.as_slice()?
    } else {
        close
    };
    let sweep = IchimokuOscillatorBatchRange {
        conversion_periods: conversion_periods_range,
        base_periods: base_periods_range,
        lagging_span_periods: lagging_span_periods_range,
        displacement: displacement_range,
        ma_length: ma_length_range,
        smoothing_length: smoothing_length_range,
        window_size: window_size_range,
        top_band: top_band_range,
        mid_band: mid_band_range,
        extra_smoothing,
        normalize,
        clamp,
    };
    let kernel = validate_kernel(kernel, true)?;
    let out = py
        .allow_threads(|| {
            ichimoku_oscillator_batch_with_kernel(high, low, close, source, &sweep, kernel)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = out.rows;
    let cols = out.cols;
    let dict = PyDict::new(py);
    macro_rules! set_matrix {
        ($name:literal, $values:expr) => {{
            let arr = PyArray1::from_vec(py, $values);
            dict.set_item($name, arr.reshape((rows, cols))?)?;
        }};
    }
    set_matrix!("signal", out.signal);
    set_matrix!("ma", out.ma);
    set_matrix!("conversion", out.conversion);
    set_matrix!("base", out.base);
    set_matrix!("chikou", out.chikou);
    set_matrix!("current_kumo_a", out.current_kumo_a);
    set_matrix!("current_kumo_b", out.current_kumo_b);
    set_matrix!("future_kumo_a", out.future_kumo_a);
    set_matrix!("future_kumo_b", out.future_kumo_b);
    set_matrix!("max_level", out.max_level);
    set_matrix!("high_level", out.high_level);
    set_matrix!("low_level", out.low_level);
    set_matrix!("min_level", out.min_level);
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ichimoku_oscillator_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ichimoku_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(ichimoku_oscillator_batch_py, m)?)?;
    m.add_class::<IchimokuOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct IchimokuOscillatorJsOutput {
    pub signal: Vec<f64>,
    pub ma: Vec<f64>,
    pub conversion: Vec<f64>,
    pub base: Vec<f64>,
    pub chikou: Vec<f64>,
    pub current_kumo_a: Vec<f64>,
    pub current_kumo_b: Vec<f64>,
    pub future_kumo_a: Vec<f64>,
    pub future_kumo_b: Vec<f64>,
    pub max_level: Vec<f64>,
    pub high_level: Vec<f64>,
    pub low_level: Vec<f64>,
    pub min_level: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ichimoku_oscillator_js")]
#[allow(clippy::too_many_arguments)]
pub fn ichimoku_oscillator_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    conversion_periods: usize,
    base_periods: usize,
    lagging_span_periods: usize,
    displacement: usize,
    ma_length: usize,
    smoothing_length: usize,
    extra_smoothing: bool,
    normalize: &str,
    window_size: usize,
    clamp: bool,
    top_band: f64,
    mid_band: f64,
) -> Result<JsValue, JsValue> {
    let normalize = normalize
        .parse::<IchimokuOscillatorNormalizeMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    let input = IchimokuOscillatorInput::from_slices(
        high,
        low,
        close,
        source,
        IchimokuOscillatorParams {
            conversion_periods: Some(conversion_periods),
            base_periods: Some(base_periods),
            lagging_span_periods: Some(lagging_span_periods),
            displacement: Some(displacement),
            ma_length: Some(ma_length),
            smoothing_length: Some(smoothing_length),
            extra_smoothing: Some(extra_smoothing),
            normalize: Some(normalize),
            window_size: Some(window_size),
            clamp: Some(clamp),
            top_band: Some(top_band),
            mid_band: Some(mid_band),
        },
    );
    let out = ichimoku_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&IchimokuOscillatorJsOutput {
        signal: out.signal,
        ma: out.ma,
        conversion: out.conversion,
        base: out.base,
        chikou: out.chikou,
        current_kumo_a: out.current_kumo_a,
        current_kumo_b: out.current_kumo_b,
        future_kumo_a: out.future_kumo_a,
        future_kumo_b: out.future_kumo_b,
        max_level: out.max_level,
        high_level: out.high_level,
        low_level: out.low_level,
        min_level: out.min_level,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ichimoku_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ichimoku_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn ichimoku_oscillator_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    source_ptr: *const f64,
    signal_ptr: *mut f64,
    ma_ptr: *mut f64,
    conversion_ptr: *mut f64,
    base_ptr: *mut f64,
    chikou_ptr: *mut f64,
    current_kumo_a_ptr: *mut f64,
    current_kumo_b_ptr: *mut f64,
    future_kumo_a_ptr: *mut f64,
    future_kumo_b_ptr: *mut f64,
    max_level_ptr: *mut f64,
    high_level_ptr: *mut f64,
    low_level_ptr: *mut f64,
    min_level_ptr: *mut f64,
    len: usize,
    conversion_periods: usize,
    base_periods: usize,
    lagging_span_periods: usize,
    displacement: usize,
    ma_length: usize,
    smoothing_length: usize,
    extra_smoothing: bool,
    normalize: &str,
    window_size: usize,
    clamp: bool,
    top_band: f64,
    mid_band: f64,
) -> Result<(), JsValue> {
    if [
        high_ptr as *const (),
        low_ptr as *const (),
        close_ptr as *const (),
        source_ptr as *const (),
        signal_ptr as *const (),
        ma_ptr as *const (),
        conversion_ptr as *const (),
        base_ptr as *const (),
        chikou_ptr as *const (),
        current_kumo_a_ptr as *const (),
        current_kumo_b_ptr as *const (),
        future_kumo_a_ptr as *const (),
        future_kumo_b_ptr as *const (),
        max_level_ptr as *const (),
        high_level_ptr as *const (),
        low_level_ptr as *const (),
        min_level_ptr as *const (),
    ]
    .iter()
    .any(|ptr| ptr.is_null())
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let normalize = normalize
        .parse::<IchimokuOscillatorNormalizeMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    unsafe {
        let input = IchimokuOscillatorInput::from_slices(
            std::slice::from_raw_parts(high_ptr, len),
            std::slice::from_raw_parts(low_ptr, len),
            std::slice::from_raw_parts(close_ptr, len),
            std::slice::from_raw_parts(source_ptr, len),
            IchimokuOscillatorParams {
                conversion_periods: Some(conversion_periods),
                base_periods: Some(base_periods),
                lagging_span_periods: Some(lagging_span_periods),
                displacement: Some(displacement),
                ma_length: Some(ma_length),
                smoothing_length: Some(smoothing_length),
                extra_smoothing: Some(extra_smoothing),
                normalize: Some(normalize),
                window_size: Some(window_size),
                clamp: Some(clamp),
                top_band: Some(top_band),
                mid_band: Some(mid_band),
            },
        );
        ichimoku_oscillator_into_slice(
            std::slice::from_raw_parts_mut(signal_ptr, len),
            std::slice::from_raw_parts_mut(ma_ptr, len),
            std::slice::from_raw_parts_mut(conversion_ptr, len),
            std::slice::from_raw_parts_mut(base_ptr, len),
            std::slice::from_raw_parts_mut(chikou_ptr, len),
            std::slice::from_raw_parts_mut(current_kumo_a_ptr, len),
            std::slice::from_raw_parts_mut(current_kumo_b_ptr, len),
            std::slice::from_raw_parts_mut(future_kumo_a_ptr, len),
            std::slice::from_raw_parts_mut(future_kumo_b_ptr, len),
            std::slice::from_raw_parts_mut(max_level_ptr, len),
            std::slice::from_raw_parts_mut(high_level_ptr, len),
            std::slice::from_raw_parts_mut(low_level_ptr, len),
            std::slice::from_raw_parts_mut(min_level_ptr, len),
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct IchimokuOscillatorBatchConfig {
    pub conversion_periods_range: Vec<f64>,
    pub base_periods_range: Vec<f64>,
    pub lagging_span_periods_range: Vec<f64>,
    pub displacement_range: Vec<f64>,
    pub ma_length_range: Vec<f64>,
    pub smoothing_length_range: Vec<f64>,
    pub window_size_range: Vec<f64>,
    pub top_band_range: Vec<f64>,
    pub mid_band_range: Vec<f64>,
    pub extra_smoothing: bool,
    pub normalize: String,
    pub clamp: bool,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct IchimokuOscillatorBatchJsOutput {
    pub signal: Vec<f64>,
    pub ma: Vec<f64>,
    pub conversion: Vec<f64>,
    pub base: Vec<f64>,
    pub chikou: Vec<f64>,
    pub current_kumo_a: Vec<f64>,
    pub current_kumo_b: Vec<f64>,
    pub future_kumo_a: Vec<f64>,
    pub future_kumo_b: Vec<f64>,
    pub max_level: Vec<f64>,
    pub high_level: Vec<f64>,
    pub low_level: Vec<f64>,
    pub min_level: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_usize(name: &str, values: &[f64]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    let mut out = [0usize; 3];
    for (i, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a finite non-negative whole number"
            )));
        }
        let rounded = value.round();
        if (value - rounded).abs() > 1e-9 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a whole number"
            )));
        }
        out[i] = rounded as usize;
    }
    Ok((out[0], out[1], out[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_f64(name: &str, values: &[f64]) -> Result<(f64, f64, f64), JsValue> {
    if values.len() != 3 || values.iter().any(|value| !value.is_finite()) {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 finite elements [start, end, step]"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ichimoku_oscillator_batch_js")]
pub fn ichimoku_oscillator_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: IchimokuOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let normalize = config
        .normalize
        .parse::<IchimokuOscillatorNormalizeMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    let sweep = IchimokuOscillatorBatchRange {
        conversion_periods: js_vec3_to_usize(
            "conversion_periods_range",
            &config.conversion_periods_range,
        )?,
        base_periods: js_vec3_to_usize("base_periods_range", &config.base_periods_range)?,
        lagging_span_periods: js_vec3_to_usize(
            "lagging_span_periods_range",
            &config.lagging_span_periods_range,
        )?,
        displacement: js_vec3_to_usize("displacement_range", &config.displacement_range)?,
        ma_length: js_vec3_to_usize("ma_length_range", &config.ma_length_range)?,
        smoothing_length: js_vec3_to_usize(
            "smoothing_length_range",
            &config.smoothing_length_range,
        )?,
        window_size: js_vec3_to_usize("window_size_range", &config.window_size_range)?,
        top_band: js_vec3_to_f64("top_band_range", &config.top_band_range)?,
        mid_band: js_vec3_to_f64("mid_band_range", &config.mid_band_range)?,
        extra_smoothing: config.extra_smoothing,
        normalize,
        clamp: config.clamp,
    };
    let out = ichimoku_oscillator_batch_with_kernel(high, low, close, source, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&IchimokuOscillatorBatchJsOutput {
        signal: out.signal,
        ma: out.ma,
        conversion: out.conversion,
        base: out.base,
        chikou: out.chikou,
        current_kumo_a: out.current_kumo_a,
        current_kumo_b: out.current_kumo_b,
        future_kumo_a: out.future_kumo_a,
        future_kumo_b: out.future_kumo_b,
        max_level: out.max_level,
        high_level: out.high_level,
        low_level: out.low_level,
        min_level: out.min_level,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn ichimoku_oscillator_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    source_ptr: *const f64,
    signal_ptr: *mut f64,
    ma_ptr: *mut f64,
    conversion_ptr: *mut f64,
    base_ptr: *mut f64,
    chikou_ptr: *mut f64,
    current_kumo_a_ptr: *mut f64,
    current_kumo_b_ptr: *mut f64,
    future_kumo_a_ptr: *mut f64,
    future_kumo_b_ptr: *mut f64,
    max_level_ptr: *mut f64,
    high_level_ptr: *mut f64,
    low_level_ptr: *mut f64,
    min_level_ptr: *mut f64,
    len: usize,
    conversion_start: usize,
    conversion_end: usize,
    conversion_step: usize,
    base_start: usize,
    base_end: usize,
    base_step: usize,
    lagging_start: usize,
    lagging_end: usize,
    lagging_step: usize,
    displacement_start: usize,
    displacement_end: usize,
    displacement_step: usize,
    ma_start: usize,
    ma_end: usize,
    ma_step: usize,
    smoothing_start: usize,
    smoothing_end: usize,
    smoothing_step: usize,
    window_start: usize,
    window_end: usize,
    window_step: usize,
    top_start: f64,
    top_end: f64,
    top_step: f64,
    mid_start: f64,
    mid_end: f64,
    mid_step: f64,
    extra_smoothing: bool,
    normalize: &str,
    clamp: bool,
) -> Result<usize, JsValue> {
    let normalize = normalize
        .parse::<IchimokuOscillatorNormalizeMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    let sweep = IchimokuOscillatorBatchRange {
        conversion_periods: (conversion_start, conversion_end, conversion_step),
        base_periods: (base_start, base_end, base_step),
        lagging_span_periods: (lagging_start, lagging_end, lagging_step),
        displacement: (displacement_start, displacement_end, displacement_step),
        ma_length: (ma_start, ma_end, ma_step),
        smoothing_length: (smoothing_start, smoothing_end, smoothing_step),
        window_size: (window_start, window_end, window_step),
        top_band: (top_start, top_end, top_step),
        mid_band: (mid_start, mid_end, mid_step),
        extra_smoothing,
        normalize,
        clamp,
    };
    let rows = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in ichimoku_oscillator_batch_into"))?;
    unsafe {
        ichimoku_oscillator_batch_into_slice(
            std::slice::from_raw_parts_mut(signal_ptr, total),
            std::slice::from_raw_parts_mut(ma_ptr, total),
            std::slice::from_raw_parts_mut(conversion_ptr, total),
            std::slice::from_raw_parts_mut(base_ptr, total),
            std::slice::from_raw_parts_mut(chikou_ptr, total),
            std::slice::from_raw_parts_mut(current_kumo_a_ptr, total),
            std::slice::from_raw_parts_mut(current_kumo_b_ptr, total),
            std::slice::from_raw_parts_mut(future_kumo_a_ptr, total),
            std::slice::from_raw_parts_mut(future_kumo_b_ptr, total),
            std::slice::from_raw_parts_mut(max_level_ptr, total),
            std::slice::from_raw_parts_mut(high_level_ptr, total),
            std::slice::from_raw_parts_mut(low_level_ptr, total),
            std::slice::from_raw_parts_mut(min_level_ptr, total),
            std::slice::from_raw_parts(high_ptr, len),
            std::slice::from_raw_parts(low_ptr, len),
            std::slice::from_raw_parts(close_ptr, len),
            std::slice::from_raw_parts(source_ptr, len),
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ichimoku_oscillator_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    conversion_periods: usize,
    base_periods: usize,
    lagging_span_periods: usize,
    displacement: usize,
    ma_length: usize,
    smoothing_length: usize,
    extra_smoothing: bool,
    normalize: &str,
    window_size: usize,
    clamp: bool,
    top_band: f64,
    mid_band: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ichimoku_oscillator_js(
        high,
        low,
        close,
        source,
        conversion_periods,
        base_periods,
        lagging_span_periods,
        displacement,
        ma_length,
        smoothing_length,
        extra_smoothing,
        normalize,
        window_size,
        clamp,
        top_band,
        mid_band,
    )?;
    crate::write_wasm_object_f64_outputs("ichimoku_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ichimoku_oscillator_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ichimoku_oscillator_batch_js(high, low, close, source, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ichimoku_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hlcs(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..len)
            .map(|i| 100.0 + (i as f64 * 0.07).sin() * 3.0 + i as f64 * 0.02)
            .collect();
        let high: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 1.5 + (i as f64 * 0.11).cos().abs())
            .collect();
        let low: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c - 1.2 - (i as f64 * 0.09).sin().abs())
            .collect();
        let source: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| 0.5 * (c + (high[i] + low[i]) * 0.5))
            .collect();
        (high, low, close, source)
    }

    fn assert_close(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12, "mismatch at {idx}: {a} vs {b}");
        }
    }

    #[test]
    fn stream_matches_batch_last_values() {
        let (high, low, close, source) = sample_hlcs(160);
        let out = ichimoku_oscillator(&IchimokuOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            IchimokuOscillatorParams::default(),
        ))
        .unwrap();
        let mut stream =
            IchimokuOscillatorStream::try_new(IchimokuOscillatorParams::default()).unwrap();
        let mut got = (
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        );
        for i in 0..high.len() {
            got = stream.update(high[i], low[i], close[i], source[i]);
        }
        let i = high.len() - 1;
        let expected = [
            out.signal[i],
            out.ma[i],
            out.conversion[i],
            out.base[i],
            out.chikou[i],
            out.current_kumo_a[i],
            out.current_kumo_b[i],
            out.future_kumo_a[i],
            out.future_kumo_b[i],
            out.max_level[i],
            out.high_level[i],
            out.low_level[i],
            out.min_level[i],
        ];
        let got = [
            got.0, got.1, got.2, got.3, got.4, got.5, got.6, got.7, got.8, got.9, got.10, got.11,
            got.12,
        ];
        assert_close(&got, &expected);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (high, low, close, source) = sample_hlcs(128);
        let batch = ichimoku_oscillator_batch_with_kernel(
            &high,
            &low,
            &close,
            &source,
            &IchimokuOscillatorBatchRange {
                conversion_periods: (9, 11, 2),
                ..IchimokuOscillatorBatchRange::default()
            },
            Kernel::Auto,
        )
        .unwrap();
        let single = ichimoku_oscillator(&IchimokuOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            IchimokuOscillatorParams::default(),
        ))
        .unwrap();
        assert_eq!(batch.rows, 2);
        assert_close(&batch.signal[..128], &single.signal);
        assert_close(&batch.current_kumo_a[..128], &single.current_kumo_a);
    }

    #[test]
    fn into_slice_matches_single() {
        let (high, low, close, source) = sample_hlcs(96);
        let input = IchimokuOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            IchimokuOscillatorParams::default(),
        );
        let single = ichimoku_oscillator(&input).unwrap();
        let n = close.len();
        let (
            mut signal,
            mut ma,
            mut conversion,
            mut base,
            mut chikou,
            mut current_kumo_a,
            mut current_kumo_b,
            mut future_kumo_a,
            mut future_kumo_b,
            mut max_level,
            mut high_level,
            mut low_level,
            mut min_level,
        ) = (
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
        );
        ichimoku_oscillator_into_slice(
            &mut signal,
            &mut ma,
            &mut conversion,
            &mut base,
            &mut chikou,
            &mut current_kumo_a,
            &mut current_kumo_b,
            &mut future_kumo_a,
            &mut future_kumo_b,
            &mut max_level,
            &mut high_level,
            &mut low_level,
            &mut min_level,
            &input,
            Kernel::Auto,
        )
        .unwrap();
        assert_close(&signal, &single.signal);
        assert_close(&future_kumo_b, &single.future_kumo_b);
    }

    #[test]
    fn invalid_period_is_rejected() {
        let (high, low, close, source) = sample_hlcs(64);
        let input = IchimokuOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            IchimokuOscillatorParams {
                conversion_periods: Some(0),
                ..IchimokuOscillatorParams::default()
            },
        );
        assert!(matches!(
            ichimoku_oscillator(&input),
            Err(IchimokuOscillatorError::InvalidPeriod { .. })
        ));
    }
}
