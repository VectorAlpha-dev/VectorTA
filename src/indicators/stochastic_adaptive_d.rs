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
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_K_LENGTH: usize = 20;
const DEFAULT_D_SMOOTHING: usize = 9;
const DEFAULT_PRE_SMOOTH: usize = 20;
const DEFAULT_ATTENUATION: f64 = 2.0;
const SCALE_100: f64 = 100.0;
const CENTER: f64 = 50.0;
const EPS: f64 = 1.0e-12;

#[derive(Debug, Clone)]
pub enum StochasticAdaptiveDData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct StochasticAdaptiveDOutput {
    pub standard_d: Vec<f64>,
    pub adaptive_d: Vec<f64>,
    pub difference: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct StochasticAdaptiveDParams {
    pub k_length: Option<usize>,
    pub d_smoothing: Option<usize>,
    pub pre_smooth: Option<usize>,
    pub attenuation: Option<f64>,
}

impl Default for StochasticAdaptiveDParams {
    fn default() -> Self {
        Self {
            k_length: Some(DEFAULT_K_LENGTH),
            d_smoothing: Some(DEFAULT_D_SMOOTHING),
            pre_smooth: Some(DEFAULT_PRE_SMOOTH),
            attenuation: Some(DEFAULT_ATTENUATION),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochasticAdaptiveDInput<'a> {
    pub data: StochasticAdaptiveDData<'a>,
    pub params: StochasticAdaptiveDParams,
}

impl<'a> StochasticAdaptiveDInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: StochasticAdaptiveDParams,
    ) -> Self {
        Self {
            data: StochasticAdaptiveDData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: StochasticAdaptiveDParams,
    ) -> Self {
        Self {
            data: StochasticAdaptiveDData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", StochasticAdaptiveDParams::default())
    }

    #[inline]
    pub fn get_k_length(&self) -> usize {
        self.params.k_length.unwrap_or(DEFAULT_K_LENGTH)
    }

    #[inline]
    pub fn get_d_smoothing(&self) -> usize {
        self.params.d_smoothing.unwrap_or(DEFAULT_D_SMOOTHING)
    }

    #[inline]
    pub fn get_pre_smooth(&self) -> usize {
        self.params.pre_smooth.unwrap_or(DEFAULT_PRE_SMOOTH)
    }

    #[inline]
    pub fn get_attenuation(&self) -> f64 {
        self.params.attenuation.unwrap_or(DEFAULT_ATTENUATION)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            StochasticAdaptiveDData::Candles { candles, source } => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                if *source == "close" {
                    candles.close.as_slice()
                } else {
                    source_type(candles, source)
                },
            ),
            StochasticAdaptiveDData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StochasticAdaptiveDBuilder {
    k_length: Option<usize>,
    d_smoothing: Option<usize>,
    pre_smooth: Option<usize>,
    attenuation: Option<f64>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for StochasticAdaptiveDBuilder {
    fn default() -> Self {
        Self {
            k_length: None,
            d_smoothing: None,
            pre_smooth: None,
            attenuation: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StochasticAdaptiveDBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn k_length(mut self, value: usize) -> Self {
        self.k_length = Some(value);
        self
    }

    #[inline]
    pub fn d_smoothing(mut self, value: usize) -> Self {
        self.d_smoothing = Some(value);
        self
    }

    #[inline]
    pub fn pre_smooth(mut self, value: usize) -> Self {
        self.pre_smooth = Some(value);
        self
    }

    #[inline]
    pub fn attenuation(mut self, value: f64) -> Self {
        self.attenuation = Some(value);
        self
    }

    #[inline]
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = Some(value.into());
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
    ) -> Result<StochasticAdaptiveDOutput, StochasticAdaptiveDError> {
        let input = StochasticAdaptiveDInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or("close"),
            StochasticAdaptiveDParams {
                k_length: self.k_length,
                d_smoothing: self.d_smoothing,
                pre_smooth: self.pre_smooth,
                attenuation: self.attenuation,
            },
        );
        stochastic_adaptive_d_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<StochasticAdaptiveDOutput, StochasticAdaptiveDError> {
        let input = StochasticAdaptiveDInput::from_slices(
            high,
            low,
            close,
            StochasticAdaptiveDParams {
                k_length: self.k_length,
                d_smoothing: self.d_smoothing,
                pre_smooth: self.pre_smooth,
                attenuation: self.attenuation,
            },
        );
        stochastic_adaptive_d_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<StochasticAdaptiveDStream, StochasticAdaptiveDError> {
        StochasticAdaptiveDStream::try_new(StochasticAdaptiveDParams {
            k_length: self.k_length,
            d_smoothing: self.d_smoothing,
            pre_smooth: self.pre_smooth,
            attenuation: self.attenuation,
        })
    }
}

#[derive(Debug, Error)]
pub enum StochasticAdaptiveDError {
    #[error("stochastic_adaptive_d: Empty input data.")]
    EmptyInputData,
    #[error("stochastic_adaptive_d: Input length mismatch: high={high}, low={low}, close={close}")]
    DataLengthMismatch {
        high: usize,
        low: usize,
        close: usize,
    },
    #[error("stochastic_adaptive_d: All input values are invalid.")]
    AllValuesNaN,
    #[error("stochastic_adaptive_d: Invalid stochastic length: k_length = {k_length}, data length = {data_len}")]
    InvalidKLength { k_length: usize, data_len: usize },
    #[error("stochastic_adaptive_d: Invalid d_smoothing: d_smoothing = {d_smoothing}, data length = {data_len}")]
    InvalidDSmoothing { d_smoothing: usize, data_len: usize },
    #[error("stochastic_adaptive_d: Invalid pre_smooth: pre_smooth = {pre_smooth}, data length = {data_len}")]
    InvalidPreSmooth { pre_smooth: usize, data_len: usize },
    #[error("stochastic_adaptive_d: Invalid attenuation: {attenuation}")]
    InvalidAttenuation { attenuation: f64 },
    #[error("stochastic_adaptive_d: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stochastic_adaptive_d: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("stochastic_adaptive_d: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("stochastic_adaptive_d: Invalid float range: start={start}, end={end}, step={step}")]
    InvalidFloatRange { start: f64, end: f64, step: f64 },
    #[error("stochastic_adaptive_d: Invalid kernel for batch: {0:?}")]
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
        Kernel::Auto => Kernel::Scalar,
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    }
}

#[inline(always)]
fn validate_lengths(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(), StochasticAdaptiveDError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(StochasticAdaptiveDError::EmptyInputData);
    }
    if high.len() != low.len() || low.len() != close.len() {
        return Err(StochasticAdaptiveDError::DataLengthMismatch {
            high: high.len(),
            low: low.len(),
            close: close.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn validate_params(
    k_length: usize,
    d_smoothing: usize,
    pre_smooth: usize,
    attenuation: f64,
    len: usize,
) -> Result<(), StochasticAdaptiveDError> {
    if k_length == 0 || k_length > len {
        return Err(StochasticAdaptiveDError::InvalidKLength {
            k_length,
            data_len: len,
        });
    }
    if d_smoothing == 0 || d_smoothing > len {
        return Err(StochasticAdaptiveDError::InvalidDSmoothing {
            d_smoothing,
            data_len: len,
        });
    }
    if pre_smooth == 0 || pre_smooth > len {
        return Err(StochasticAdaptiveDError::InvalidPreSmooth {
            pre_smooth,
            data_len: len,
        });
    }
    if !attenuation.is_finite() || attenuation < 0.1 {
        return Err(StochasticAdaptiveDError::InvalidAttenuation { attenuation });
    }
    Ok(())
}

#[inline(always)]
fn compute_warmup(
    first_valid: usize,
    k_length: usize,
    d_smoothing: usize,
    pre_smooth: usize,
) -> usize {
    first_valid
        .saturating_add(pre_smooth.saturating_sub(1))
        .saturating_add(k_length.saturating_sub(1))
        .saturating_add(d_smoothing.saturating_sub(1))
}

#[derive(Clone, Debug)]
struct RollingSma {
    period: usize,
    sum: f64,
    buf: Vec<f64>,
    head: usize,
    count: usize,
}

impl RollingSma {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            sum: 0.0,
            buf: vec![0.0; period.max(1)],
            head: 0,
            count: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.sum = 0.0;
        self.head = 0;
        self.count = 0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.count < self.period {
            self.buf[self.head] = value;
            self.sum += value;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
            self.count += 1;
            if self.count == self.period {
                Some(self.sum / self.period as f64)
            } else {
                None
            }
        } else {
            let old = self.buf[self.head];
            self.buf[self.head] = value;
            self.sum += value;
            self.sum -= old;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
            Some(self.sum / self.period as f64)
        }
    }
}

#[derive(Clone, Debug)]
struct RollingExtrema {
    period: usize,
    index: usize,
    maxq: VecDeque<(usize, f64)>,
    minq: VecDeque<(usize, f64)>,
}

impl RollingExtrema {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            index: 0,
            maxq: VecDeque::with_capacity(period),
            minq: VecDeque::with_capacity(period),
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.index = 0;
        self.maxq.clear();
        self.minq.clear();
    }

    #[inline]
    fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        let idx = self.index;
        self.index += 1;

        while let Some(&(_, value)) = self.maxq.back() {
            if value <= high {
                self.maxq.pop_back();
            } else {
                break;
            }
        }
        self.maxq.push_back((idx, high));

        while let Some(&(_, value)) = self.minq.back() {
            if value >= low {
                self.minq.pop_back();
            } else {
                break;
            }
        }
        self.minq.push_back((idx, low));

        let window_start = idx.saturating_add(1).saturating_sub(self.period);
        while let Some(&(front_idx, _)) = self.maxq.front() {
            if front_idx < window_start {
                self.maxq.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(front_idx, _)) = self.minq.front() {
            if front_idx < window_start {
                self.minq.pop_front();
            } else {
                break;
            }
        }

        if idx + 1 >= self.period {
            Some((
                self.maxq.front().map(|(_, value)| *value).unwrap_or(high),
                self.minq.front().map(|(_, value)| *value).unwrap_or(low),
            ))
        } else {
            None
        }
    }
}

#[inline(always)]
fn compute_stochastic_raw(close: f64, highest: f64, lowest: f64) -> f64 {
    let range = highest - lowest;
    if range.abs() <= EPS {
        CENTER
    } else {
        (close - lowest).mul_add(SCALE_100 / range, 0.0)
    }
}

#[inline(always)]
fn compute_ama(prev: f64, standard_d: f64, attenuation: f64) -> f64 {
    let alpha = ((standard_d - CENTER).abs() / SCALE_100) / attenuation;
    let src_ama = (standard_d - CENTER) / attenuation + CENTER;
    prev + alpha * (src_ama - prev)
}

fn stochastic_adaptive_d_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: &StochasticAdaptiveDParams,
    out_standard_d: &mut [f64],
    out_adaptive_d: &mut [f64],
    out_difference: &mut [f64],
) {
    let k_length = params.k_length.unwrap_or(DEFAULT_K_LENGTH);
    let d_smoothing = params.d_smoothing.unwrap_or(DEFAULT_D_SMOOTHING);
    let pre_smooth = params.pre_smooth.unwrap_or(DEFAULT_PRE_SMOOTH);
    let attenuation = params.attenuation.unwrap_or(DEFAULT_ATTENUATION);

    let mut pre_high = RollingSma::new(pre_smooth);
    let mut pre_low = RollingSma::new(pre_smooth);
    let mut pre_close = RollingSma::new(pre_smooth);
    let mut stoch_window = RollingExtrema::new(k_length);
    let mut d_sma = RollingSma::new(d_smoothing);
    let mut adaptive = CENTER;

    for i in 0..close.len() {
        let h = high[i];
        let l = low[i];
        let c = close[i];

        if !valid_bar(h, l, c) {
            out_standard_d[i] = f64::NAN;
            out_adaptive_d[i] = f64::NAN;
            out_difference[i] = f64::NAN;
            pre_high.reset();
            pre_low.reset();
            pre_close.reset();
            stoch_window.reset();
            d_sma.reset();
            adaptive = CENTER;
            continue;
        }

        let Some(s_high) = pre_high.update(h) else {
            out_standard_d[i] = f64::NAN;
            out_adaptive_d[i] = f64::NAN;
            out_difference[i] = f64::NAN;
            continue;
        };
        let Some(s_low) = pre_low.update(l) else {
            out_standard_d[i] = f64::NAN;
            out_adaptive_d[i] = f64::NAN;
            out_difference[i] = f64::NAN;
            continue;
        };
        let Some(s_close) = pre_close.update(c) else {
            out_standard_d[i] = f64::NAN;
            out_adaptive_d[i] = f64::NAN;
            out_difference[i] = f64::NAN;
            continue;
        };
        let Some((highest, lowest)) = stoch_window.update(s_high, s_low) else {
            out_standard_d[i] = f64::NAN;
            out_adaptive_d[i] = f64::NAN;
            out_difference[i] = f64::NAN;
            continue;
        };
        let stoch_raw = compute_stochastic_raw(s_close, highest, lowest);
        let Some(stoch_d_raw) = d_sma.update(stoch_raw) else {
            out_standard_d[i] = f64::NAN;
            out_adaptive_d[i] = f64::NAN;
            out_difference[i] = f64::NAN;
            continue;
        };

        let standard_d = CENTER + (stoch_d_raw - CENTER) * 0.5;
        adaptive = compute_ama(adaptive, standard_d, attenuation);
        let difference = CENTER + (standard_d - adaptive) * 2.0;
        out_standard_d[i] = standard_d;
        out_adaptive_d[i] = adaptive;
        out_difference[i] = difference;
    }
}

#[inline]
pub fn stochastic_adaptive_d(
    input: &StochasticAdaptiveDInput,
) -> Result<StochasticAdaptiveDOutput, StochasticAdaptiveDError> {
    stochastic_adaptive_d_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn stochastic_adaptive_d_with_kernel(
    input: &StochasticAdaptiveDInput,
    kernel: Kernel,
) -> Result<StochasticAdaptiveDOutput, StochasticAdaptiveDError> {
    let (high, low, close) = input.as_refs();
    validate_lengths(high, low, close)?;
    let len = close.len();
    let k_length = input.get_k_length();
    let d_smoothing = input.get_d_smoothing();
    let pre_smooth = input.get_pre_smooth();
    let attenuation = input.get_attenuation();
    validate_params(k_length, d_smoothing, pre_smooth, attenuation, len)?;
    let first_valid =
        first_valid_bar(high, low, close).ok_or(StochasticAdaptiveDError::AllValuesNaN)?;
    if len - first_valid < pre_smooth + k_length + d_smoothing - 2 {
        return Err(StochasticAdaptiveDError::NotEnoughValidData {
            needed: pre_smooth + k_length + d_smoothing - 2,
            valid: len - first_valid,
        });
    }

    let _kernel = normalize_kernel(kernel);
    let mut standard_d = alloc_uninit_f64(len);
    let mut adaptive_d = alloc_uninit_f64(len);
    let mut difference = alloc_uninit_f64(len);
    stochastic_adaptive_d_compute_into(
        high,
        low,
        close,
        &input.params,
        &mut standard_d,
        &mut adaptive_d,
        &mut difference,
    );
    Ok(StochasticAdaptiveDOutput {
        standard_d,
        adaptive_d,
        difference,
    })
}

#[inline]
pub fn stochastic_adaptive_d_into_slice(
    out_standard_d: &mut [f64],
    out_adaptive_d: &mut [f64],
    out_difference: &mut [f64],
    input: &StochasticAdaptiveDInput,
    kernel: Kernel,
) -> Result<(), StochasticAdaptiveDError> {
    let (high, low, close) = input.as_refs();
    validate_lengths(high, low, close)?;
    let len = close.len();
    if out_standard_d.len() != len || out_adaptive_d.len() != len || out_difference.len() != len {
        return Err(StochasticAdaptiveDError::OutputLengthMismatch {
            expected: len,
            got: out_standard_d
                .len()
                .max(out_adaptive_d.len())
                .max(out_difference.len()),
        });
    }
    let k_length = input.get_k_length();
    let d_smoothing = input.get_d_smoothing();
    let pre_smooth = input.get_pre_smooth();
    let attenuation = input.get_attenuation();
    validate_params(k_length, d_smoothing, pre_smooth, attenuation, len)?;
    let first_valid =
        first_valid_bar(high, low, close).ok_or(StochasticAdaptiveDError::AllValuesNaN)?;
    if len - first_valid < pre_smooth + k_length + d_smoothing - 2 {
        return Err(StochasticAdaptiveDError::NotEnoughValidData {
            needed: pre_smooth + k_length + d_smoothing - 2,
            valid: len - first_valid,
        });
    }

    let _kernel = normalize_kernel(kernel);
    stochastic_adaptive_d_compute_into(
        high,
        low,
        close,
        &input.params,
        out_standard_d,
        out_adaptive_d,
        out_difference,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn stochastic_adaptive_d_into(
    input: &StochasticAdaptiveDInput,
    out_standard_d: &mut [f64],
    out_adaptive_d: &mut [f64],
    out_difference: &mut [f64],
) -> Result<(), StochasticAdaptiveDError> {
    stochastic_adaptive_d_into_slice(
        out_standard_d,
        out_adaptive_d,
        out_difference,
        input,
        Kernel::Auto,
    )
}

#[derive(Clone, Debug)]
pub struct StochasticAdaptiveDStream {
    pre_high: RollingSma,
    pre_low: RollingSma,
    pre_close: RollingSma,
    stoch_window: RollingExtrema,
    d_sma: RollingSma,
    attenuation: f64,
    adaptive: f64,
}

impl StochasticAdaptiveDStream {
    #[inline]
    pub fn try_new(params: StochasticAdaptiveDParams) -> Result<Self, StochasticAdaptiveDError> {
        let k_length = params.k_length.unwrap_or(DEFAULT_K_LENGTH);
        let d_smoothing = params.d_smoothing.unwrap_or(DEFAULT_D_SMOOTHING);
        let pre_smooth = params.pre_smooth.unwrap_or(DEFAULT_PRE_SMOOTH);
        let attenuation = params.attenuation.unwrap_or(DEFAULT_ATTENUATION);
        validate_params(k_length, d_smoothing, pre_smooth, attenuation, usize::MAX)?;
        Ok(Self {
            pre_high: RollingSma::new(pre_smooth),
            pre_low: RollingSma::new(pre_smooth),
            pre_close: RollingSma::new(pre_smooth),
            stoch_window: RollingExtrema::new(k_length),
            d_sma: RollingSma::new(d_smoothing),
            attenuation,
            adaptive: CENTER,
        })
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64)> {
        if !valid_bar(high, low, close) {
            self.pre_high.reset();
            self.pre_low.reset();
            self.pre_close.reset();
            self.stoch_window.reset();
            self.d_sma.reset();
            self.adaptive = CENTER;
            return None;
        }

        let s_high = self.pre_high.update(high)?;
        let s_low = self.pre_low.update(low)?;
        let s_close = self.pre_close.update(close)?;
        let (highest, lowest) = self.stoch_window.update(s_high, s_low)?;
        let stoch_raw = compute_stochastic_raw(s_close, highest, lowest);
        let stoch_d_raw = self.d_sma.update(stoch_raw)?;
        let standard_d = CENTER + (stoch_d_raw - CENTER) * 0.5;
        self.adaptive = compute_ama(self.adaptive, standard_d, self.attenuation);
        let difference = CENTER + (standard_d - self.adaptive) * 2.0;
        Some((standard_d, self.adaptive, difference))
    }
}

#[derive(Clone, Debug)]
pub struct StochasticAdaptiveDBatchRange {
    pub k_length: (usize, usize, usize),
    pub d_smoothing: (usize, usize, usize),
    pub pre_smooth: (usize, usize, usize),
    pub attenuation: (f64, f64, f64),
}

impl Default for StochasticAdaptiveDBatchRange {
    fn default() -> Self {
        Self {
            k_length: (DEFAULT_K_LENGTH, DEFAULT_K_LENGTH, 0),
            d_smoothing: (DEFAULT_D_SMOOTHING, DEFAULT_D_SMOOTHING, 0),
            pre_smooth: (DEFAULT_PRE_SMOOTH, DEFAULT_PRE_SMOOTH, 0),
            attenuation: (DEFAULT_ATTENUATION, DEFAULT_ATTENUATION, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StochasticAdaptiveDBatchOutput {
    pub standard_d: Vec<f64>,
    pub adaptive_d: Vec<f64>,
    pub difference: Vec<f64>,
    pub combos: Vec<StochasticAdaptiveDParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct StochasticAdaptiveDBatchBuilder {
    range: StochasticAdaptiveDBatchRange,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for StochasticAdaptiveDBatchBuilder {
    fn default() -> Self {
        Self {
            range: StochasticAdaptiveDBatchRange::default(),
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StochasticAdaptiveDBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn k_length_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.k_length = range;
        self
    }

    #[inline]
    pub fn d_smoothing_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.d_smoothing = range;
        self
    }

    #[inline]
    pub fn pre_smooth_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.pre_smooth = range;
        self
    }

    #[inline]
    pub fn attenuation_range(mut self, range: (f64, f64, f64)) -> Self {
        self.range.attenuation = range;
        self
    }

    #[inline]
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = Some(value.into());
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
    ) -> Result<StochasticAdaptiveDBatchOutput, StochasticAdaptiveDError> {
        stochastic_adaptive_d_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<StochasticAdaptiveDBatchOutput, StochasticAdaptiveDError> {
        stochastic_adaptive_d_batch_with_kernel(
            &candles.high,
            &candles.low,
            source_type(candles, self.source.as_deref().unwrap_or("close")),
            &self.range,
            self.kernel,
        )
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, StochasticAdaptiveDError> {
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
            if x < step {
                break;
            }
            x -= step;
        }
    }
    if out.is_empty() {
        return Err(StochasticAdaptiveDError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, StochasticAdaptiveDError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(StochasticAdaptiveDError::InvalidFloatRange { start, end, step });
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
        return Err(StochasticAdaptiveDError::InvalidFloatRange { start, end, step });
    }
    Ok(out)
}

pub fn expand_grid_stochastic_adaptive_d(
    range: &StochasticAdaptiveDBatchRange,
) -> Result<Vec<StochasticAdaptiveDParams>, StochasticAdaptiveDError> {
    let k_lengths = axis_usize(range.k_length)?;
    let d_smoothings = axis_usize(range.d_smoothing)?;
    let pre_smooths = axis_usize(range.pre_smooth)?;
    let attenuations = axis_f64(range.attenuation)?;
    let cap = k_lengths
        .len()
        .checked_mul(d_smoothings.len())
        .and_then(|value| value.checked_mul(pre_smooths.len()))
        .and_then(|value| value.checked_mul(attenuations.len()))
        .ok_or(StochasticAdaptiveDError::InvalidRange {
            start: range.k_length.0.to_string(),
            end: range.k_length.1.to_string(),
            step: range.k_length.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &k_length in &k_lengths {
        for &d_smoothing in &d_smoothings {
            for &pre_smooth in &pre_smooths {
                for &attenuation in &attenuations {
                    out.push(StochasticAdaptiveDParams {
                        k_length: Some(k_length),
                        d_smoothing: Some(d_smoothing),
                        pre_smooth: Some(pre_smooth),
                        attenuation: Some(attenuation),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn stochastic_adaptive_d_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochasticAdaptiveDBatchRange,
    kernel: Kernel,
) -> Result<StochasticAdaptiveDBatchOutput, StochasticAdaptiveDError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(StochasticAdaptiveDError::InvalidKernelForBatch(other)),
    };
    stochastic_adaptive_d_batch_par_slice(high, low, close, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn stochastic_adaptive_d_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochasticAdaptiveDBatchRange,
    kernel: Kernel,
) -> Result<StochasticAdaptiveDBatchOutput, StochasticAdaptiveDError> {
    stochastic_adaptive_d_batch_inner(high, low, close, sweep, kernel, false)
}

#[inline]
pub fn stochastic_adaptive_d_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochasticAdaptiveDBatchRange,
    kernel: Kernel,
) -> Result<StochasticAdaptiveDBatchOutput, StochasticAdaptiveDError> {
    stochastic_adaptive_d_batch_inner(high, low, close, sweep, kernel, true)
}

fn stochastic_adaptive_d_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochasticAdaptiveDBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<StochasticAdaptiveDBatchOutput, StochasticAdaptiveDError> {
    validate_lengths(high, low, close)?;
    let cols = close.len();
    let combos = expand_grid_stochastic_adaptive_d(sweep)?;
    for params in &combos {
        validate_params(
            params.k_length.unwrap_or(DEFAULT_K_LENGTH),
            params.d_smoothing.unwrap_or(DEFAULT_D_SMOOTHING),
            params.pre_smooth.unwrap_or(DEFAULT_PRE_SMOOTH),
            params.attenuation.unwrap_or(DEFAULT_ATTENUATION),
            cols,
        )?;
    }
    let first_valid =
        first_valid_bar(high, low, close).ok_or(StochasticAdaptiveDError::AllValuesNaN)?;
    let rows = combos.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(StochasticAdaptiveDError::OutputLengthMismatch {
            expected: usize::MAX,
            got: 0,
        })?;
    let _kernel = kernel;

    let mut standard_matrix = make_uninit_matrix(rows, cols);
    let mut adaptive_matrix = make_uninit_matrix(rows, cols);
    let mut difference_matrix = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            compute_warmup(
                first_valid,
                params.k_length.unwrap_or(DEFAULT_K_LENGTH),
                params.d_smoothing.unwrap_or(DEFAULT_D_SMOOTHING),
                params.pre_smooth.unwrap_or(DEFAULT_PRE_SMOOTH),
            )
        })
        .collect();
    init_matrix_prefixes(&mut standard_matrix, cols, &warmups);
    init_matrix_prefixes(&mut adaptive_matrix, cols, &warmups);
    init_matrix_prefixes(&mut difference_matrix, cols, &warmups);

    let mut standard_guard = ManuallyDrop::new(standard_matrix);
    let mut adaptive_guard = ManuallyDrop::new(adaptive_matrix);
    let mut difference_guard = ManuallyDrop::new(difference_matrix);
    let standard_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(standard_guard.as_mut_ptr(), standard_guard.len())
    };
    let adaptive_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(adaptive_guard.as_mut_ptr(), adaptive_guard.len())
    };
    let difference_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(difference_guard.as_mut_ptr(), difference_guard.len())
    };

    let do_row = |row: usize,
                  standard_row: &mut [MaybeUninit<f64>],
                  adaptive_row: &mut [MaybeUninit<f64>],
                  difference_row: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let dst_standard =
            unsafe { std::slice::from_raw_parts_mut(standard_row.as_mut_ptr() as *mut f64, cols) };
        let dst_adaptive =
            unsafe { std::slice::from_raw_parts_mut(adaptive_row.as_mut_ptr() as *mut f64, cols) };
        let dst_difference = unsafe {
            std::slice::from_raw_parts_mut(difference_row.as_mut_ptr() as *mut f64, cols)
        };
        stochastic_adaptive_d_compute_into(
            high,
            low,
            close,
            params,
            dst_standard,
            dst_adaptive,
            dst_difference,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        standard_mu
            .par_chunks_mut(cols)
            .zip(adaptive_mu.par_chunks_mut(cols))
            .zip(difference_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((standard_row, adaptive_row), difference_row))| {
                do_row(row, standard_row, adaptive_row, difference_row)
            });

        #[cfg(target_arch = "wasm32")]
        for (row, ((standard_row, adaptive_row), difference_row)) in standard_mu
            .chunks_mut(cols)
            .zip(adaptive_mu.chunks_mut(cols))
            .zip(difference_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, standard_row, adaptive_row, difference_row);
        }
    } else {
        for (row, ((standard_row, adaptive_row), difference_row)) in standard_mu
            .chunks_mut(cols)
            .zip(adaptive_mu.chunks_mut(cols))
            .zip(difference_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, standard_row, adaptive_row, difference_row);
        }
    }

    let standard_d = unsafe {
        Vec::from_raw_parts(
            standard_guard.as_mut_ptr() as *mut f64,
            total,
            standard_guard.capacity(),
        )
    };
    let adaptive_d = unsafe {
        Vec::from_raw_parts(
            adaptive_guard.as_mut_ptr() as *mut f64,
            total,
            adaptive_guard.capacity(),
        )
    };
    let difference = unsafe {
        Vec::from_raw_parts(
            difference_guard.as_mut_ptr() as *mut f64,
            total,
            difference_guard.capacity(),
        )
    };

    Ok(StochasticAdaptiveDBatchOutput {
        standard_d,
        adaptive_d,
        difference,
        combos,
        rows,
        cols,
    })
}

fn stochastic_adaptive_d_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochasticAdaptiveDBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_standard_d: &mut [f64],
    out_adaptive_d: &mut [f64],
    out_difference: &mut [f64],
) -> Result<Vec<StochasticAdaptiveDParams>, StochasticAdaptiveDError> {
    validate_lengths(high, low, close)?;
    let cols = close.len();
    let combos = expand_grid_stochastic_adaptive_d(sweep)?;
    for params in &combos {
        validate_params(
            params.k_length.unwrap_or(DEFAULT_K_LENGTH),
            params.d_smoothing.unwrap_or(DEFAULT_D_SMOOTHING),
            params.pre_smooth.unwrap_or(DEFAULT_PRE_SMOOTH),
            params.attenuation.unwrap_or(DEFAULT_ATTENUATION),
            cols,
        )?;
    }
    let rows = combos.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(StochasticAdaptiveDError::OutputLengthMismatch {
            expected: usize::MAX,
            got: 0,
        })?;
    if out_standard_d.len() != total
        || out_adaptive_d.len() != total
        || out_difference.len() != total
    {
        return Err(StochasticAdaptiveDError::OutputLengthMismatch {
            expected: total,
            got: out_standard_d
                .len()
                .max(out_adaptive_d.len())
                .max(out_difference.len()),
        });
    }
    let _kernel = kernel;

    let do_row = |row: usize,
                  standard_row: &mut [f64],
                  adaptive_row: &mut [f64],
                  difference_row: &mut [f64]| {
        stochastic_adaptive_d_compute_into(
            high,
            low,
            close,
            &combos[row],
            standard_row,
            adaptive_row,
            difference_row,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_standard_d
            .par_chunks_mut(cols)
            .zip(out_adaptive_d.par_chunks_mut(cols))
            .zip(out_difference.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((standard_row, adaptive_row), difference_row))| {
                do_row(row, standard_row, adaptive_row, difference_row)
            });

        #[cfg(target_arch = "wasm32")]
        for (row, ((standard_row, adaptive_row), difference_row)) in out_standard_d
            .chunks_mut(cols)
            .zip(out_adaptive_d.chunks_mut(cols))
            .zip(out_difference.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, standard_row, adaptive_row, difference_row);
        }
    } else {
        for (row, ((standard_row, adaptive_row), difference_row)) in out_standard_d
            .chunks_mut(cols)
            .zip(out_adaptive_d.chunks_mut(cols))
            .zip(out_difference.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, standard_row, adaptive_row, difference_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "stochastic_adaptive_d")]
#[pyo3(signature = (high, low, close, k_length=DEFAULT_K_LENGTH, d_smoothing=DEFAULT_D_SMOOTHING, pre_smooth=DEFAULT_PRE_SMOOTH, attenuation=DEFAULT_ATTENUATION, kernel=None))]
pub fn stochastic_adaptive_d_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    k_length: usize,
    d_smoothing: usize,
    pre_smooth: usize,
    attenuation: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let input = StochasticAdaptiveDInput::from_slices(
        high,
        low,
        close,
        StochasticAdaptiveDParams {
            k_length: Some(k_length),
            d_smoothing: Some(d_smoothing),
            pre_smooth: Some(pre_smooth),
            attenuation: Some(attenuation),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| stochastic_adaptive_d_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.standard_d.into_pyarray(py),
        out.adaptive_d.into_pyarray(py),
        out.difference.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "StochasticAdaptiveDStream")]
pub struct StochasticAdaptiveDStreamPy {
    stream: StochasticAdaptiveDStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StochasticAdaptiveDStreamPy {
    #[new]
    #[pyo3(signature = (k_length=DEFAULT_K_LENGTH, d_smoothing=DEFAULT_D_SMOOTHING, pre_smooth=DEFAULT_PRE_SMOOTH, attenuation=DEFAULT_ATTENUATION))]
    fn new(
        k_length: usize,
        d_smoothing: usize,
        pre_smooth: usize,
        attenuation: f64,
    ) -> PyResult<Self> {
        let stream = StochasticAdaptiveDStream::try_new(StochasticAdaptiveDParams {
            k_length: Some(k_length),
            d_smoothing: Some(d_smoothing),
            pre_smooth: Some(pre_smooth),
            attenuation: Some(attenuation),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "stochastic_adaptive_d_batch")]
#[pyo3(signature = (high, low, close, k_length_range, d_smoothing_range, pre_smooth_range, attenuation_range, kernel=None))]
pub fn stochastic_adaptive_d_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    k_length_range: (usize, usize, usize),
    d_smoothing_range: (usize, usize, usize),
    pre_smooth_range: (usize, usize, usize),
    attenuation_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = StochasticAdaptiveDBatchRange {
        k_length: k_length_range,
        d_smoothing: d_smoothing_range,
        pre_smooth: pre_smooth_range,
        attenuation: attenuation_range,
    };
    let combos = expand_grid_stochastic_adaptive_d(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let standard_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let adaptive_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let difference_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_standard = unsafe { standard_arr.as_slice_mut()? };
    let out_adaptive = unsafe { adaptive_arr.as_slice_mut()? };
    let out_difference = unsafe { difference_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        stochastic_adaptive_d_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_standard,
            out_adaptive,
            out_difference,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let k_lengths: Vec<u64> = combos
        .iter()
        .map(|params| params.k_length.unwrap_or(DEFAULT_K_LENGTH) as u64)
        .collect();
    let d_smoothings: Vec<u64> = combos
        .iter()
        .map(|params| params.d_smoothing.unwrap_or(DEFAULT_D_SMOOTHING) as u64)
        .collect();
    let pre_smooths: Vec<u64> = combos
        .iter()
        .map(|params| params.pre_smooth.unwrap_or(DEFAULT_PRE_SMOOTH) as u64)
        .collect();
    let attenuations: Vec<f64> = combos
        .iter()
        .map(|params| params.attenuation.unwrap_or(DEFAULT_ATTENUATION))
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("standard_d", standard_arr.reshape((rows, cols))?)?;
    dict.set_item("adaptive_d", adaptive_arr.reshape((rows, cols))?)?;
    dict.set_item("difference", difference_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("k_lengths", k_lengths.into_pyarray(py))?;
    dict.set_item("d_smoothings", d_smoothings.into_pyarray(py))?;
    dict.set_item("pre_smooths", pre_smooths.into_pyarray(py))?;
    dict.set_item("attenuations", attenuations.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_stochastic_adaptive_d_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stochastic_adaptive_d_py, m)?)?;
    m.add_function(wrap_pyfunction!(stochastic_adaptive_d_batch_py, m)?)?;
    m.add_class::<StochasticAdaptiveDStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StochasticAdaptiveDJsOutput {
    standard_d: Vec<f64>,
    adaptive_d: Vec<f64>,
    difference: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StochasticAdaptiveDBatchConfig {
    k_length_range: Vec<usize>,
    d_smoothing_range: Vec<usize>,
    pre_smooth_range: Vec<usize>,
    attenuation_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StochasticAdaptiveDBatchJsOutput {
    standard_d: Vec<f64>,
    adaptive_d: Vec<f64>,
    difference: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<StochasticAdaptiveDParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "stochastic_adaptive_d")]
pub fn stochastic_adaptive_d_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k_length: usize,
    d_smoothing: usize,
    pre_smooth: usize,
    attenuation: f64,
) -> Result<JsValue, JsValue> {
    let input = StochasticAdaptiveDInput::from_slices(
        high,
        low,
        close,
        StochasticAdaptiveDParams {
            k_length: Some(k_length),
            d_smoothing: Some(d_smoothing),
            pre_smooth: Some(pre_smooth),
            attenuation: Some(attenuation),
        },
    );
    let out = stochastic_adaptive_d(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&StochasticAdaptiveDJsOutput {
        standard_d: out.standard_d,
        adaptive_d: out.adaptive_d,
        difference: out.difference,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_adaptive_d_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    k_length: usize,
    d_smoothing: usize,
    pre_smooth: usize,
    attenuation: f64,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to stochastic_adaptive_d_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 3);
        let (out_standard, rest) = out.split_at_mut(len);
        let (out_adaptive, out_difference) = rest.split_at_mut(len);
        let input = StochasticAdaptiveDInput::from_slices(
            high,
            low,
            close,
            StochasticAdaptiveDParams {
                k_length: Some(k_length),
                d_smoothing: Some(d_smoothing),
                pre_smooth: Some(pre_smooth),
                attenuation: Some(attenuation),
            },
        );
        stochastic_adaptive_d_into_slice(
            out_standard,
            out_adaptive,
            out_difference,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "stochastic_adaptive_d_into_host")]
pub fn stochastic_adaptive_d_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out_ptr: *mut f64,
    k_length: usize,
    d_smoothing: usize,
    pre_smooth: usize,
    attenuation: f64,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to stochastic_adaptive_d_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, close.len() * 3);
        let (out_standard, rest) = out.split_at_mut(close.len());
        let (out_adaptive, out_difference) = rest.split_at_mut(close.len());
        let input = StochasticAdaptiveDInput::from_slices(
            high,
            low,
            close,
            StochasticAdaptiveDParams {
                k_length: Some(k_length),
                d_smoothing: Some(d_smoothing),
                pre_smooth: Some(pre_smooth),
                attenuation: Some(attenuation),
            },
        );
        stochastic_adaptive_d_into_slice(
            out_standard,
            out_adaptive,
            out_difference,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_adaptive_d_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len * 3];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_adaptive_d_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 3);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "stochastic_adaptive_d_batch")]
pub fn stochastic_adaptive_d_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: StochasticAdaptiveDBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.k_length_range.len() != 3
        || config.d_smoothing_range.len() != 3
        || config.pre_smooth_range.len() != 3
        || config.attenuation_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = StochasticAdaptiveDBatchRange {
        k_length: (
            config.k_length_range[0],
            config.k_length_range[1],
            config.k_length_range[2],
        ),
        d_smoothing: (
            config.d_smoothing_range[0],
            config.d_smoothing_range[1],
            config.d_smoothing_range[2],
        ),
        pre_smooth: (
            config.pre_smooth_range[0],
            config.pre_smooth_range[1],
            config.pre_smooth_range[2],
        ),
        attenuation: (
            config.attenuation_range[0],
            config.attenuation_range[1],
            config.attenuation_range[2],
        ),
    };
    let batch = stochastic_adaptive_d_batch_slice(high, low, close, &sweep, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&StochasticAdaptiveDBatchJsOutput {
        standard_d: batch.standard_d,
        adaptive_d: batch.adaptive_d,
        difference: batch.difference,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_adaptive_d_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    standard_ptr: *mut f64,
    adaptive_ptr: *mut f64,
    difference_ptr: *mut f64,
    len: usize,
    k_length_start: usize,
    k_length_end: usize,
    k_length_step: usize,
    d_smoothing_start: usize,
    d_smoothing_end: usize,
    d_smoothing_step: usize,
    pre_smooth_start: usize,
    pre_smooth_end: usize,
    pre_smooth_step: usize,
    attenuation_start: f64,
    attenuation_end: f64,
    attenuation_step: f64,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || standard_ptr.is_null()
        || adaptive_ptr.is_null()
        || difference_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to stochastic_adaptive_d_batch_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = StochasticAdaptiveDBatchRange {
            k_length: (k_length_start, k_length_end, k_length_step),
            d_smoothing: (d_smoothing_start, d_smoothing_end, d_smoothing_step),
            pre_smooth: (pre_smooth_start, pre_smooth_end, pre_smooth_step),
            attenuation: (attenuation_start, attenuation_end, attenuation_step),
        };
        let combos = expand_grid_stochastic_adaptive_d(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out_standard = std::slice::from_raw_parts_mut(standard_ptr, total);
        let out_adaptive = std::slice::from_raw_parts_mut(adaptive_ptr, total);
        let out_difference = std::slice::from_raw_parts_mut(difference_ptr, total);
        stochastic_adaptive_d_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            Kernel::Scalar,
            false,
            out_standard,
            out_adaptive,
            out_difference,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_adaptive_d_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k_length: usize,
    d_smoothing: usize,
    pre_smooth: usize,
    attenuation: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = stochastic_adaptive_d_js(
        high,
        low,
        close,
        k_length,
        d_smoothing,
        pre_smooth,
        attenuation,
    )?;
    crate::write_wasm_object_f64_outputs("stochastic_adaptive_d_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stochastic_adaptive_d_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = stochastic_adaptive_d_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "stochastic_adaptive_d_batch_output_into_js",
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
        for (idx, (&lhs, &rhs)) in a.iter().zip(b.iter()).enumerate() {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(
                    lhs.is_nan() && rhs.is_nan(),
                    "nan mismatch at {idx}: {lhs} vs {rhs}"
                );
            } else {
                assert!(
                    (lhs - rhs).abs() <= tol,
                    "mismatch at {idx}: {lhs} vs {rhs} with tol {tol}"
                );
            }
        }
    }

    fn sample_hlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + i as f64 * 0.11 + (i as f64 * 0.07).sin() * 1.7;
            let spread = 1.2 + (i as f64 * 0.05).cos().abs() * 1.1;
            let c = base + (i as f64 * 0.13).sin() * 0.9;
            high.push(base + spread);
            low.push(base - spread);
            close.push(c);
        }
        (high, low, close)
    }

    #[test]
    fn stochastic_adaptive_d_output_contract() {
        let (high, low, close) = sample_hlc(320);
        let input = StochasticAdaptiveDInput::from_slices(
            &high,
            &low,
            &close,
            StochasticAdaptiveDParams::default(),
        );
        let out = stochastic_adaptive_d(&input).expect("indicator");
        assert_eq!(out.standard_d.len(), close.len());
        assert_eq!(out.adaptive_d.len(), close.len());
        assert_eq!(out.difference.len(), close.len());
        assert!(out.standard_d.iter().any(|v| v.is_finite()));
        assert!(out.adaptive_d.iter().any(|v| v.is_finite()));
        assert!(out.difference.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn stochastic_adaptive_d_into_matches_api() {
        let (high, low, close) = sample_hlc(240);
        let input = StochasticAdaptiveDInput::from_slices(
            &high,
            &low,
            &close,
            StochasticAdaptiveDParams {
                k_length: Some(18),
                d_smoothing: Some(7),
                pre_smooth: Some(16),
                attenuation: Some(1.7),
            },
        );
        let baseline = stochastic_adaptive_d(&input).expect("baseline");
        let mut standard_d = vec![0.0; close.len()];
        let mut adaptive_d = vec![0.0; close.len()];
        let mut difference = vec![0.0; close.len()];
        stochastic_adaptive_d_into_slice(
            &mut standard_d,
            &mut adaptive_d,
            &mut difference,
            &input,
            Kernel::Scalar,
        )
        .expect("into");
        assert_close(&baseline.standard_d, &standard_d, 1e-12);
        assert_close(&baseline.adaptive_d, &adaptive_d, 1e-12);
        assert_close(&baseline.difference, &difference, 1e-12);
    }

    #[test]
    fn stochastic_adaptive_d_stream_matches_batch() {
        let (high, low, close) = sample_hlc(260);
        let params = StochasticAdaptiveDParams {
            k_length: Some(14),
            d_smoothing: Some(5),
            pre_smooth: Some(10),
            attenuation: Some(1.6),
        };
        let input = StochasticAdaptiveDInput::from_slices(&high, &low, &close, params.clone());
        let batch = stochastic_adaptive_d(&input).expect("batch");
        let mut stream = StochasticAdaptiveDStream::try_new(params).expect("stream");
        let mut standard_d = vec![f64::NAN; close.len()];
        let mut adaptive_d = vec![f64::NAN; close.len()];
        let mut difference = vec![f64::NAN; close.len()];
        for i in 0..close.len() {
            if let Some((standard, adaptive, diff)) = stream.update(high[i], low[i], close[i]) {
                standard_d[i] = standard;
                adaptive_d[i] = adaptive;
                difference[i] = diff;
            }
        }
        assert_close(&batch.standard_d, &standard_d, 1e-12);
        assert_close(&batch.adaptive_d, &adaptive_d, 1e-12);
        assert_close(&batch.difference, &difference, 1e-12);
    }

    #[test]
    fn stochastic_adaptive_d_batch_single_param_matches_single() {
        let (high, low, close) = sample_hlc(220);
        let sweep = StochasticAdaptiveDBatchRange {
            k_length: (20, 20, 0),
            d_smoothing: (9, 9, 0),
            pre_smooth: (20, 20, 0),
            attenuation: (2.0, 2.0, 0.0),
        };
        let batch = stochastic_adaptive_d_batch_with_kernel(
            &high,
            &low,
            &close,
            &sweep,
            Kernel::ScalarBatch,
        )
        .expect("batch");
        let single = stochastic_adaptive_d(&StochasticAdaptiveDInput::from_slices(
            &high,
            &low,
            &close,
            StochasticAdaptiveDParams::default(),
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_close(&batch.standard_d[..close.len()], &single.standard_d, 1e-12);
        assert_close(&batch.adaptive_d[..close.len()], &single.adaptive_d, 1e-12);
        assert_close(&batch.difference[..close.len()], &single.difference, 1e-12);
    }

    #[test]
    fn stochastic_adaptive_d_rejects_invalid_attenuation() {
        let (high, low, close) = sample_hlc(128);
        let input = StochasticAdaptiveDInput::from_slices(
            &high,
            &low,
            &close,
            StochasticAdaptiveDParams {
                k_length: Some(20),
                d_smoothing: Some(9),
                pre_smooth: Some(20),
                attenuation: Some(0.0),
            },
        );
        let err = stochastic_adaptive_d(&input).expect_err("invalid");
        assert!(matches!(
            err,
            StochasticAdaptiveDError::InvalidAttenuation { .. }
        ));
    }

    #[test]
    fn stochastic_adaptive_d_dispatch_matches_direct() {
        let (high, low, close) = sample_hlc(240);
        let combo = [
            ParamKV {
                key: "k_length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "d_smoothing",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "pre_smooth",
                value: ParamValue::Int(10),
            },
            ParamKV {
                key: "attenuation",
                value: ParamValue::Float(1.6),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "stochastic_adaptive_d",
            data: IndicatorDataRef::Ohlc {
                open: &close,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            output_id: Some("adaptive_d"),
            kernel: Kernel::ScalarBatch,
        };
        let batch = compute_cpu_batch(req).expect("dispatch");
        let direct = stochastic_adaptive_d(&StochasticAdaptiveDInput::from_slices(
            &high,
            &low,
            &close,
            StochasticAdaptiveDParams {
                k_length: Some(14),
                d_smoothing: Some(5),
                pre_smooth: Some(10),
                attenuation: Some(1.6),
            },
        ))
        .expect("direct");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        let row = &batch.values_f64.as_ref().expect("f64 output")[0..close.len()];
        assert_close(row, &direct.adaptive_d, 1e-12);
    }
}
