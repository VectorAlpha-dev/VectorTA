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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const ZERO_RANGE_DIVISOR: f64 = 9_999_999.0;

#[derive(Debug, Clone)]
pub enum TwiggsMoneyFlowData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct TwiggsMoneyFlowOutput {
    pub tmf: Vec<f64>,
    pub smoothed: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct TwiggsMoneyFlowParams {
    pub length: Option<usize>,
    pub smoothing_length: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for TwiggsMoneyFlowParams {
    fn default() -> Self {
        Self {
            length: Some(21),
            smoothing_length: Some(14),
            ma_type: Some("WMA".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TwiggsMoneyFlowInput<'a> {
    pub data: TwiggsMoneyFlowData<'a>,
    pub params: TwiggsMoneyFlowParams,
}

impl<'a> TwiggsMoneyFlowInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: TwiggsMoneyFlowParams) -> Self {
        Self {
            data: TwiggsMoneyFlowData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: TwiggsMoneyFlowParams,
    ) -> Self {
        Self {
            data: TwiggsMoneyFlowData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, TwiggsMoneyFlowParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(21)
    }

    #[inline]
    pub fn get_smoothing_length(&self) -> usize {
        self.params.smoothing_length.unwrap_or(14)
    }

    #[inline]
    pub fn ma_type_str(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or("WMA")
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            TwiggsMoneyFlowData::Candles { candles } => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                candles.volume.as_slice(),
            ),
            TwiggsMoneyFlowData::Slices {
                high,
                low,
                close,
                volume,
            } => (*high, *low, *close, *volume),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TwiggsMoneyFlowBuilder {
    length: Option<usize>,
    smoothing_length: Option<usize>,
    ma_type: Option<&'static str>,
    kernel: Kernel,
}

impl Default for TwiggsMoneyFlowBuilder {
    fn default() -> Self {
        Self {
            length: None,
            smoothing_length: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TwiggsMoneyFlowBuilder {
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
    pub fn smoothing_length(mut self, value: usize) -> Self {
        self.smoothing_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn ma_type(mut self, value: &'static str) -> Self {
        self.ma_type = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<TwiggsMoneyFlowOutput, TwiggsMoneyFlowError> {
        let params = TwiggsMoneyFlowParams {
            length: self.length,
            smoothing_length: self.smoothing_length,
            ma_type: self.ma_type.map(|x| x.to_string()),
        };
        let input = TwiggsMoneyFlowInput::from_candles(candles, params);
        twiggs_money_flow_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<TwiggsMoneyFlowOutput, TwiggsMoneyFlowError> {
        let params = TwiggsMoneyFlowParams {
            length: self.length,
            smoothing_length: self.smoothing_length,
            ma_type: self.ma_type.map(|x| x.to_string()),
        };
        let input = TwiggsMoneyFlowInput::from_slices(high, low, close, volume, params);
        twiggs_money_flow_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<TwiggsMoneyFlowStream, TwiggsMoneyFlowError> {
        let params = TwiggsMoneyFlowParams {
            length: self.length,
            smoothing_length: self.smoothing_length,
            ma_type: self.ma_type.map(|x| x.to_string()),
        };
        TwiggsMoneyFlowStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum TwiggsMoneyFlowError {
    #[error("twiggs_money_flow: Empty input data.")]
    EmptyInputData,
    #[error("twiggs_money_flow: Data length mismatch across high, low, close, and volume.")]
    DataLengthMismatch,
    #[error("twiggs_money_flow: All OHLCV values are invalid.")]
    AllValuesNaN,
    #[error("twiggs_money_flow: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("twiggs_money_flow: Invalid smoothing length: smoothing_length = {smoothing_length}, data length = {data_len}")]
    InvalidSmoothingLength {
        smoothing_length: usize,
        data_len: usize,
    },
    #[error("twiggs_money_flow: Invalid MA type: {ma_type}")]
    InvalidMaType { ma_type: String },
    #[error("twiggs_money_flow: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("twiggs_money_flow: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("twiggs_money_flow: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("twiggs_money_flow: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("twiggs_money_flow: Invalid input: {0}")]
    InvalidInput(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwiggsMaType {
    Sma,
    Ema,
    Wma,
    Vwma,
}

#[inline(always)]
fn parse_ma_type(value: &str) -> Result<TwiggsMaType, TwiggsMoneyFlowError> {
    match value {
        "SMA" => return Ok(TwiggsMaType::Sma),
        "EMA" => return Ok(TwiggsMaType::Ema),
        "WMA" => return Ok(TwiggsMaType::Wma),
        "VWMA" => return Ok(TwiggsMaType::Vwma),
        _ => {}
    }
    let normalized = value.trim().to_ascii_uppercase();
    match normalized.as_str() {
        "SMA" => Ok(TwiggsMaType::Sma),
        "EMA" => Ok(TwiggsMaType::Ema),
        "WMA" => Ok(TwiggsMaType::Wma),
        "VWMA" => Ok(TwiggsMaType::Vwma),
        _ => Err(TwiggsMoneyFlowError::InvalidMaType {
            ma_type: value.into(),
        }),
    }
}

#[inline(always)]
fn normalize_ma_type(value: &str) -> Result<String, TwiggsMoneyFlowError> {
    Ok(match parse_ma_type(value)? {
        TwiggsMaType::Sma => "SMA".to_string(),
        TwiggsMaType::Ema => "EMA".to_string(),
        TwiggsMaType::Wma => "WMA".to_string(),
        TwiggsMaType::Vwma => "VWMA".to_string(),
    })
}

#[inline(always)]
fn is_valid_bar(high: f64, low: f64, close: f64, volume: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && volume.is_finite() && high >= low
}

#[inline(always)]
fn first_valid_adv(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Option<usize> {
    if close.len() < 2 {
        return None;
    }
    for i in 1..close.len() {
        if is_valid_bar(high[i], low[i], close[i], volume[i]) && close[i - 1].is_finite() {
            return Some(i);
        }
    }
    None
}

#[inline(always)]
fn tmf_warmup(length: usize, ma_type: TwiggsMaType, first_adv: usize) -> usize {
    match ma_type {
        TwiggsMaType::Ema => first_adv,
        TwiggsMaType::Sma | TwiggsMaType::Wma | TwiggsMaType::Vwma => {
            first_adv + length.saturating_sub(1)
        }
    }
}

#[inline(always)]
fn smoothed_warmup(tmf_first: usize, smoothing_length: usize) -> usize {
    if smoothing_length <= 1 {
        tmf_first
    } else {
        let sqrt_len = (smoothing_length as f64).sqrt().floor() as usize;
        tmf_first + smoothing_length + sqrt_len.saturating_sub(2)
    }
}

#[derive(Clone, Debug)]
struct SmaState {
    window: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
}

impl SmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            window: vec![0.0; period],
            head: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        let period = self.window.len();
        if self.count < period {
            self.window[self.head] = value;
            self.head = (self.head + 1) % period;
            self.count += 1;
            self.sum += value;
            if self.count == period {
                self.sum / period as f64
            } else {
                f64::NAN
            }
        } else {
            let old = self.window[self.head];
            self.window[self.head] = value;
            self.head = (self.head + 1) % period;
            self.sum += value - old;
            self.sum / period as f64
        }
    }
}

#[derive(Clone, Debug)]
struct EmaState {
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl EmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            value: f64::NAN,
            initialized: false,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        if !self.initialized {
            self.value = value;
            self.initialized = true;
        } else {
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value;
        }
        self.value
    }
}

#[derive(Clone, Debug)]
struct WmaState {
    window: Vec<f64>,
    head: usize,
    count: usize,
    plain_sum: f64,
    weighted_sum: f64,
    divisor: f64,
}

impl WmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            window: vec![0.0; period],
            head: 0,
            count: 0,
            plain_sum: 0.0,
            weighted_sum: 0.0,
            divisor: (period * (period + 1) / 2) as f64,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        let period = self.window.len();
        if self.count < period {
            self.window[self.head] = value;
            self.head = (self.head + 1) % period;
            self.count += 1;
            self.plain_sum += value;
            self.weighted_sum += value * self.count as f64;
            if self.count == period {
                self.weighted_sum / self.divisor
            } else {
                f64::NAN
            }
        } else {
            let old = self.window[self.head];
            let prev_plain = self.plain_sum;
            self.window[self.head] = value;
            self.head = (self.head + 1) % period;
            self.plain_sum = prev_plain - old + value;
            self.weighted_sum = self.weighted_sum - prev_plain + value * period as f64;
            self.weighted_sum / self.divisor
        }
    }
}

#[derive(Clone, Debug)]
struct WmaFixed<const N: usize> {
    window: [f64; N],
    head: usize,
    count: usize,
    plain_sum: f64,
    weighted_sum: f64,
    divisor: f64,
}

impl<const N: usize> WmaFixed<N> {
    #[inline]
    fn new() -> Self {
        Self {
            window: [0.0; N],
            head: 0,
            count: 0,
            plain_sum: 0.0,
            weighted_sum: 0.0,
            divisor: (N * (N + 1) / 2) as f64,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        if self.count < N {
            self.window[self.head] = value;
            self.head += 1;
            if self.head == N {
                self.head = 0;
            }
            self.count += 1;
            self.plain_sum += value;
            self.weighted_sum += value * self.count as f64;
            if self.count == N {
                self.weighted_sum / self.divisor
            } else {
                f64::NAN
            }
        } else {
            let old = self.window[self.head];
            let prev_plain = self.plain_sum;
            self.window[self.head] = value;
            self.head += 1;
            if self.head == N {
                self.head = 0;
            }
            self.plain_sum = prev_plain - old + value;
            self.weighted_sum = self.weighted_sum - prev_plain + value * N as f64;
            self.weighted_sum / self.divisor
        }
    }
}

#[derive(Clone, Debug)]
struct VwmaState {
    values: Vec<f64>,
    weights: Vec<f64>,
    head: usize,
    count: usize,
    num_sum: f64,
    den_sum: f64,
}

impl VwmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            values: vec![0.0; period],
            weights: vec![0.0; period],
            head: 0,
            count: 0,
            num_sum: 0.0,
            den_sum: 0.0,
        }
    }

    #[inline]
    fn update(&mut self, value: f64, weight: f64) -> f64 {
        let period = self.values.len();
        if self.count < period {
            self.values[self.head] = value;
            self.weights[self.head] = weight;
            self.head = (self.head + 1) % period;
            self.count += 1;
            self.num_sum += value * weight;
            self.den_sum += weight;
        } else {
            let old_value = self.values[self.head];
            let old_weight = self.weights[self.head];
            self.values[self.head] = value;
            self.weights[self.head] = weight;
            self.head = (self.head + 1) % period;
            self.num_sum += value * weight - old_value * old_weight;
            self.den_sum += weight - old_weight;
        }

        if self.count == period && self.den_sum != 0.0 {
            self.num_sum / self.den_sum
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Debug)]
enum BaseMaState {
    Sma(SmaState),
    Ema(EmaState),
    Wma(WmaState),
    Vwma(VwmaState),
}

impl BaseMaState {
    #[inline]
    fn new(kind: TwiggsMaType, period: usize) -> Self {
        match kind {
            TwiggsMaType::Sma => Self::Sma(SmaState::new(period)),
            TwiggsMaType::Ema => Self::Ema(EmaState::new(period)),
            TwiggsMaType::Wma => Self::Wma(WmaState::new(period)),
            TwiggsMaType::Vwma => Self::Vwma(VwmaState::new(period)),
        }
    }

    #[inline]
    fn update(&mut self, value: f64, weight: f64) -> f64 {
        match self {
            BaseMaState::Sma(state) => state.update(value),
            BaseMaState::Ema(state) => state.update(value),
            BaseMaState::Wma(state) => state.update(value),
            BaseMaState::Vwma(state) => state.update(value, weight),
        }
    }
}

#[derive(Clone, Debug)]
struct HmaState {
    passthrough: bool,
    half: WmaState,
    full: WmaState,
    sqrt: WmaState,
}

impl HmaState {
    #[inline]
    fn new(period: usize) -> Self {
        let passthrough = period <= 1;
        let full_period = period.max(1);
        let half_period = (period / 2).max(1);
        let sqrt_period = (period as f64).sqrt().floor() as usize;
        Self {
            passthrough,
            half: WmaState::new(half_period),
            full: WmaState::new(full_period),
            sqrt: WmaState::new(sqrt_period.max(1)),
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        if self.passthrough {
            return value;
        }
        let half = self.half.update(value);
        let full = self.full.update(value);
        if !half.is_finite() || !full.is_finite() {
            return f64::NAN;
        }
        self.sqrt.update(2.0 * half - full)
    }
}

#[derive(Clone, Debug)]
struct TwiggsMoneyFlowState {
    adv_ma: BaseMaState,
    vol_ma: BaseMaState,
    smoother: HmaState,
    prev_close: Option<f64>,
}

impl TwiggsMoneyFlowState {
    #[inline]
    fn new(length: usize, smoothing_length: usize, ma_type: TwiggsMaType) -> Self {
        Self {
            adv_ma: BaseMaState::new(ma_type, length),
            vol_ma: BaseMaState::new(ma_type, length),
            smoother: HmaState::new(smoothing_length),
            prev_close: None,
        }
    }

    #[inline]
    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> (f64, f64) {
        if !is_valid_bar(high, low, close, volume) {
            self.prev_close = if close.is_finite() { Some(close) } else { None };
            return (f64::NAN, f64::NAN);
        }

        let prev_close = match self.prev_close {
            Some(prev) if prev.is_finite() => prev,
            _ => {
                self.prev_close = Some(close);
                return (f64::NAN, f64::NAN);
            }
        };

        let tr_h = prev_close.max(high);
        let tr_l = prev_close.min(low);
        let tr_c = tr_h - tr_l;
        let denom = if tr_c == 0.0 {
            ZERO_RANGE_DIVISOR
        } else {
            tr_c
        };
        let adv = volume * (((close - tr_l) - (tr_h - close)) / denom);

        let wm_v = self.vol_ma.update(volume, volume);
        let wm_a = self.adv_ma.update(adv, volume);
        let tmf = if wm_v == 0.0 {
            0.0
        } else if wm_v.is_finite() && wm_a.is_finite() {
            wm_a / wm_v
        } else {
            f64::NAN
        };
        let smoothed = if tmf.is_finite() {
            self.smoother.update(tmf)
        } else {
            f64::NAN
        };

        self.prev_close = Some(close);
        (tmf, smoothed)
    }
}

#[derive(Clone, Debug)]
struct TwiggsMoneyFlowDefaultWmaState {
    adv_ma: WmaFixed<21>,
    vol_ma: WmaFixed<21>,
    half: WmaFixed<7>,
    full: WmaFixed<14>,
    sqrt: WmaFixed<3>,
    prev_close: Option<f64>,
}

impl TwiggsMoneyFlowDefaultWmaState {
    #[inline]
    fn new() -> Self {
        Self {
            adv_ma: WmaFixed::new(),
            vol_ma: WmaFixed::new(),
            half: WmaFixed::new(),
            full: WmaFixed::new(),
            sqrt: WmaFixed::new(),
            prev_close: None,
        }
    }

    #[inline]
    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> (f64, f64) {
        if !is_valid_bar(high, low, close, volume) {
            self.prev_close = if close.is_finite() { Some(close) } else { None };
            return (f64::NAN, f64::NAN);
        }

        let prev_close = match self.prev_close {
            Some(prev) if prev.is_finite() => prev,
            _ => {
                self.prev_close = Some(close);
                return (f64::NAN, f64::NAN);
            }
        };

        let tr_h = prev_close.max(high);
        let tr_l = prev_close.min(low);
        let tr_c = tr_h - tr_l;
        let denom = if tr_c == 0.0 {
            ZERO_RANGE_DIVISOR
        } else {
            tr_c
        };
        let adv = volume * (((close - tr_l) - (tr_h - close)) / denom);

        let wm_v = self.vol_ma.update(volume);
        let wm_a = self.adv_ma.update(adv);
        let tmf = if wm_v == 0.0 {
            0.0
        } else if wm_v.is_finite() && wm_a.is_finite() {
            wm_a / wm_v
        } else {
            f64::NAN
        };
        let smoothed = if tmf.is_finite() {
            let half = self.half.update(tmf);
            let full = self.full.update(tmf);
            if !half.is_finite() || !full.is_finite() {
                f64::NAN
            } else {
                self.sqrt.update(2.0 * half - full)
            }
        } else {
            f64::NAN
        };

        self.prev_close = Some(close);
        (tmf, smoothed)
    }
}

#[inline(always)]
fn twiggs_money_flow_default_wma_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out_tmf: &mut [f64],
    out_smoothed: &mut [f64],
) {
    let mut state = TwiggsMoneyFlowDefaultWmaState::new();
    for i in 0..close.len() {
        let (tmf, smoothed) = state.update(high[i], low[i], close[i], volume[i]);
        out_tmf[i] = tmf;
        out_smoothed[i] = smoothed;
    }
}

#[inline(always)]
fn twiggs_money_flow_all_valid(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> bool {
    for i in 0..close.len() {
        if !is_valid_bar(high[i], low[i], close[i], volume[i]) {
            return false;
        }
    }
    true
}

#[inline(always)]
fn twiggs_money_flow_default_wma_all_valid_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out_tmf: &mut [f64],
    out_smoothed: &mut [f64],
) {
    let len = close.len();
    if len == 0 {
        return;
    }

    let mut adv_ma = WmaFixed::<21>::new();
    let mut vol_ma = WmaFixed::<21>::new();
    let mut half = WmaFixed::<7>::new();
    let mut full = WmaFixed::<14>::new();
    let mut sqrt = WmaFixed::<3>::new();

    out_tmf[0] = f64::NAN;
    out_smoothed[0] = f64::NAN;
    let mut prev_close = close[0];

    for i in 1..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let v = volume[i];
        let tr_h = prev_close.max(h);
        let tr_l = prev_close.min(l);
        let tr_c = tr_h - tr_l;
        let denom = if tr_c == 0.0 {
            ZERO_RANGE_DIVISOR
        } else {
            tr_c
        };
        let adv = v * (((c - tr_l) - (tr_h - c)) / denom);

        let wm_v = vol_ma.update(v);
        let wm_a = adv_ma.update(adv);
        let tmf = if wm_v == 0.0 {
            0.0
        } else if wm_v.is_finite() && wm_a.is_finite() {
            wm_a / wm_v
        } else {
            f64::NAN
        };
        let smoothed = if tmf.is_finite() {
            let hma_half = half.update(tmf);
            let hma_full = full.update(tmf);
            if !hma_half.is_finite() || !hma_full.is_finite() {
                f64::NAN
            } else {
                sqrt.update(2.0 * hma_half - hma_full)
            }
        } else {
            f64::NAN
        };

        out_tmf[i] = tmf;
        out_smoothed[i] = smoothed;
        prev_close = c;
    }
}

#[inline(always)]
fn twiggs_money_flow_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    length: usize,
    smoothing_length: usize,
    ma_type: TwiggsMaType,
    out_tmf: &mut [f64],
    out_smoothed: &mut [f64],
) {
    if length == 21 && smoothing_length == 14 && ma_type == TwiggsMaType::Wma {
        if twiggs_money_flow_all_valid(high, low, close, volume) {
            twiggs_money_flow_default_wma_all_valid_compute_into(
                high,
                low,
                close,
                volume,
                out_tmf,
                out_smoothed,
            );
            return;
        }
        twiggs_money_flow_default_wma_compute_into(high, low, close, volume, out_tmf, out_smoothed);
        return;
    }

    let mut state = TwiggsMoneyFlowState::new(length, smoothing_length, ma_type);
    for i in 0..close.len() {
        let (tmf, smoothed) = state.update(high[i], low[i], close[i], volume[i]);
        out_tmf[i] = tmf;
        out_smoothed[i] = smoothed;
    }
}

#[inline(always)]
fn twiggs_money_flow_prepare<'a>(
    input: &'a TwiggsMoneyFlowInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        TwiggsMaType,
        usize,
        Kernel,
    ),
    TwiggsMoneyFlowError,
> {
    let (high, low, close, volume) = input.as_refs();
    let len = close.len();
    if len == 0 {
        return Err(TwiggsMoneyFlowError::EmptyInputData);
    }
    if high.len() != len || low.len() != len || volume.len() != len {
        return Err(TwiggsMoneyFlowError::DataLengthMismatch);
    }

    let length = input.get_length();
    let smoothing_length = input.get_smoothing_length();
    let ma_type = parse_ma_type(input.ma_type_str())?;

    if length == 0 || length > len {
        return Err(TwiggsMoneyFlowError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if smoothing_length > len {
        return Err(TwiggsMoneyFlowError::InvalidSmoothingLength {
            smoothing_length,
            data_len: len,
        });
    }

    let first_adv =
        first_valid_adv(high, low, close, volume).ok_or(TwiggsMoneyFlowError::AllValuesNaN)?;
    let needed = match ma_type {
        TwiggsMaType::Ema => 1,
        TwiggsMaType::Sma | TwiggsMaType::Wma | TwiggsMaType::Vwma => length,
    };
    let valid = len - first_adv;
    if valid < needed {
        return Err(TwiggsMoneyFlowError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };

    Ok((
        high,
        low,
        close,
        volume,
        length,
        smoothing_length,
        ma_type,
        first_adv,
        chosen,
    ))
}

#[inline]
pub fn twiggs_money_flow(
    input: &TwiggsMoneyFlowInput,
) -> Result<TwiggsMoneyFlowOutput, TwiggsMoneyFlowError> {
    twiggs_money_flow_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn twiggs_money_flow_with_kernel(
    input: &TwiggsMoneyFlowInput,
    kernel: Kernel,
) -> Result<TwiggsMoneyFlowOutput, TwiggsMoneyFlowError> {
    let (high, low, close, volume, length, smoothing_length, ma_type, _first_adv, _chosen) =
        twiggs_money_flow_prepare(input, kernel)?;
    let mut tmf = alloc_uninit_f64(close.len());
    let mut smoothed = alloc_uninit_f64(close.len());
    twiggs_money_flow_compute_into(
        high,
        low,
        close,
        volume,
        length,
        smoothing_length,
        ma_type,
        &mut tmf,
        &mut smoothed,
    );
    Ok(TwiggsMoneyFlowOutput { tmf, smoothed })
}

#[inline]
pub fn twiggs_money_flow_into_slice(
    dst_tmf: &mut [f64],
    dst_smoothed: &mut [f64],
    input: &TwiggsMoneyFlowInput,
    kernel: Kernel,
) -> Result<(), TwiggsMoneyFlowError> {
    let (high, low, close, volume, length, smoothing_length, ma_type, _first_adv, _chosen) =
        twiggs_money_flow_prepare(input, kernel)?;
    if dst_tmf.len() != close.len() || dst_smoothed.len() != close.len() {
        return Err(TwiggsMoneyFlowError::OutputLengthMismatch {
            expected: close.len(),
            got: dst_tmf.len().max(dst_smoothed.len()),
        });
    }
    twiggs_money_flow_compute_into(
        high,
        low,
        close,
        volume,
        length,
        smoothing_length,
        ma_type,
        dst_tmf,
        dst_smoothed,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn twiggs_money_flow_into(
    input: &TwiggsMoneyFlowInput,
    out_tmf: &mut [f64],
    out_smoothed: &mut [f64],
) -> Result<(), TwiggsMoneyFlowError> {
    twiggs_money_flow_into_slice(out_tmf, out_smoothed, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct TwiggsMoneyFlowStream {
    state: TwiggsMoneyFlowState,
}

impl TwiggsMoneyFlowStream {
    pub fn try_new(params: TwiggsMoneyFlowParams) -> Result<Self, TwiggsMoneyFlowError> {
        let length = params.length.unwrap_or(21);
        let smoothing_length = params.smoothing_length.unwrap_or(14);
        let ma_type = parse_ma_type(params.ma_type.as_deref().unwrap_or("WMA"))?;
        if length == 0 {
            return Err(TwiggsMoneyFlowError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        Ok(Self {
            state: TwiggsMoneyFlowState::new(length, smoothing_length, ma_type),
        })
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<(f64, f64)> {
        let out = self.state.update(high, low, close, volume);
        if out.0.is_finite() || out.1.is_finite() {
            Some(out)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct TwiggsMoneyFlowBatchRange {
    pub length: (usize, usize, usize),
    pub smoothing_length: (usize, usize, usize),
    pub ma_type: String,
}

impl Default for TwiggsMoneyFlowBatchRange {
    fn default() -> Self {
        Self {
            length: (21, 63, 1),
            smoothing_length: (14, 14, 0),
            ma_type: "WMA".to_string(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TwiggsMoneyFlowBatchBuilder {
    range: TwiggsMoneyFlowBatchRange,
    kernel: Kernel,
}

impl TwiggsMoneyFlowBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn smoothing_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing_length = (start, end, step);
        self
    }

    #[inline]
    pub fn ma_type<S: Into<String>>(mut self, value: S) -> Self {
        self.range.ma_type = value.into();
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<TwiggsMoneyFlowBatchOutput, TwiggsMoneyFlowError> {
        twiggs_money_flow_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<TwiggsMoneyFlowBatchOutput, TwiggsMoneyFlowError> {
        twiggs_money_flow_batch_with_kernel(
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
            source_type(candles, "volume"),
            &self.range,
            self.kernel,
        )
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct TwiggsMoneyFlowBatchConfig {
    pub length_range: Vec<usize>,
    pub smoothing_length_range: Vec<usize>,
    pub ma_type: String,
}

#[derive(Clone, Debug)]
pub struct TwiggsMoneyFlowBatchOutput {
    pub tmf: Vec<f64>,
    pub smoothed: Vec<f64>,
    pub combos: Vec<TwiggsMoneyFlowParams>,
    pub rows: usize,
    pub cols: usize,
}

impl TwiggsMoneyFlowBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &TwiggsMoneyFlowParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(21) == params.length.unwrap_or(21)
                && combo.smoothing_length.unwrap_or(14) == params.smoothing_length.unwrap_or(14)
                && combo.ma_type.as_deref().unwrap_or("WMA")
                    == params.ma_type.as_deref().unwrap_or("WMA")
        })
    }

    #[inline]
    pub fn tmf_for(&self, params: &TwiggsMoneyFlowParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.tmf.get(start..start + self.cols)
        })
    }

    #[inline]
    pub fn smoothed_for(&self, params: &TwiggsMoneyFlowParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.smoothed.get(start..start + self.cols)
        })
    }
}

#[inline]
pub fn expand_grid_twiggs_money_flow(
    range: &TwiggsMoneyFlowBatchRange,
) -> Result<Vec<TwiggsMoneyFlowParams>, TwiggsMoneyFlowError> {
    let ma_type = normalize_ma_type(&range.ma_type)?;

    fn axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, TwiggsMoneyFlowError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start <= end {
            let mut x = start;
            while x <= end {
                out.push(x);
                match x.checked_add(step.max(1)) {
                    Some(next) if next > x => x = next,
                    _ => break,
                }
            }
        } else {
            let mut x = start;
            while x >= end {
                out.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(step.max(1));
                if next == x || next < end {
                    break;
                }
                x = next;
            }
        }
        if out.is_empty() {
            return Err(TwiggsMoneyFlowError::InvalidRange { start, end, step });
        }
        Ok(out)
    }

    let lengths = axis(range.length)?;
    let smoothing_lengths = axis(range.smoothing_length)?;
    let mut combos = Vec::with_capacity(lengths.len().saturating_mul(smoothing_lengths.len()));
    for length in lengths {
        for smoothing_length in smoothing_lengths.iter().copied() {
            combos.push(TwiggsMoneyFlowParams {
                length: Some(length),
                smoothing_length: Some(smoothing_length),
                ma_type: Some(ma_type.clone()),
            });
        }
    }
    Ok(combos)
}

#[inline]
pub fn twiggs_money_flow_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &TwiggsMoneyFlowBatchRange,
    kernel: Kernel,
) -> Result<TwiggsMoneyFlowBatchOutput, TwiggsMoneyFlowError> {
    let batch = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(TwiggsMoneyFlowError::InvalidKernelForBatch(other)),
    };
    twiggs_money_flow_batch_par_slice(high, low, close, volume, sweep, batch.to_non_batch())
}

#[inline(always)]
pub fn twiggs_money_flow_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &TwiggsMoneyFlowBatchRange,
    kernel: Kernel,
) -> Result<TwiggsMoneyFlowBatchOutput, TwiggsMoneyFlowError> {
    twiggs_money_flow_batch_inner(high, low, close, volume, sweep, kernel, false)
}

#[inline(always)]
pub fn twiggs_money_flow_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &TwiggsMoneyFlowBatchRange,
    kernel: Kernel,
) -> Result<TwiggsMoneyFlowBatchOutput, TwiggsMoneyFlowError> {
    twiggs_money_flow_batch_inner(high, low, close, volume, sweep, kernel, true)
}

#[inline(always)]
fn twiggs_money_flow_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &TwiggsMoneyFlowBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<TwiggsMoneyFlowBatchOutput, TwiggsMoneyFlowError> {
    let combos = expand_grid_twiggs_money_flow(sweep)?;
    let len = close.len();
    if len == 0 {
        return Err(TwiggsMoneyFlowError::EmptyInputData);
    }
    if high.len() != len || low.len() != len || volume.len() != len {
        return Err(TwiggsMoneyFlowError::DataLengthMismatch);
    }

    let first_adv =
        first_valid_adv(high, low, close, volume).ok_or(TwiggsMoneyFlowError::AllValuesNaN)?;
    let max_needed = combos
        .iter()
        .map(
            |p| match parse_ma_type(p.ma_type.as_deref().unwrap_or("WMA")).unwrap() {
                TwiggsMaType::Ema => 1,
                TwiggsMaType::Sma | TwiggsMaType::Wma | TwiggsMaType::Vwma => {
                    p.length.unwrap_or(21)
                }
            },
        )
        .max()
        .unwrap_or(0);
    let valid = len - first_adv;
    if valid < max_needed {
        return Err(TwiggsMoneyFlowError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let rows = combos.len();
    let cols = len;
    let mut tmf_mu = make_uninit_matrix(rows, cols);
    let mut smoothed_mu = make_uninit_matrix(rows, cols);
    let tmf_warmups: Vec<usize> = combos
        .iter()
        .map(|p| {
            tmf_warmup(
                p.length.unwrap_or(21),
                parse_ma_type(p.ma_type.as_deref().unwrap_or("WMA")).unwrap(),
                first_adv,
            )
        })
        .collect();
    let smoothed_warmups: Vec<usize> = combos
        .iter()
        .zip(tmf_warmups.iter().copied())
        .map(|(p, tmf_first)| smoothed_warmup(tmf_first, p.smoothing_length.unwrap_or(14)))
        .collect();
    init_matrix_prefixes(&mut tmf_mu, cols, &tmf_warmups);
    init_matrix_prefixes(&mut smoothed_mu, cols, &smoothed_warmups);

    let mut tmf_guard = ManuallyDrop::new(tmf_mu);
    let mut smoothed_guard = ManuallyDrop::new(smoothed_mu);
    let tmf = unsafe {
        std::slice::from_raw_parts_mut(tmf_guard.as_mut_ptr() as *mut f64, tmf_guard.len())
    };
    let smoothed = unsafe {
        std::slice::from_raw_parts_mut(
            smoothed_guard.as_mut_ptr() as *mut f64,
            smoothed_guard.len(),
        )
    };

    twiggs_money_flow_batch_inner_into(
        high,
        low,
        close,
        volume,
        sweep,
        Kernel::Scalar,
        parallel,
        tmf,
        smoothed,
    )?;

    let tmf_values = unsafe {
        Vec::from_raw_parts(
            tmf_guard.as_mut_ptr() as *mut f64,
            tmf_guard.len(),
            tmf_guard.capacity(),
        )
    };
    let smoothed_values = unsafe {
        Vec::from_raw_parts(
            smoothed_guard.as_mut_ptr() as *mut f64,
            smoothed_guard.len(),
            smoothed_guard.capacity(),
        )
    };

    Ok(TwiggsMoneyFlowBatchOutput {
        tmf: tmf_values,
        smoothed: smoothed_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn twiggs_money_flow_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &TwiggsMoneyFlowBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_tmf: &mut [f64],
    out_smoothed: &mut [f64],
) -> Result<Vec<TwiggsMoneyFlowParams>, TwiggsMoneyFlowError> {
    let combos = expand_grid_twiggs_money_flow(sweep)?;
    let len = close.len();
    if len == 0 {
        return Err(TwiggsMoneyFlowError::EmptyInputData);
    }
    if high.len() != len || low.len() != len || volume.len() != len {
        return Err(TwiggsMoneyFlowError::DataLengthMismatch);
    }

    let first_adv =
        first_valid_adv(high, low, close, volume).ok_or(TwiggsMoneyFlowError::AllValuesNaN)?;
    let max_needed = combos
        .iter()
        .map(
            |p| match parse_ma_type(p.ma_type.as_deref().unwrap_or("WMA")).unwrap() {
                TwiggsMaType::Ema => 1,
                TwiggsMaType::Sma | TwiggsMaType::Wma | TwiggsMaType::Vwma => {
                    p.length.unwrap_or(21)
                }
            },
        )
        .max()
        .unwrap_or(0);
    let valid = len - first_adv;
    if valid < max_needed {
        return Err(TwiggsMoneyFlowError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let rows = combos.len();
    let cols = len;
    let total = rows
        .checked_mul(cols)
        .ok_or(TwiggsMoneyFlowError::InvalidInput("rows*cols overflow"))?;
    if out_tmf.len() != total || out_smoothed.len() != total {
        return Err(TwiggsMoneyFlowError::OutputLengthMismatch {
            expected: total,
            got: out_tmf.len().max(out_smoothed.len()),
        });
    }

    unsafe {
        let out_tmf_mu =
            std::slice::from_raw_parts_mut(out_tmf.as_mut_ptr() as *mut MaybeUninit<f64>, total);
        let out_smoothed_mu = std::slice::from_raw_parts_mut(
            out_smoothed.as_mut_ptr() as *mut MaybeUninit<f64>,
            total,
        );
        let tmf_warmups: Vec<usize> = combos
            .iter()
            .map(|p| {
                tmf_warmup(
                    p.length.unwrap_or(21),
                    parse_ma_type(p.ma_type.as_deref().unwrap_or("WMA")).unwrap(),
                    first_adv,
                )
            })
            .collect();
        let smoothed_warmups: Vec<usize> = combos
            .iter()
            .zip(tmf_warmups.iter().copied())
            .map(|(p, tmf_first)| smoothed_warmup(tmf_first, p.smoothing_length.unwrap_or(14)))
            .collect();
        init_matrix_prefixes(out_tmf_mu, cols, &tmf_warmups);
        init_matrix_prefixes(out_smoothed_mu, cols, &smoothed_warmups);
    }

    let do_row = |row: usize, tmf_row: &mut [f64], smoothed_row: &mut [f64]| {
        let params = &combos[row];
        twiggs_money_flow_compute_into(
            high,
            low,
            close,
            volume,
            params.length.unwrap_or(21),
            params.smoothing_length.unwrap_or(14),
            parse_ma_type(params.ma_type.as_deref().unwrap_or("WMA")).unwrap(),
            tmf_row,
            smoothed_row,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_tmf
                .par_chunks_mut(cols)
                .zip(out_smoothed.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (tmf_row, smoothed_row))| do_row(row, tmf_row, smoothed_row));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (tmf_row, smoothed_row)) in out_tmf
                .chunks_mut(cols)
                .zip(out_smoothed.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, tmf_row, smoothed_row);
            }
        }
    } else {
        for (row, (tmf_row, smoothed_row)) in out_tmf
            .chunks_mut(cols)
            .zip(out_smoothed.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, tmf_row, smoothed_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "twiggs_money_flow")]
#[pyo3(signature = (high, low, close, volume, length=21, smoothing_length=14, ma_type="WMA", kernel=None))]
pub fn twiggs_money_flow_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length: usize,
    smoothing_length: usize,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let input = TwiggsMoneyFlowInput::from_slices(
        high,
        low,
        close,
        volume,
        TwiggsMoneyFlowParams {
            length: Some(length),
            smoothing_length: Some(smoothing_length),
            ma_type: Some(ma_type.to_string()),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| twiggs_money_flow_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.tmf.into_pyarray(py), out.smoothed.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "TwiggsMoneyFlowStream")]
pub struct TwiggsMoneyFlowStreamPy {
    stream: TwiggsMoneyFlowStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TwiggsMoneyFlowStreamPy {
    #[new]
    #[pyo3(signature = (length=21, smoothing_length=14, ma_type="WMA"))]
    fn new(length: usize, smoothing_length: usize, ma_type: &str) -> PyResult<Self> {
        let stream = TwiggsMoneyFlowStream::try_new(TwiggsMoneyFlowParams {
            length: Some(length),
            smoothing_length: Some(smoothing_length),
            ma_type: Some(ma_type.to_string()),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low, close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "twiggs_money_flow_batch")]
#[pyo3(signature = (high, low, close, volume, length_range, smoothing_length_range, ma_type="WMA", kernel=None))]
pub fn twiggs_money_flow_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    smoothing_length_range: (usize, usize, usize),
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let sweep = TwiggsMoneyFlowBatchRange {
        length: length_range,
        smoothing_length: smoothing_length_range,
        ma_type: ma_type.to_string(),
    };
    let combos =
        expand_grid_twiggs_money_flow(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let tmf_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let smoothed_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let tmf_out = unsafe { tmf_arr.as_slice_mut()? };
    let smoothed_out = unsafe { smoothed_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        twiggs_money_flow_batch_inner_into(
            high,
            low,
            close,
            volume,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            tmf_out,
            smoothed_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("tmf", tmf_arr.reshape((rows, cols))?)?;
    dict.set_item("smoothed", smoothed_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(21) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smoothing_lengths",
        combos
            .iter()
            .map(|p| p.smoothing_length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_types",
        combos
            .iter()
            .map(|p| p.ma_type.clone().unwrap_or_else(|| "WMA".to_string()))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_twiggs_money_flow_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(twiggs_money_flow_py, m)?)?;
    m.add_function(wrap_pyfunction!(twiggs_money_flow_batch_py, m)?)?;
    m.add_class::<TwiggsMoneyFlowStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "twiggs_money_flow_js")]
pub fn twiggs_money_flow_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    length: usize,
    smoothing_length: usize,
    ma_type: String,
) -> Result<JsValue, JsValue> {
    let input = TwiggsMoneyFlowInput::from_slices(
        high,
        low,
        close,
        volume,
        TwiggsMoneyFlowParams {
            length: Some(length),
            smoothing_length: Some(smoothing_length),
            ma_type: Some(ma_type),
        },
    );
    let mut tmf = vec![0.0; close.len()];
    let mut smoothed = vec![0.0; close.len()];
    twiggs_money_flow_into_slice(&mut tmf, &mut smoothed, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("tmf"),
        &serde_wasm_bindgen::to_value(&tmf).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("smoothed"),
        &serde_wasm_bindgen::to_value(&smoothed).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "twiggs_money_flow_batch_js")]
pub fn twiggs_money_flow_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: TwiggsMoneyFlowBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.smoothing_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = TwiggsMoneyFlowBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
        smoothing_length: (
            config.smoothing_length_range[0],
            config.smoothing_length_range[1],
            config.smoothing_length_range[2],
        ),
        ma_type: config.ma_type,
    };
    let combos =
        expand_grid_twiggs_money_flow(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    let mut tmf = vec![0.0; total];
    let mut smoothed = vec![0.0; total];
    twiggs_money_flow_batch_inner_into(
        high,
        low,
        close,
        volume,
        &sweep,
        Kernel::Scalar,
        false,
        &mut tmf,
        &mut smoothed,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("tmf"),
        &serde_wasm_bindgen::to_value(&tmf).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("smoothed"),
        &serde_wasm_bindgen::to_value(&smoothed).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn twiggs_money_flow_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn twiggs_money_flow_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn twiggs_money_flow_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    smoothing_length: usize,
    ma_type: String,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to twiggs_money_flow_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (tmf, smoothed) = out.split_at_mut(len);
        let input = TwiggsMoneyFlowInput::from_slices(
            high,
            low,
            close,
            volume,
            TwiggsMoneyFlowParams {
                length: Some(length),
                smoothing_length: Some(smoothing_length),
                ma_type: Some(ma_type),
            },
        );
        twiggs_money_flow_into_slice(tmf, smoothed, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "twiggs_money_flow_into_host")]
pub fn twiggs_money_flow_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out_ptr: *mut f64,
    length: usize,
    smoothing_length: usize,
    ma_type: String,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to twiggs_money_flow_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * close.len());
        let (tmf, smoothed) = out.split_at_mut(close.len());
        let input = TwiggsMoneyFlowInput::from_slices(
            high,
            low,
            close,
            volume,
            TwiggsMoneyFlowParams {
                length: Some(length),
                smoothing_length: Some(smoothing_length),
                ma_type: Some(ma_type),
            },
        );
        twiggs_money_flow_into_slice(tmf, smoothed, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn twiggs_money_flow_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    tmf_ptr: *mut f64,
    smoothed_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    smoothing_length_start: usize,
    smoothing_length_end: usize,
    smoothing_length_step: usize,
    ma_type: String,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || tmf_ptr.is_null()
        || smoothed_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to twiggs_money_flow_batch_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let sweep = TwiggsMoneyFlowBatchRange {
            length: (length_start, length_end, length_step),
            smoothing_length: (
                smoothing_length_start,
                smoothing_length_end,
                smoothing_length_step,
            ),
            ma_type,
        };
        let combos =
            expand_grid_twiggs_money_flow(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let tmf = std::slice::from_raw_parts_mut(tmf_ptr, total);
        let smoothed = std::slice::from_raw_parts_mut(smoothed_ptr, total);
        twiggs_money_flow_batch_inner_into(
            high,
            low,
            close,
            volume,
            &sweep,
            Kernel::Scalar,
            false,
            tmf,
            smoothed,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn twiggs_money_flow_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    length: usize,
    smoothing_length: usize,
    ma_type: String,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = twiggs_money_flow_js(high, low, close, volume, length, smoothing_length, ma_type)?;
    crate::write_wasm_object_f64_outputs("twiggs_money_flow_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn twiggs_money_flow_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = twiggs_money_flow_batch_js(high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "twiggs_money_flow_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series_close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(&x, &y)| {
                (x.is_nan() && y.is_nan())
                    || (x.is_finite() && y.is_finite() && (x - y).abs() <= tol)
            })
    }

    fn sample_hlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![
                10.4, 10.8, 11.0, 11.2, 11.5, 11.7, 11.6, 11.9, 12.1, 12.4, 12.6, 12.8, 12.7, 12.9,
                13.2, 13.4, 13.6, 13.5, 13.8, 14.0, 14.2, 14.4, 14.7, 14.9, 15.1, 15.0, 15.3, 15.5,
                15.8, 16.0, 16.2, 16.5, 16.7, 16.9, 17.1, 17.4, 17.6, 17.8, 18.0, 18.2,
            ],
            vec![
                9.8, 10.1, 10.4, 10.7, 10.9, 11.0, 11.1, 11.3, 11.6, 11.8, 12.0, 12.2, 12.1, 12.4,
                12.6, 12.8, 13.0, 13.1, 13.3, 13.5, 13.8, 14.0, 14.1, 14.3, 14.5, 14.6, 14.8, 15.0,
                15.2, 15.5, 15.7, 15.9, 16.1, 16.3, 16.6, 16.8, 17.0, 17.2, 17.4, 17.7,
            ],
            vec![
                10.1, 10.6, 10.8, 11.0, 11.3, 11.4, 11.5, 11.8, 11.9, 12.2, 12.4, 12.5, 12.4, 12.8,
                13.0, 13.1, 13.4, 13.3, 13.6, 13.8, 14.0, 14.2, 14.5, 14.6, 14.8, 14.9, 15.1, 15.3,
                15.6, 15.7, 16.0, 16.2, 16.4, 16.6, 16.9, 17.1, 17.3, 17.5, 17.7, 18.0,
            ],
            vec![
                1000.0, 1020.0, 1040.0, 1010.0, 1035.0, 1055.0, 1060.0, 1075.0, 1090.0, 1110.0,
                1085.0, 1100.0, 1125.0, 1140.0, 1160.0, 1180.0, 1170.0, 1195.0, 1210.0, 1230.0,
                1250.0, 1240.0, 1265.0, 1285.0, 1300.0, 1315.0, 1330.0, 1355.0, 1375.0, 1390.0,
                1410.0, 1430.0, 1450.0, 1475.0, 1490.0, 1510.0, 1530.0, 1550.0, 1575.0, 1590.0,
            ],
        )
    }

    fn naive_sma(data: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        for i in (period - 1)..data.len() {
            let window = &data[i + 1 - period..=i];
            if window.iter().all(|v| v.is_finite()) {
                out[i] = window.iter().sum::<f64>() / period as f64;
            }
        }
        out
    }

    fn naive_ema(data: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut state = f64::NAN;
        let mut initialized = false;
        for (i, &x) in data.iter().enumerate() {
            if !x.is_finite() {
                initialized = false;
                state = f64::NAN;
                continue;
            }
            if !initialized {
                state = x;
                initialized = true;
            } else {
                state = alpha * x + (1.0 - alpha) * state;
            }
            out[i] = state;
        }
        out
    }

    fn naive_wma(data: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        let denom = (period * (period + 1) / 2) as f64;
        for i in (period - 1)..data.len() {
            let window = &data[i + 1 - period..=i];
            if window.iter().all(|v| v.is_finite()) {
                let mut num = 0.0;
                for (j, &v) in window.iter().enumerate() {
                    num += (j + 1) as f64 * v;
                }
                out[i] = num / denom;
            }
        }
        out
    }

    fn naive_vwma(values: &[f64], weights: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; values.len()];
        for i in (period - 1)..values.len() {
            let vals = &values[i + 1 - period..=i];
            let wts = &weights[i + 1 - period..=i];
            if vals.iter().all(|v| v.is_finite()) && wts.iter().all(|v| v.is_finite()) {
                let num: f64 = vals.iter().zip(wts.iter()).map(|(v, w)| v * w).sum();
                let den: f64 = wts.iter().sum();
                if den != 0.0 {
                    out[i] = num / den;
                }
            }
        }
        out
    }

    fn naive_hma(data: &[f64], period: usize) -> Vec<f64> {
        if period <= 1 {
            return data.to_vec();
        }
        let half = (period / 2).max(1);
        let sqrt_len = (period as f64).sqrt().floor() as usize;
        let w_half = naive_wma(data, half);
        let w_full = naive_wma(data, period);
        let mut diff = vec![f64::NAN; data.len()];
        for i in 0..data.len() {
            if w_half[i].is_finite() && w_full[i].is_finite() {
                diff[i] = 2.0 * w_half[i] - w_full[i];
            }
        }
        naive_wma(&diff, sqrt_len.max(1))
    }

    fn naive_tmf(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        length: usize,
        smoothing_length: usize,
        ma_type: TwiggsMaType,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut adv = vec![f64::NAN; close.len()];
        for i in 1..close.len() {
            if is_valid_bar(high[i], low[i], close[i], volume[i]) && close[i - 1].is_finite() {
                let tr_h = close[i - 1].max(high[i]);
                let tr_l = close[i - 1].min(low[i]);
                let tr_c = tr_h - tr_l;
                let denom = if tr_c == 0.0 {
                    ZERO_RANGE_DIVISOR
                } else {
                    tr_c
                };
                adv[i] = volume[i] * (((close[i] - tr_l) - (tr_h - close[i])) / denom);
            }
        }

        let vol_ma = match ma_type {
            TwiggsMaType::Sma => naive_sma(volume, length),
            TwiggsMaType::Ema => naive_ema(volume, length),
            TwiggsMaType::Wma => naive_wma(volume, length),
            TwiggsMaType::Vwma => naive_vwma(volume, volume, length),
        };
        let adv_ma = match ma_type {
            TwiggsMaType::Sma => naive_sma(&adv, length),
            TwiggsMaType::Ema => naive_ema(&adv, length),
            TwiggsMaType::Wma => naive_wma(&adv, length),
            TwiggsMaType::Vwma => naive_vwma(&adv, volume, length),
        };

        let mut tmf = vec![f64::NAN; close.len()];
        for i in 0..close.len() {
            if vol_ma[i] == 0.0 {
                tmf[i] = 0.0;
            } else if vol_ma[i].is_finite() && adv_ma[i].is_finite() {
                tmf[i] = adv_ma[i] / vol_ma[i];
            }
        }
        let smoothed = naive_hma(&tmf, smoothing_length);
        (tmf, smoothed)
    }

    #[test]
    fn twiggs_money_flow_matches_naive() {
        let (high, low, close, volume) = sample_hlcv();
        let params = TwiggsMoneyFlowParams {
            length: Some(5),
            smoothing_length: Some(4),
            ma_type: Some("WMA".to_string()),
        };
        let input = TwiggsMoneyFlowInput::from_slices(&high, &low, &close, &volume, params);
        let out = twiggs_money_flow(&input).expect("tmf output");
        let (exp_tmf, exp_smoothed) =
            naive_tmf(&high, &low, &close, &volume, 5, 4, TwiggsMaType::Wma);
        assert!(series_close(&out.tmf, &exp_tmf, 1e-12));
        assert!(series_close(&out.smoothed, &exp_smoothed, 1e-12));
    }

    #[test]
    fn twiggs_money_flow_into_matches_api() -> Result<(), TwiggsMoneyFlowError> {
        let (high, low, close, volume) = sample_hlcv();
        let params = TwiggsMoneyFlowParams {
            length: Some(6),
            smoothing_length: Some(5),
            ma_type: Some("EMA".to_string()),
        };
        let input = TwiggsMoneyFlowInput::from_slices(&high, &low, &close, &volume, params);
        let base = twiggs_money_flow(&input)?;
        let mut tmf = vec![0.0; close.len()];
        let mut smoothed = vec![0.0; close.len()];
        twiggs_money_flow_into(&input, &mut tmf, &mut smoothed)?;
        assert!(series_close(&base.tmf, &tmf, 1e-12));
        assert!(series_close(&base.smoothed, &smoothed, 1e-12));
        Ok(())
    }

    #[test]
    fn twiggs_money_flow_stream_matches_batch() {
        let (high, low, close, volume) = sample_hlcv();
        let params = TwiggsMoneyFlowParams {
            length: Some(5),
            smoothing_length: Some(4),
            ma_type: Some("VWMA".to_string()),
        };
        let input = TwiggsMoneyFlowInput::from_slices(&high, &low, &close, &volume, params.clone());
        let batch = twiggs_money_flow(&input).expect("batch");
        let mut stream = TwiggsMoneyFlowStream::try_new(params).expect("stream");
        let mut tmf = Vec::with_capacity(close.len());
        let mut smoothed = Vec::with_capacity(close.len());

        for i in 0..close.len() {
            if let Some((a, b)) = stream.update(high[i], low[i], close[i], volume[i]) {
                tmf.push(a);
                smoothed.push(b);
            } else {
                tmf.push(f64::NAN);
                smoothed.push(f64::NAN);
            }
        }

        assert!(series_close(&batch.tmf, &tmf, 1e-12));
        assert!(series_close(&batch.smoothed, &smoothed, 1e-12));
    }

    #[test]
    fn twiggs_money_flow_batch_single_param_matches_single() {
        let (high, low, close, volume) = sample_hlcv();
        let sweep = TwiggsMoneyFlowBatchRange {
            length: (5, 5, 0),
            smoothing_length: (4, 4, 0),
            ma_type: "WMA".to_string(),
        };
        let batch =
            twiggs_money_flow_batch_with_kernel(&high, &low, &close, &volume, &sweep, Kernel::Auto)
                .expect("batch output");
        let single = twiggs_money_flow(&TwiggsMoneyFlowInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            TwiggsMoneyFlowParams {
                length: Some(5),
                smoothing_length: Some(4),
                ma_type: Some("WMA".to_string()),
            },
        ))
        .expect("single output");

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert!(series_close(&batch.tmf[..close.len()], &single.tmf, 1e-12));
        assert!(series_close(
            &batch.smoothed[..close.len()],
            &single.smoothed,
            1e-12
        ));
    }

    #[test]
    fn twiggs_money_flow_rejects_invalid_ma_type() {
        let (high, low, close, volume) = sample_hlcv();
        let input = TwiggsMoneyFlowInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            TwiggsMoneyFlowParams {
                length: Some(5),
                smoothing_length: Some(4),
                ma_type: Some("INVALID".to_string()),
            },
        );
        assert!(matches!(
            twiggs_money_flow(&input),
            Err(TwiggsMoneyFlowError::InvalidMaType { .. })
        ));
    }
}
