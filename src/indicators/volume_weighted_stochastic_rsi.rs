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
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for VolumeWeightedStochasticRsiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VolumeWeightedStochasticRsiData::Slice { source, .. } => source,
            VolumeWeightedStochasticRsiData::Candles { candles, source } => {
                if source.eq_ignore_ascii_case("close") {
                    candles.close.as_slice()
                } else {
                    source_type(candles, source)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum VolumeWeightedStochasticRsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice {
        source: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VolumeWeightedStochasticRsiOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VolumeWeightedStochasticRsiParams {
    pub rsi_length: Option<usize>,
    pub stoch_length: Option<usize>,
    pub k_length: Option<usize>,
    pub d_length: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for VolumeWeightedStochasticRsiParams {
    fn default() -> Self {
        Self {
            rsi_length: Some(14),
            stoch_length: Some(14),
            k_length: Some(3),
            d_length: Some(3),
            ma_type: Some("WSMA".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeWeightedStochasticRsiInput<'a> {
    pub data: VolumeWeightedStochasticRsiData<'a>,
    pub params: VolumeWeightedStochasticRsiParams,
}

impl<'a> VolumeWeightedStochasticRsiInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: VolumeWeightedStochasticRsiParams,
    ) -> Self {
        Self {
            data: VolumeWeightedStochasticRsiData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        source: &'a [f64],
        volume: &'a [f64],
        params: VolumeWeightedStochasticRsiParams,
    ) -> Self {
        Self {
            data: VolumeWeightedStochasticRsiData::Slice { source, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            VolumeWeightedStochasticRsiParams::default(),
        )
    }

    #[inline]
    pub fn get_rsi_length(&self) -> usize {
        self.params.rsi_length.unwrap_or(14)
    }

    #[inline]
    pub fn get_stoch_length(&self) -> usize {
        self.params.stoch_length.unwrap_or(14)
    }

    #[inline]
    pub fn get_k_length(&self) -> usize {
        self.params.k_length.unwrap_or(3)
    }

    #[inline]
    pub fn get_d_length(&self) -> usize {
        self.params.d_length.unwrap_or(3)
    }

    #[inline]
    pub fn ma_type_str(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or("WSMA")
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            VolumeWeightedStochasticRsiData::Candles { candles, source } => (
                if source.eq_ignore_ascii_case("close") {
                    candles.close.as_slice()
                } else {
                    source_type(candles, source)
                },
                candles.volume.as_slice(),
            ),
            VolumeWeightedStochasticRsiData::Slice { source, volume } => (*source, *volume),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VolumeWeightedStochasticRsiBuilder {
    rsi_length: Option<usize>,
    stoch_length: Option<usize>,
    k_length: Option<usize>,
    d_length: Option<usize>,
    ma_type: Option<String>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for VolumeWeightedStochasticRsiBuilder {
    fn default() -> Self {
        Self {
            rsi_length: None,
            stoch_length: None,
            k_length: None,
            d_length: None,
            ma_type: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VolumeWeightedStochasticRsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn rsi_length(mut self, value: usize) -> Self {
        self.rsi_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn stoch_length(mut self, value: usize) -> Self {
        self.stoch_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn k_length(mut self, value: usize) -> Self {
        self.k_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn d_length(mut self, value: usize) -> Self {
        self.d_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn ma_type<S: Into<String>>(mut self, value: S) -> Self {
        self.ma_type = Some(value.into());
        self
    }

    #[inline(always)]
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = Some(value.into());
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
    ) -> Result<VolumeWeightedStochasticRsiOutput, VolumeWeightedStochasticRsiError> {
        let params = VolumeWeightedStochasticRsiParams {
            rsi_length: self.rsi_length,
            stoch_length: self.stoch_length,
            k_length: self.k_length,
            d_length: self.d_length,
            ma_type: self.ma_type,
        };
        let input = VolumeWeightedStochasticRsiInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or("close"),
            params,
        );
        volume_weighted_stochastic_rsi_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<VolumeWeightedStochasticRsiOutput, VolumeWeightedStochasticRsiError> {
        let params = VolumeWeightedStochasticRsiParams {
            rsi_length: self.rsi_length,
            stoch_length: self.stoch_length,
            k_length: self.k_length,
            d_length: self.d_length,
            ma_type: self.ma_type,
        };
        let input = VolumeWeightedStochasticRsiInput::from_slices(source, volume, params);
        volume_weighted_stochastic_rsi_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<VolumeWeightedStochasticRsiStream, VolumeWeightedStochasticRsiError> {
        let params = VolumeWeightedStochasticRsiParams {
            rsi_length: self.rsi_length,
            stoch_length: self.stoch_length,
            k_length: self.k_length,
            d_length: self.d_length,
            ma_type: self.ma_type,
        };
        VolumeWeightedStochasticRsiStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum VolumeWeightedStochasticRsiError {
    #[error("volume_weighted_stochastic_rsi: Empty input data.")]
    EmptyInputData,
    #[error("volume_weighted_stochastic_rsi: Source and volume length mismatch.")]
    DataLengthMismatch,
    #[error("volume_weighted_stochastic_rsi: All source/volume pairs are invalid.")]
    AllValuesNaN,
    #[error("volume_weighted_stochastic_rsi: Invalid RSI length: rsi_length = {rsi_length}, data length = {data_len}")]
    InvalidRsiLength { rsi_length: usize, data_len: usize },
    #[error("volume_weighted_stochastic_rsi: Invalid stochastic length: stoch_length = {stoch_length}, data length = {data_len}")]
    InvalidStochLength {
        stoch_length: usize,
        data_len: usize,
    },
    #[error("volume_weighted_stochastic_rsi: Invalid K length: k_length = {k_length}, data length = {data_len}")]
    InvalidKLength { k_length: usize, data_len: usize },
    #[error("volume_weighted_stochastic_rsi: Invalid D length: d_length = {d_length}, data length = {data_len}")]
    InvalidDLength { d_length: usize, data_len: usize },
    #[error("volume_weighted_stochastic_rsi: Invalid MA type: {ma_type}")]
    InvalidMaType { ma_type: String },
    #[error(
        "volume_weighted_stochastic_rsi: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("volume_weighted_stochastic_rsi: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "volume_weighted_stochastic_rsi: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("volume_weighted_stochastic_rsi: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("volume_weighted_stochastic_rsi: Invalid input: {0}")]
    InvalidInput(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VwsrsiMaType {
    Wsma,
    Sma,
    Ema,
    Wma,
    Vwma,
}

#[inline(always)]
fn parse_ma_type(value: &str) -> Result<VwsrsiMaType, VolumeWeightedStochasticRsiError> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("WSMA")
        || value.eq_ignore_ascii_case("SMMA")
        || value.eq_ignore_ascii_case("RMA")
        || value.eq_ignore_ascii_case("WILDERS")
        || value.eq_ignore_ascii_case("WILDER")
    {
        Ok(VwsrsiMaType::Wsma)
    } else if value.eq_ignore_ascii_case("SMA") {
        Ok(VwsrsiMaType::Sma)
    } else if value.eq_ignore_ascii_case("EMA") {
        Ok(VwsrsiMaType::Ema)
    } else if value.eq_ignore_ascii_case("WMA") {
        Ok(VwsrsiMaType::Wma)
    } else if value.eq_ignore_ascii_case("VWMA") {
        Ok(VwsrsiMaType::Vwma)
    } else {
        Err(VolumeWeightedStochasticRsiError::InvalidMaType {
            ma_type: value.to_string(),
        })
    }
}

#[inline(always)]
fn normalize_ma_type(value: &str) -> Result<String, VolumeWeightedStochasticRsiError> {
    Ok(match parse_ma_type(value)? {
        VwsrsiMaType::Wsma => "WSMA".to_string(),
        VwsrsiMaType::Sma => "SMA".to_string(),
        VwsrsiMaType::Ema => "EMA".to_string(),
        VwsrsiMaType::Wma => "WMA".to_string(),
        VwsrsiMaType::Vwma => "VWMA".to_string(),
    })
}

#[inline(always)]
fn ma_extra_bars(ma_type: VwsrsiMaType, period: usize) -> usize {
    match ma_type {
        VwsrsiMaType::Ema => 0,
        VwsrsiMaType::Wsma | VwsrsiMaType::Sma | VwsrsiMaType::Wma | VwsrsiMaType::Vwma => {
            period.saturating_sub(1)
        }
    }
}

#[inline(always)]
fn k_warmup(
    first: usize,
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    ma_type: VwsrsiMaType,
) -> usize {
    first + rsi_length + stoch_length - 1 + ma_extra_bars(ma_type, k_length)
}

#[inline(always)]
fn d_warmup(
    first: usize,
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    d_length: usize,
    ma_type: VwsrsiMaType,
) -> usize {
    k_warmup(first, rsi_length, stoch_length, k_length, ma_type) + ma_extra_bars(ma_type, d_length)
}

#[inline(always)]
fn needed_bars(
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    d_length: usize,
    ma_type: VwsrsiMaType,
) -> usize {
    rsi_length + stoch_length + ma_extra_bars(ma_type, k_length) + ma_extra_bars(ma_type, d_length)
}

#[inline(always)]
fn is_valid_pair(source: f64, volume: f64) -> bool {
    source.is_finite() && volume.is_finite()
}

#[inline(always)]
fn first_valid_pair(source: &[f64], volume: &[f64]) -> Option<usize> {
    source
        .iter()
        .zip(volume.iter())
        .position(|(&s, &v)| is_valid_pair(s, v))
}

#[inline(always)]
fn rsi_from_avgs(avg_gain: f64, avg_loss: f64) -> f64 {
    if avg_loss <= 0.0 {
        if avg_gain <= 0.0 {
            50.0
        } else {
            100.0
        }
    } else if avg_gain <= 0.0 {
        0.0
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - 100.0 / (1.0 + rs)
    }
}

#[derive(Clone, Debug)]
struct WeightedRsiState {
    period: usize,
    prev_source: Option<f64>,
    gain_sum: f64,
    loss_sum: f64,
    count: usize,
    avg_gain: f64,
    avg_loss: f64,
    initialized: bool,
}

impl WeightedRsiState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            prev_source: None,
            gain_sum: 0.0,
            loss_sum: 0.0,
            count: 0,
            avg_gain: 0.0,
            avg_loss: 0.0,
            initialized: false,
        }
    }

    #[inline]
    fn update(&mut self, source: f64, volume: f64) -> f64 {
        if !is_valid_pair(source, volume) {
            self.prev_source = if source.is_finite() {
                Some(source)
            } else {
                None
            };
            return f64::NAN;
        }

        let prev_source = match self.prev_source {
            Some(prev) if prev.is_finite() => prev,
            _ => {
                self.prev_source = Some(source);
                return f64::NAN;
            }
        };

        let change = source - prev_source;
        let gain = change.max(0.0) * volume;
        let loss = (-change).max(0.0) * volume;
        self.prev_source = Some(source);

        if !self.initialized {
            self.gain_sum += gain;
            self.loss_sum += loss;
            self.count += 1;
            if self.count == self.period {
                self.avg_gain = self.gain_sum / self.period as f64;
                self.avg_loss = self.loss_sum / self.period as f64;
                self.initialized = true;
                return rsi_from_avgs(self.avg_gain, self.avg_loss);
            }
            return f64::NAN;
        }

        let period = self.period as f64;
        self.avg_gain = (self.avg_gain * (period - 1.0) + gain) / period;
        self.avg_loss = (self.avg_loss * (period - 1.0) + loss) / period;
        rsi_from_avgs(self.avg_gain, self.avg_loss)
    }
}

#[derive(Clone, Debug)]
struct StochState {
    window: Vec<f64>,
    head: usize,
    count: usize,
    valid: usize,
}

impl StochState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            window: vec![f64::NAN; period],
            head: 0,
            count: 0,
            valid: 0,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        let period = self.window.len();
        if self.count == period {
            let old = self.window[self.head];
            if old.is_finite() {
                self.valid -= 1;
            }
        } else {
            self.count += 1;
        }

        self.window[self.head] = value;
        self.head += 1;
        if self.head == period {
            self.head = 0;
        }

        if value.is_finite() {
            self.valid += 1;
        }

        if self.count < period || self.valid < period || !value.is_finite() {
            return f64::NAN;
        }

        let mut lowest = f64::INFINITY;
        let mut highest = f64::NEG_INFINITY;
        for &entry in &self.window {
            lowest = lowest.min(entry);
            highest = highest.max(entry);
        }
        let denom = highest - lowest;
        if !denom.is_finite() || denom == 0.0 {
            f64::NAN
        } else {
            (value - lowest) / denom * 100.0
        }
    }
}

#[derive(Clone, Debug)]
struct WeightedSmaState {
    numerators: Vec<f64>,
    weights: Vec<f64>,
    head: usize,
    count: usize,
    numerator_sum: f64,
    weight_sum: f64,
}

impl WeightedSmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            numerators: vec![0.0; period],
            weights: vec![0.0; period],
            head: 0,
            count: 0,
            numerator_sum: 0.0,
            weight_sum: 0.0,
        }
    }

    #[inline]
    fn update(&mut self, value: f64, weight: f64) -> f64 {
        let numerator = value * weight;
        let period = self.numerators.len();
        if self.count < period {
            self.numerators[self.head] = numerator;
            self.weights[self.head] = weight;
            self.head = (self.head + 1) % period;
            self.count += 1;
            self.numerator_sum += numerator;
            self.weight_sum += weight;
        } else {
            let old_numerator = self.numerators[self.head];
            let old_weight = self.weights[self.head];
            self.numerators[self.head] = numerator;
            self.weights[self.head] = weight;
            self.head = (self.head + 1) % period;
            self.numerator_sum += numerator - old_numerator;
            self.weight_sum += weight - old_weight;
        }

        if self.count == period && self.weight_sum != 0.0 {
            self.numerator_sum / self.weight_sum
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Debug)]
struct WeightedEmaState {
    alpha: f64,
    numerator: f64,
    denominator: f64,
    initialized: bool,
}

impl WeightedEmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            numerator: 0.0,
            denominator: 0.0,
            initialized: false,
        }
    }

    #[inline]
    fn update(&mut self, value: f64, weight: f64) -> f64 {
        let numerator = value * weight;
        if !self.initialized {
            self.numerator = numerator;
            self.denominator = weight;
            self.initialized = true;
        } else {
            let beta = 1.0 - self.alpha;
            self.numerator = self.alpha * numerator + beta * self.numerator;
            self.denominator = self.alpha * weight + beta * self.denominator;
        }

        if self.denominator != 0.0 {
            self.numerator / self.denominator
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Debug)]
struct WeightedWsmaState {
    period: usize,
    numerator_sum: f64,
    denominator_sum: f64,
    count: usize,
    numerator_avg: f64,
    denominator_avg: f64,
    initialized: bool,
}

impl WeightedWsmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            numerator_sum: 0.0,
            denominator_sum: 0.0,
            count: 0,
            numerator_avg: 0.0,
            denominator_avg: 0.0,
            initialized: false,
        }
    }

    #[inline]
    fn update(&mut self, value: f64, weight: f64) -> f64 {
        let numerator = value * weight;
        if !self.initialized {
            self.numerator_sum += numerator;
            self.denominator_sum += weight;
            self.count += 1;
            if self.count == self.period {
                self.numerator_avg = self.numerator_sum / self.period as f64;
                self.denominator_avg = self.denominator_sum / self.period as f64;
                self.initialized = true;
                if self.denominator_avg != 0.0 {
                    return self.numerator_avg / self.denominator_avg;
                }
            }
            return f64::NAN;
        }

        let period = self.period as f64;
        self.numerator_avg = (self.numerator_avg * (period - 1.0) + numerator) / period;
        self.denominator_avg = (self.denominator_avg * (period - 1.0) + weight) / period;
        if self.denominator_avg != 0.0 {
            self.numerator_avg / self.denominator_avg
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Debug)]
struct WeightedWmaState {
    numerators: Vec<f64>,
    weights: Vec<f64>,
    head: usize,
    count: usize,
    numerator_plain_sum: f64,
    numerator_weighted_sum: f64,
    weight_plain_sum: f64,
    weight_weighted_sum: f64,
}

impl WeightedWmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            numerators: vec![0.0; period],
            weights: vec![0.0; period],
            head: 0,
            count: 0,
            numerator_plain_sum: 0.0,
            numerator_weighted_sum: 0.0,
            weight_plain_sum: 0.0,
            weight_weighted_sum: 0.0,
        }
    }

    #[inline]
    fn update(&mut self, value: f64, weight: f64) -> f64 {
        let numerator = value * weight;
        let period = self.numerators.len();
        if self.count < period {
            self.numerators[self.head] = numerator;
            self.weights[self.head] = weight;
            self.head = (self.head + 1) % period;
            self.count += 1;
            self.numerator_plain_sum += numerator;
            self.weight_plain_sum += weight;
            self.numerator_weighted_sum += numerator * self.count as f64;
            self.weight_weighted_sum += weight * self.count as f64;
        } else {
            let old_numerator = self.numerators[self.head];
            let old_weight = self.weights[self.head];
            let prev_numerator_plain = self.numerator_plain_sum;
            let prev_weight_plain = self.weight_plain_sum;
            self.numerators[self.head] = numerator;
            self.weights[self.head] = weight;
            self.head = (self.head + 1) % period;
            self.numerator_plain_sum = prev_numerator_plain - old_numerator + numerator;
            self.weight_plain_sum = prev_weight_plain - old_weight + weight;
            self.numerator_weighted_sum =
                self.numerator_weighted_sum - prev_numerator_plain + numerator * period as f64;
            self.weight_weighted_sum =
                self.weight_weighted_sum - prev_weight_plain + weight * period as f64;
        }

        if self.count == period && self.weight_weighted_sum != 0.0 {
            self.numerator_weighted_sum / self.weight_weighted_sum
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Debug)]
enum WeightedMaState {
    Wsma(WeightedWsmaState),
    Sma(WeightedSmaState),
    Ema(WeightedEmaState),
    Wma(WeightedWmaState),
    Vwma(WeightedSmaState),
}

impl WeightedMaState {
    #[inline]
    fn new(ma_type: VwsrsiMaType, period: usize) -> Self {
        match ma_type {
            VwsrsiMaType::Wsma => Self::Wsma(WeightedWsmaState::new(period)),
            VwsrsiMaType::Sma => Self::Sma(WeightedSmaState::new(period)),
            VwsrsiMaType::Ema => Self::Ema(WeightedEmaState::new(period)),
            VwsrsiMaType::Wma => Self::Wma(WeightedWmaState::new(period)),
            VwsrsiMaType::Vwma => Self::Vwma(WeightedSmaState::new(period)),
        }
    }

    #[inline]
    fn update(&mut self, value: f64, weight: f64) -> f64 {
        match self {
            WeightedMaState::Wsma(state) => state.update(value, weight),
            WeightedMaState::Sma(state) => state.update(value, weight),
            WeightedMaState::Ema(state) => state.update(value, weight),
            WeightedMaState::Wma(state) => state.update(value, weight),
            WeightedMaState::Vwma(state) => state.update(value, weight),
        }
    }
}

#[derive(Clone, Debug)]
struct VolumeWeightedStochasticRsiState {
    rsi: WeightedRsiState,
    stoch: StochState,
    k_ma: WeightedMaState,
    d_ma: WeightedMaState,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum VolumeWeightedStochasticRsiOutputField {
    K,
    D,
}

impl VolumeWeightedStochasticRsiState {
    #[inline]
    fn new(
        rsi_length: usize,
        stoch_length: usize,
        k_length: usize,
        d_length: usize,
        ma_type: VwsrsiMaType,
    ) -> Self {
        Self {
            rsi: WeightedRsiState::new(rsi_length),
            stoch: StochState::new(stoch_length),
            k_ma: WeightedMaState::new(ma_type, k_length),
            d_ma: WeightedMaState::new(ma_type, d_length),
        }
    }

    #[inline]
    fn update(&mut self, source: f64, volume: f64) -> (f64, f64) {
        let rsi = self.rsi.update(source, volume);
        let stoch = self.stoch.update(rsi);
        let k = if stoch.is_finite() {
            self.k_ma.update(stoch, volume)
        } else {
            f64::NAN
        };
        let d = if k.is_finite() {
            self.d_ma.update(k, volume)
        } else {
            f64::NAN
        };
        (k, d)
    }
}

#[inline(always)]
fn volume_weighted_stochastic_rsi_compute_k_into(
    source: &[f64],
    volume: &[f64],
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    ma_type: VwsrsiMaType,
    out_k: &mut [f64],
) {
    let mut rsi = WeightedRsiState::new(rsi_length);
    let mut stoch_state = StochState::new(stoch_length);
    let mut k_ma = WeightedMaState::new(ma_type, k_length);
    for i in 0..source.len() {
        let rsi_value = rsi.update(source[i], volume[i]);
        let stoch = stoch_state.update(rsi_value);
        out_k[i] = if stoch.is_finite() {
            k_ma.update(stoch, volume[i])
        } else {
            f64::NAN
        };
    }
}

#[inline(always)]
fn volume_weighted_stochastic_rsi_compute_into(
    source: &[f64],
    volume: &[f64],
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    d_length: usize,
    ma_type: VwsrsiMaType,
    out_k: &mut [f64],
    out_d: &mut [f64],
) {
    let mut state = VolumeWeightedStochasticRsiState::new(
        rsi_length,
        stoch_length,
        k_length,
        d_length,
        ma_type,
    );
    for i in 0..source.len() {
        let (k, d) = state.update(source[i], volume[i]);
        out_k[i] = k;
        out_d[i] = d;
    }
}

#[inline(always)]
fn volume_weighted_stochastic_rsi_prepare<'a>(
    input: &'a VolumeWeightedStochasticRsiInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        usize,
        usize,
        VwsrsiMaType,
        usize,
        Kernel,
    ),
    VolumeWeightedStochasticRsiError,
> {
    let (source, volume) = input.as_refs();
    let len = source.len();
    if len == 0 {
        return Err(VolumeWeightedStochasticRsiError::EmptyInputData);
    }
    if volume.len() != len {
        return Err(VolumeWeightedStochasticRsiError::DataLengthMismatch);
    }

    let rsi_length = input.get_rsi_length();
    let stoch_length = input.get_stoch_length();
    let k_length = input.get_k_length();
    let d_length = input.get_d_length();
    let ma_type = parse_ma_type(input.ma_type_str())?;

    if rsi_length == 0 || rsi_length > len {
        return Err(VolumeWeightedStochasticRsiError::InvalidRsiLength {
            rsi_length,
            data_len: len,
        });
    }
    if stoch_length == 0 || stoch_length > len {
        return Err(VolumeWeightedStochasticRsiError::InvalidStochLength {
            stoch_length,
            data_len: len,
        });
    }
    if k_length == 0 || k_length > len {
        return Err(VolumeWeightedStochasticRsiError::InvalidKLength {
            k_length,
            data_len: len,
        });
    }
    if d_length == 0 || d_length > len {
        return Err(VolumeWeightedStochasticRsiError::InvalidDLength {
            d_length,
            data_len: len,
        });
    }

    let first =
        first_valid_pair(source, volume).ok_or(VolumeWeightedStochasticRsiError::AllValuesNaN)?;
    let valid = len - first;
    let needed = needed_bars(rsi_length, stoch_length, k_length, d_length, ma_type);
    if valid < needed {
        return Err(VolumeWeightedStochasticRsiError::NotEnoughValidData { needed, valid });
    }

    let chosen = kernel.to_non_batch();

    Ok((
        source,
        volume,
        rsi_length,
        stoch_length,
        k_length,
        d_length,
        ma_type,
        first,
        chosen,
    ))
}

#[inline]
pub fn volume_weighted_stochastic_rsi(
    input: &VolumeWeightedStochasticRsiInput,
) -> Result<VolumeWeightedStochasticRsiOutput, VolumeWeightedStochasticRsiError> {
    volume_weighted_stochastic_rsi_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn volume_weighted_stochastic_rsi_with_kernel(
    input: &VolumeWeightedStochasticRsiInput,
    kernel: Kernel,
) -> Result<VolumeWeightedStochasticRsiOutput, VolumeWeightedStochasticRsiError> {
    let (source, volume, rsi_length, stoch_length, k_length, d_length, ma_type, first, _chosen) =
        volume_weighted_stochastic_rsi_prepare(input, kernel)?;
    let _ = first;
    let mut k = alloc_uninit_f64(source.len());
    let mut d = alloc_uninit_f64(source.len());
    volume_weighted_stochastic_rsi_compute_into(
        source,
        volume,
        rsi_length,
        stoch_length,
        k_length,
        d_length,
        ma_type,
        &mut k,
        &mut d,
    );
    Ok(VolumeWeightedStochasticRsiOutput { k, d })
}

#[inline]
pub fn volume_weighted_stochastic_rsi_into_slice(
    dst_k: &mut [f64],
    dst_d: &mut [f64],
    input: &VolumeWeightedStochasticRsiInput,
    kernel: Kernel,
) -> Result<(), VolumeWeightedStochasticRsiError> {
    let (source, volume, rsi_length, stoch_length, k_length, d_length, ma_type, _first, _chosen) =
        volume_weighted_stochastic_rsi_prepare(input, kernel)?;
    if dst_k.len() != source.len() || dst_d.len() != source.len() {
        return Err(VolumeWeightedStochasticRsiError::OutputLengthMismatch {
            expected: source.len(),
            got: dst_k.len().max(dst_d.len()),
        });
    }
    volume_weighted_stochastic_rsi_compute_into(
        source,
        volume,
        rsi_length,
        stoch_length,
        k_length,
        d_length,
        ma_type,
        dst_k,
        dst_d,
    );
    Ok(())
}

#[inline]
pub(crate) fn volume_weighted_stochastic_rsi_output_into_slice(
    dst: &mut [f64],
    input: &VolumeWeightedStochasticRsiInput,
    kernel: Kernel,
    field: VolumeWeightedStochasticRsiOutputField,
) -> Result<(), VolumeWeightedStochasticRsiError> {
    let (source, volume, rsi_length, stoch_length, k_length, d_length, ma_type, _first, _chosen) =
        volume_weighted_stochastic_rsi_prepare(input, kernel)?;
    if dst.len() != source.len() {
        return Err(VolumeWeightedStochasticRsiError::OutputLengthMismatch {
            expected: source.len(),
            got: dst.len(),
        });
    }
    match field {
        VolumeWeightedStochasticRsiOutputField::K => {
            volume_weighted_stochastic_rsi_compute_k_into(
                source,
                volume,
                rsi_length,
                stoch_length,
                k_length,
                ma_type,
                dst,
            );
        }
        VolumeWeightedStochasticRsiOutputField::D => {
            let mut k = alloc_uninit_f64(source.len());
            volume_weighted_stochastic_rsi_compute_into(
                source,
                volume,
                rsi_length,
                stoch_length,
                k_length,
                d_length,
                ma_type,
                &mut k,
                dst,
            );
        }
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn volume_weighted_stochastic_rsi_into(
    input: &VolumeWeightedStochasticRsiInput,
    out_k: &mut [f64],
    out_d: &mut [f64],
) -> Result<(), VolumeWeightedStochasticRsiError> {
    volume_weighted_stochastic_rsi_into_slice(out_k, out_d, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct VolumeWeightedStochasticRsiStream {
    state: VolumeWeightedStochasticRsiState,
}

impl VolumeWeightedStochasticRsiStream {
    pub fn try_new(
        params: VolumeWeightedStochasticRsiParams,
    ) -> Result<Self, VolumeWeightedStochasticRsiError> {
        let rsi_length = params.rsi_length.unwrap_or(14);
        let stoch_length = params.stoch_length.unwrap_or(14);
        let k_length = params.k_length.unwrap_or(3);
        let d_length = params.d_length.unwrap_or(3);
        let ma_type = parse_ma_type(params.ma_type.as_deref().unwrap_or("WSMA"))?;
        if rsi_length == 0 {
            return Err(VolumeWeightedStochasticRsiError::InvalidRsiLength {
                rsi_length,
                data_len: 0,
            });
        }
        if stoch_length == 0 {
            return Err(VolumeWeightedStochasticRsiError::InvalidStochLength {
                stoch_length,
                data_len: 0,
            });
        }
        if k_length == 0 {
            return Err(VolumeWeightedStochasticRsiError::InvalidKLength {
                k_length,
                data_len: 0,
            });
        }
        if d_length == 0 {
            return Err(VolumeWeightedStochasticRsiError::InvalidDLength {
                d_length,
                data_len: 0,
            });
        }
        Ok(Self {
            state: VolumeWeightedStochasticRsiState::new(
                rsi_length,
                stoch_length,
                k_length,
                d_length,
                ma_type,
            ),
        })
    }

    #[inline]
    pub fn update(&mut self, source: f64, volume: f64) -> Option<(f64, f64)> {
        let out = self.state.update(source, volume);
        if out.0.is_finite() || out.1.is_finite() {
            Some(out)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct VolumeWeightedStochasticRsiBatchRange {
    pub rsi_length: (usize, usize, usize),
    pub stoch_length: (usize, usize, usize),
    pub k_length: (usize, usize, usize),
    pub d_length: (usize, usize, usize),
    pub ma_type: String,
}

impl Default for VolumeWeightedStochasticRsiBatchRange {
    fn default() -> Self {
        Self {
            rsi_length: (14, 14, 0),
            stoch_length: (14, 14, 0),
            k_length: (3, 3, 0),
            d_length: (3, 3, 0),
            ma_type: "WSMA".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VolumeWeightedStochasticRsiBatchBuilder {
    range: VolumeWeightedStochasticRsiBatchRange,
    source: String,
    kernel: Kernel,
}

impl Default for VolumeWeightedStochasticRsiBatchBuilder {
    fn default() -> Self {
        Self {
            range: VolumeWeightedStochasticRsiBatchRange::default(),
            source: "close".to_string(),
            kernel: Kernel::Auto,
        }
    }
}

impl VolumeWeightedStochasticRsiBatchBuilder {
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
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = value.into();
        self
    }

    #[inline]
    pub fn ma_type<S: Into<String>>(mut self, value: S) -> Self {
        self.range.ma_type = value.into();
        self
    }

    #[inline]
    pub fn rsi_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rsi_length = (start, end, step);
        self
    }

    #[inline]
    pub fn stoch_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.stoch_length = (start, end, step);
        self
    }

    #[inline]
    pub fn k_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.k_length = (start, end, step);
        self
    }

    #[inline]
    pub fn d_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.d_length = (start, end, step);
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<VolumeWeightedStochasticRsiBatchOutput, VolumeWeightedStochasticRsiError> {
        volume_weighted_stochastic_rsi_batch_with_kernel(source, volume, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<VolumeWeightedStochasticRsiBatchOutput, VolumeWeightedStochasticRsiError> {
        volume_weighted_stochastic_rsi_batch_with_kernel(
            source_type(candles, &self.source),
            candles.volume.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolumeWeightedStochasticRsiBatchConfig {
    pub rsi_length_range: Vec<usize>,
    pub stoch_length_range: Vec<usize>,
    pub k_length_range: Vec<usize>,
    pub d_length_range: Vec<usize>,
    pub ma_type: String,
}

#[derive(Clone, Debug)]
pub struct VolumeWeightedStochasticRsiBatchOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub combos: Vec<VolumeWeightedStochasticRsiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VolumeWeightedStochasticRsiBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &VolumeWeightedStochasticRsiParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.rsi_length.unwrap_or(14) == params.rsi_length.unwrap_or(14)
                && combo.stoch_length.unwrap_or(14) == params.stoch_length.unwrap_or(14)
                && combo.k_length.unwrap_or(3) == params.k_length.unwrap_or(3)
                && combo.d_length.unwrap_or(3) == params.d_length.unwrap_or(3)
                && combo.ma_type.as_deref().unwrap_or("WSMA")
                    == params.ma_type.as_deref().unwrap_or("WSMA")
        })
    }
}

pub fn expand_grid_volume_weighted_stochastic_rsi(
    range: &VolumeWeightedStochasticRsiBatchRange,
) -> Result<Vec<VolumeWeightedStochasticRsiParams>, VolumeWeightedStochasticRsiError> {
    fn axis(range: (usize, usize, usize)) -> Result<Vec<usize>, VolumeWeightedStochasticRsiError> {
        let (start, end, step) = range;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            while value <= end {
                out.push(value);
                value = value.saturating_add(step);
                if step == 0 {
                    break;
                }
            }
        } else {
            let mut value = start;
            while value >= end {
                out.push(value);
                if value < step {
                    break;
                }
                value -= step;
                if step == 0 {
                    break;
                }
            }
        }
        if out.is_empty() {
            return Err(VolumeWeightedStochasticRsiError::InvalidRange { start, end, step });
        }
        Ok(out)
    }

    let rsi_lengths = axis(range.rsi_length)?;
    let stoch_lengths = axis(range.stoch_length)?;
    let k_lengths = axis(range.k_length)?;
    let d_lengths = axis(range.d_length)?;
    let ma_type = normalize_ma_type(&range.ma_type)?;
    let mut combos = Vec::with_capacity(
        rsi_lengths.len() * stoch_lengths.len() * k_lengths.len() * d_lengths.len(),
    );
    for rsi_length in rsi_lengths {
        for stoch_length in stoch_lengths.iter().copied() {
            for k_length in k_lengths.iter().copied() {
                for d_length in d_lengths.iter().copied() {
                    combos.push(VolumeWeightedStochasticRsiParams {
                        rsi_length: Some(rsi_length),
                        stoch_length: Some(stoch_length),
                        k_length: Some(k_length),
                        d_length: Some(d_length),
                        ma_type: Some(ma_type.clone()),
                    });
                }
            }
        }
    }
    Ok(combos)
}

#[inline]
pub fn volume_weighted_stochastic_rsi_batch_with_kernel(
    source: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedStochasticRsiBatchRange,
    kernel: Kernel,
) -> Result<VolumeWeightedStochasticRsiBatchOutput, VolumeWeightedStochasticRsiError> {
    let batch = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(VolumeWeightedStochasticRsiError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    volume_weighted_stochastic_rsi_batch_par_slice(source, volume, sweep, batch.to_non_batch())
}

#[inline(always)]
pub fn volume_weighted_stochastic_rsi_batch_slice(
    source: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedStochasticRsiBatchRange,
    kernel: Kernel,
) -> Result<VolumeWeightedStochasticRsiBatchOutput, VolumeWeightedStochasticRsiError> {
    volume_weighted_stochastic_rsi_batch_inner(source, volume, sweep, kernel, false)
}

#[inline(always)]
pub fn volume_weighted_stochastic_rsi_batch_par_slice(
    source: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedStochasticRsiBatchRange,
    kernel: Kernel,
) -> Result<VolumeWeightedStochasticRsiBatchOutput, VolumeWeightedStochasticRsiError> {
    volume_weighted_stochastic_rsi_batch_inner(source, volume, sweep, kernel, true)
}

#[inline(always)]
fn volume_weighted_stochastic_rsi_batch_inner(
    source: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedStochasticRsiBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<VolumeWeightedStochasticRsiBatchOutput, VolumeWeightedStochasticRsiError> {
    let combos = expand_grid_volume_weighted_stochastic_rsi(sweep)?;
    if source.is_empty() {
        return Err(VolumeWeightedStochasticRsiError::EmptyInputData);
    }
    if volume.len() != source.len() {
        return Err(VolumeWeightedStochasticRsiError::DataLengthMismatch);
    }
    let first =
        first_valid_pair(source, volume).ok_or(VolumeWeightedStochasticRsiError::AllValuesNaN)?;
    let max_needed = combos
        .iter()
        .map(|p| {
            needed_bars(
                p.rsi_length.unwrap_or(14),
                p.stoch_length.unwrap_or(14),
                p.k_length.unwrap_or(3),
                p.d_length.unwrap_or(3),
                parse_ma_type(p.ma_type.as_deref().unwrap_or("WSMA")).unwrap(),
            )
        })
        .max()
        .unwrap_or(0);
    let valid = source.len() - first;
    if valid < max_needed {
        return Err(VolumeWeightedStochasticRsiError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let rows = combos.len();
    let cols = source.len();
    let mut k_mu = make_uninit_matrix(rows, cols);
    let mut d_mu = make_uninit_matrix(rows, cols);
    let k_warmups: Vec<usize> = combos
        .iter()
        .map(|p| {
            k_warmup(
                first,
                p.rsi_length.unwrap_or(14),
                p.stoch_length.unwrap_or(14),
                p.k_length.unwrap_or(3),
                parse_ma_type(p.ma_type.as_deref().unwrap_or("WSMA")).unwrap(),
            )
        })
        .collect();
    let d_warmups: Vec<usize> = combos
        .iter()
        .map(|p| {
            d_warmup(
                first,
                p.rsi_length.unwrap_or(14),
                p.stoch_length.unwrap_or(14),
                p.k_length.unwrap_or(3),
                p.d_length.unwrap_or(3),
                parse_ma_type(p.ma_type.as_deref().unwrap_or("WSMA")).unwrap(),
            )
        })
        .collect();
    init_matrix_prefixes(&mut k_mu, cols, &k_warmups);
    init_matrix_prefixes(&mut d_mu, cols, &d_warmups);

    let mut k_guard = ManuallyDrop::new(k_mu);
    let mut d_guard = ManuallyDrop::new(d_mu);
    let k =
        unsafe { core::slice::from_raw_parts_mut(k_guard.as_mut_ptr() as *mut f64, k_guard.len()) };
    let d =
        unsafe { core::slice::from_raw_parts_mut(d_guard.as_mut_ptr() as *mut f64, d_guard.len()) };

    volume_weighted_stochastic_rsi_batch_inner_into(
        source,
        volume,
        sweep,
        Kernel::Scalar,
        parallel,
        k,
        d,
    )?;

    let k_values = unsafe {
        Vec::from_raw_parts(
            k_guard.as_mut_ptr() as *mut f64,
            k_guard.len(),
            k_guard.capacity(),
        )
    };
    let d_values = unsafe {
        Vec::from_raw_parts(
            d_guard.as_mut_ptr() as *mut f64,
            d_guard.len(),
            d_guard.capacity(),
        )
    };

    Ok(VolumeWeightedStochasticRsiBatchOutput {
        k: k_values,
        d: d_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn volume_weighted_stochastic_rsi_batch_inner_into(
    source: &[f64],
    volume: &[f64],
    sweep: &VolumeWeightedStochasticRsiBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_k: &mut [f64],
    out_d: &mut [f64],
) -> Result<Vec<VolumeWeightedStochasticRsiParams>, VolumeWeightedStochasticRsiError> {
    let combos = expand_grid_volume_weighted_stochastic_rsi(sweep)?;
    if source.is_empty() {
        return Err(VolumeWeightedStochasticRsiError::EmptyInputData);
    }
    if volume.len() != source.len() {
        return Err(VolumeWeightedStochasticRsiError::DataLengthMismatch);
    }
    let first =
        first_valid_pair(source, volume).ok_or(VolumeWeightedStochasticRsiError::AllValuesNaN)?;
    let max_needed = combos
        .iter()
        .map(|p| {
            needed_bars(
                p.rsi_length.unwrap_or(14),
                p.stoch_length.unwrap_or(14),
                p.k_length.unwrap_or(3),
                p.d_length.unwrap_or(3),
                parse_ma_type(p.ma_type.as_deref().unwrap_or("WSMA")).unwrap(),
            )
        })
        .max()
        .unwrap_or(0);
    let valid = source.len() - first;
    if valid < max_needed {
        return Err(VolumeWeightedStochasticRsiError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(VolumeWeightedStochasticRsiError::InvalidInput(
            "rows*cols overflow",
        ))?;
    if out_k.len() != total || out_d.len() != total {
        return Err(VolumeWeightedStochasticRsiError::OutputLengthMismatch {
            expected: total,
            got: out_k.len().max(out_d.len()),
        });
    }

    unsafe {
        let out_k_mu =
            std::slice::from_raw_parts_mut(out_k.as_mut_ptr() as *mut MaybeUninit<f64>, total);
        let out_d_mu =
            std::slice::from_raw_parts_mut(out_d.as_mut_ptr() as *mut MaybeUninit<f64>, total);
        let k_warmups: Vec<usize> = combos
            .iter()
            .map(|p| {
                k_warmup(
                    first,
                    p.rsi_length.unwrap_or(14),
                    p.stoch_length.unwrap_or(14),
                    p.k_length.unwrap_or(3),
                    parse_ma_type(p.ma_type.as_deref().unwrap_or("WSMA")).unwrap(),
                )
            })
            .collect();
        let d_warmups: Vec<usize> = combos
            .iter()
            .map(|p| {
                d_warmup(
                    first,
                    p.rsi_length.unwrap_or(14),
                    p.stoch_length.unwrap_or(14),
                    p.k_length.unwrap_or(3),
                    p.d_length.unwrap_or(3),
                    parse_ma_type(p.ma_type.as_deref().unwrap_or("WSMA")).unwrap(),
                )
            })
            .collect();
        init_matrix_prefixes(out_k_mu, cols, &k_warmups);
        init_matrix_prefixes(out_d_mu, cols, &d_warmups);
    }

    let do_row = |row: usize, k_row: &mut [f64], d_row: &mut [f64]| {
        let params = &combos[row];
        volume_weighted_stochastic_rsi_compute_into(
            source,
            volume,
            params.rsi_length.unwrap_or(14),
            params.stoch_length.unwrap_or(14),
            params.k_length.unwrap_or(3),
            params.d_length.unwrap_or(3),
            parse_ma_type(params.ma_type.as_deref().unwrap_or("WSMA")).unwrap(),
            k_row,
            d_row,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_k
                .par_chunks_mut(cols)
                .zip(out_d.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (k_row, d_row))| do_row(row, k_row, d_row));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (k_row, d_row)) in out_k
                .chunks_mut(cols)
                .zip(out_d.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, k_row, d_row);
            }
        }
    } else {
        for (row, (k_row, d_row)) in out_k
            .chunks_mut(cols)
            .zip(out_d.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, k_row, d_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "volume_weighted_stochastic_rsi")]
#[pyo3(signature = (source, volume, rsi_length=14, stoch_length=14, k_length=3, d_length=3, ma_type="WSMA", kernel=None))]
pub fn volume_weighted_stochastic_rsi_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    d_length: usize,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = VolumeWeightedStochasticRsiInput::from_slices(
        source,
        volume,
        VolumeWeightedStochasticRsiParams {
            rsi_length: Some(rsi_length),
            stoch_length: Some(stoch_length),
            k_length: Some(k_length),
            d_length: Some(d_length),
            ma_type: Some(ma_type.to_string()),
        },
    );
    let output = py
        .allow_threads(|| volume_weighted_stochastic_rsi_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((output.k.into_pyarray(py), output.d.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "VolumeWeightedStochasticRsiStream")]
pub struct VolumeWeightedStochasticRsiStreamPy {
    stream: VolumeWeightedStochasticRsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VolumeWeightedStochasticRsiStreamPy {
    #[new]
    #[pyo3(signature = (rsi_length=14, stoch_length=14, k_length=3, d_length=3, ma_type="WSMA"))]
    fn new(
        rsi_length: usize,
        stoch_length: usize,
        k_length: usize,
        d_length: usize,
        ma_type: &str,
    ) -> PyResult<Self> {
        let stream =
            VolumeWeightedStochasticRsiStream::try_new(VolumeWeightedStochasticRsiParams {
                rsi_length: Some(rsi_length),
                stoch_length: Some(stoch_length),
                k_length: Some(k_length),
                d_length: Some(d_length),
                ma_type: Some(ma_type.to_string()),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, source: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update(source, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "volume_weighted_stochastic_rsi_batch")]
#[pyo3(signature = (source, volume, rsi_length_range, stoch_length_range, k_length_range, d_length_range, ma_type="WSMA", kernel=None))]
pub fn volume_weighted_stochastic_rsi_batch_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    rsi_length_range: (usize, usize, usize),
    stoch_length_range: (usize, usize, usize),
    k_length_range: (usize, usize, usize),
    d_length_range: (usize, usize, usize),
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let sweep = VolumeWeightedStochasticRsiBatchRange {
        rsi_length: rsi_length_range,
        stoch_length: stoch_length_range,
        k_length: k_length_range,
        d_length: d_length_range,
        ma_type: ma_type.to_string(),
    };
    let combos = expand_grid_volume_weighted_stochastic_rsi(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let k_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let d_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let k_out = unsafe { k_arr.as_slice_mut()? };
    let d_out = unsafe { d_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        volume_weighted_stochastic_rsi_batch_inner_into(
            source,
            volume,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            k_out,
            d_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("k", k_arr.reshape((rows, cols))?)?;
    dict.set_item("d", d_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "rsi_lengths",
        combos
            .iter()
            .map(|p| p.rsi_length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stoch_lengths",
        combos
            .iter()
            .map(|p| p.stoch_length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "k_lengths",
        combos
            .iter()
            .map(|p| p.k_length.unwrap_or(3) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "d_lengths",
        combos
            .iter()
            .map(|p| p.d_length.unwrap_or(3) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_types",
        combos
            .iter()
            .map(|p| p.ma_type.clone().unwrap_or_else(|| "WSMA".to_string()))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_volume_weighted_stochastic_rsi_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(volume_weighted_stochastic_rsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        volume_weighted_stochastic_rsi_batch_py,
        m
    )?)?;
    m.add_class::<VolumeWeightedStochasticRsiStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volume_weighted_stochastic_rsi_js")]
pub fn volume_weighted_stochastic_rsi_js(
    source: &[f64],
    volume: &[f64],
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    d_length: usize,
    ma_type: String,
) -> Result<JsValue, JsValue> {
    let input = VolumeWeightedStochasticRsiInput::from_slices(
        source,
        volume,
        VolumeWeightedStochasticRsiParams {
            rsi_length: Some(rsi_length),
            stoch_length: Some(stoch_length),
            k_length: Some(k_length),
            d_length: Some(d_length),
            ma_type: Some(ma_type),
        },
    );
    let mut k = vec![0.0; source.len()];
    let mut d = vec![0.0; source.len()];
    volume_weighted_stochastic_rsi_into_slice(&mut k, &mut d, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("k"),
        &serde_wasm_bindgen::to_value(&k).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("d"),
        &serde_wasm_bindgen::to_value(&d).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volume_weighted_stochastic_rsi_batch_js")]
pub fn volume_weighted_stochastic_rsi_batch_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VolumeWeightedStochasticRsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.rsi_length_range.len() != 3
        || config.stoch_length_range.len() != 3
        || config.k_length_range.len() != 3
        || config.d_length_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = VolumeWeightedStochasticRsiBatchRange {
        rsi_length: (
            config.rsi_length_range[0],
            config.rsi_length_range[1],
            config.rsi_length_range[2],
        ),
        stoch_length: (
            config.stoch_length_range[0],
            config.stoch_length_range[1],
            config.stoch_length_range[2],
        ),
        k_length: (
            config.k_length_range[0],
            config.k_length_range[1],
            config.k_length_range[2],
        ),
        d_length: (
            config.d_length_range[0],
            config.d_length_range[1],
            config.d_length_range[2],
        ),
        ma_type: config.ma_type,
    };
    let combos = expand_grid_volume_weighted_stochastic_rsi(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    let mut k = vec![0.0; total];
    let mut d = vec![0.0; total];
    volume_weighted_stochastic_rsi_batch_inner_into(
        source,
        volume,
        &sweep,
        Kernel::Scalar,
        false,
        &mut k,
        &mut d,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("k"),
        &serde_wasm_bindgen::to_value(&k).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("d"),
        &serde_wasm_bindgen::to_value(&d).unwrap(),
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
pub fn volume_weighted_stochastic_rsi_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_stochastic_rsi_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_stochastic_rsi_into(
    source_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    d_length: usize,
    ma_type: String,
) -> Result<(), JsValue> {
    if source_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to volume_weighted_stochastic_rsi_into",
        ));
    }
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (k, d) = out.split_at_mut(len);
        let input = VolumeWeightedStochasticRsiInput::from_slices(
            source,
            volume,
            VolumeWeightedStochasticRsiParams {
                rsi_length: Some(rsi_length),
                stoch_length: Some(stoch_length),
                k_length: Some(k_length),
                d_length: Some(d_length),
                ma_type: Some(ma_type),
            },
        );
        volume_weighted_stochastic_rsi_into_slice(k, d, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volume_weighted_stochastic_rsi_into_host")]
pub fn volume_weighted_stochastic_rsi_into_host(
    source: &[f64],
    volume: &[f64],
    out_ptr: *mut f64,
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    d_length: usize,
    ma_type: String,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to volume_weighted_stochastic_rsi_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * source.len());
        let (k, d) = out.split_at_mut(source.len());
        let input = VolumeWeightedStochasticRsiInput::from_slices(
            source,
            volume,
            VolumeWeightedStochasticRsiParams {
                rsi_length: Some(rsi_length),
                stoch_length: Some(stoch_length),
                k_length: Some(k_length),
                d_length: Some(d_length),
                ma_type: Some(ma_type),
            },
        );
        volume_weighted_stochastic_rsi_into_slice(k, d, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_stochastic_rsi_batch_into(
    source_ptr: *const f64,
    volume_ptr: *const f64,
    k_ptr: *mut f64,
    d_ptr: *mut f64,
    len: usize,
    rsi_length_start: usize,
    rsi_length_end: usize,
    rsi_length_step: usize,
    stoch_length_start: usize,
    stoch_length_end: usize,
    stoch_length_step: usize,
    k_length_start: usize,
    k_length_end: usize,
    k_length_step: usize,
    d_length_start: usize,
    d_length_end: usize,
    d_length_step: usize,
    ma_type: String,
) -> Result<usize, JsValue> {
    if source_ptr.is_null() || volume_ptr.is_null() || k_ptr.is_null() || d_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to volume_weighted_stochastic_rsi_batch_into",
        ));
    }
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let sweep = VolumeWeightedStochasticRsiBatchRange {
            rsi_length: (rsi_length_start, rsi_length_end, rsi_length_step),
            stoch_length: (stoch_length_start, stoch_length_end, stoch_length_step),
            k_length: (k_length_start, k_length_end, k_length_step),
            d_length: (d_length_start, d_length_end, d_length_step),
            ma_type,
        };
        let combos = expand_grid_volume_weighted_stochastic_rsi(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out_k = std::slice::from_raw_parts_mut(k_ptr, total);
        let out_d = std::slice::from_raw_parts_mut(d_ptr, total);
        volume_weighted_stochastic_rsi_batch_inner_into(
            source,
            volume,
            &sweep,
            Kernel::Scalar,
            false,
            out_k,
            out_d,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_stochastic_rsi_output_into_js(
    source: &[f64],
    volume: &[f64],
    rsi_length: usize,
    stoch_length: usize,
    k_length: usize,
    d_length: usize,
    ma_type: String,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volume_weighted_stochastic_rsi_js(
        source,
        volume,
        rsi_length,
        stoch_length,
        k_length,
        d_length,
        ma_type,
    )?;
    crate::write_wasm_object_f64_outputs(
        "volume_weighted_stochastic_rsi_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_stochastic_rsi_batch_output_into_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volume_weighted_stochastic_rsi_batch_js(source, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "volume_weighted_stochastic_rsi_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::cpu_batch::compute_cpu_batch;
    use crate::indicators::dispatch::{
        IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV, ParamValue,
    };

    fn sample_source() -> Vec<f64> {
        (0..160)
            .map(|i| 100.0 + i as f64 * 0.4 + ((i % 9) as f64 - 4.0) * 0.35)
            .collect()
    }

    fn sample_volume() -> Vec<f64> {
        (0..160)
            .map(|i| 900.0 + (i % 13) as f64 * 37.0 + (i % 5) as f64 * 11.0)
            .collect()
    }

    fn naive_weighted_rsi(source: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; source.len()];
        if source.len() <= period {
            return out;
        }
        let mut gain_sum = 0.0;
        let mut loss_sum = 0.0;
        for i in 1..=period {
            let change = source[i] - source[i - 1];
            gain_sum += change.max(0.0) * volume[i];
            loss_sum += (-change).max(0.0) * volume[i];
        }
        let mut avg_gain = gain_sum / period as f64;
        let mut avg_loss = loss_sum / period as f64;
        out[period] = rsi_from_avgs(avg_gain, avg_loss);
        for i in (period + 1)..source.len() {
            let change = source[i] - source[i - 1];
            let gain = change.max(0.0) * volume[i];
            let loss = (-change).max(0.0) * volume[i];
            avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
            out[i] = rsi_from_avgs(avg_gain, avg_loss);
        }
        out
    }

    fn naive_stoch(values: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; values.len()];
        for i in 0..values.len() {
            if i + 1 < period {
                continue;
            }
            let window = &values[(i + 1 - period)..=i];
            if window.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let mut low = f64::INFINITY;
            let mut high = f64::NEG_INFINITY;
            for &value in window {
                low = low.min(value);
                high = high.max(value);
            }
            let denom = high - low;
            if denom != 0.0 {
                out[i] = (values[i] - low) / denom * 100.0;
            }
        }
        out
    }

    fn naive_ma(values: &[f64], volume: &[f64], period: usize, ma_type: VwsrsiMaType) -> Vec<f64> {
        let len = values.len();
        let mut out = vec![f64::NAN; len];
        match ma_type {
            VwsrsiMaType::Ema => {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                let mut initialized = false;
                let alpha = 2.0 / (period as f64 + 1.0);
                for i in 0..len {
                    let value = values[i];
                    if !value.is_finite() {
                        continue;
                    }
                    let w = volume[i];
                    if !initialized {
                        numerator = value * w;
                        denominator = w;
                        initialized = true;
                    } else {
                        let beta = 1.0 - alpha;
                        numerator = alpha * value * w + beta * numerator;
                        denominator = alpha * w + beta * denominator;
                    }
                    if denominator != 0.0 {
                        out[i] = numerator / denominator;
                    }
                }
            }
            VwsrsiMaType::Wsma => {
                let mut numerator_sum = 0.0;
                let mut denominator_sum = 0.0;
                let mut count = 0usize;
                let mut numerator_avg = 0.0;
                let mut denominator_avg = 0.0;
                let mut initialized = false;
                for i in 0..len {
                    let value = values[i];
                    if !value.is_finite() {
                        continue;
                    }
                    let w = volume[i];
                    if !initialized {
                        numerator_sum += value * w;
                        denominator_sum += w;
                        count += 1;
                        if count == period {
                            numerator_avg = numerator_sum / period as f64;
                            denominator_avg = denominator_sum / period as f64;
                            initialized = true;
                            if denominator_avg != 0.0 {
                                out[i] = numerator_avg / denominator_avg;
                            }
                        }
                    } else {
                        numerator_avg =
                            (numerator_avg * (period as f64 - 1.0) + value * w) / period as f64;
                        denominator_avg =
                            (denominator_avg * (period as f64 - 1.0) + w) / period as f64;
                        if denominator_avg != 0.0 {
                            out[i] = numerator_avg / denominator_avg;
                        }
                    }
                }
            }
            VwsrsiMaType::Sma | VwsrsiMaType::Vwma => {
                for i in 0..len {
                    if i + 1 < period {
                        continue;
                    }
                    let window = &values[(i + 1 - period)..=i];
                    if window.iter().any(|v| !v.is_finite()) {
                        continue;
                    }
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;
                    for j in 0..period {
                        let w = volume[i + 1 - period + j];
                        numerator += window[j] * w;
                        denominator += w;
                    }
                    if denominator != 0.0 {
                        out[i] = numerator / denominator;
                    }
                }
            }
            VwsrsiMaType::Wma => {
                for i in 0..len {
                    if i + 1 < period {
                        continue;
                    }
                    let window = &values[(i + 1 - period)..=i];
                    if window.iter().any(|v| !v.is_finite()) {
                        continue;
                    }
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;
                    for j in 0..period {
                        let w = volume[i + 1 - period + j];
                        let pos = (j + 1) as f64;
                        numerator += window[j] * w * pos;
                        denominator += w * pos;
                    }
                    if denominator != 0.0 {
                        out[i] = numerator / denominator;
                    }
                }
            }
        }
        out
    }

    fn naive_indicator(
        source: &[f64],
        volume: &[f64],
        rsi_length: usize,
        stoch_length: usize,
        k_length: usize,
        d_length: usize,
        ma_type: VwsrsiMaType,
    ) -> (Vec<f64>, Vec<f64>) {
        let rsi = naive_weighted_rsi(source, volume, rsi_length);
        let stoch = naive_stoch(&rsi, stoch_length);
        let k = naive_ma(&stoch, volume, k_length, ma_type);
        let d = naive_ma(&k, volume, d_length, ma_type);
        (k, d)
    }

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            let lhs = a[i];
            let rhs = b[i];
            if lhs.is_nan() || rhs.is_nan() {
                assert!(
                    lhs.is_nan() && rhs.is_nan(),
                    "nan mismatch at {i}: {lhs} vs {rhs}"
                );
            } else {
                assert!(
                    (lhs - rhs).abs() <= 1e-10,
                    "mismatch at {i}: {lhs} vs {rhs}"
                );
            }
        }
    }

    #[test]
    fn volume_weighted_stochastic_rsi_matches_naive() {
        let source = sample_source();
        let volume = sample_volume();
        let input = VolumeWeightedStochasticRsiInput::from_slices(
            &source,
            &volume,
            VolumeWeightedStochasticRsiParams {
                rsi_length: Some(14),
                stoch_length: Some(14),
                k_length: Some(3),
                d_length: Some(3),
                ma_type: Some("WSMA".to_string()),
            },
        );
        let out = volume_weighted_stochastic_rsi(&input).expect("indicator");
        let (k_ref, d_ref) = naive_indicator(&source, &volume, 14, 14, 3, 3, VwsrsiMaType::Wsma);
        assert_close(&out.k, &k_ref);
        assert_close(&out.d, &d_ref);
    }

    #[test]
    fn volume_weighted_stochastic_rsi_into_matches_api() {
        let source = sample_source();
        let volume = sample_volume();
        let input = VolumeWeightedStochasticRsiInput::from_slices(
            &source,
            &volume,
            VolumeWeightedStochasticRsiParams {
                rsi_length: Some(10),
                stoch_length: Some(12),
                k_length: Some(4),
                d_length: Some(3),
                ma_type: Some("EMA".to_string()),
            },
        );
        let baseline = volume_weighted_stochastic_rsi(&input).expect("baseline");
        let mut k = vec![0.0; source.len()];
        let mut d = vec![0.0; source.len()];
        volume_weighted_stochastic_rsi_into(&input, &mut k, &mut d).expect("into");
        assert_close(&baseline.k, &k);
        assert_close(&baseline.d, &d);
    }

    #[test]
    fn volume_weighted_stochastic_rsi_stream_matches_batch() {
        let source = sample_source();
        let volume = sample_volume();
        let input = VolumeWeightedStochasticRsiInput::from_slices(
            &source,
            &volume,
            VolumeWeightedStochasticRsiParams {
                rsi_length: Some(14),
                stoch_length: Some(10),
                k_length: Some(5),
                d_length: Some(4),
                ma_type: Some("WMA".to_string()),
            },
        );
        let batch = volume_weighted_stochastic_rsi(&input).expect("batch");
        let mut stream =
            VolumeWeightedStochasticRsiStream::try_new(input.params.clone()).expect("stream");
        let mut k = Vec::with_capacity(source.len());
        let mut d = Vec::with_capacity(source.len());
        for i in 0..source.len() {
            match stream.update(source[i], volume[i]) {
                Some((kv, dv)) => {
                    k.push(kv);
                    d.push(dv);
                }
                None => {
                    k.push(f64::NAN);
                    d.push(f64::NAN);
                }
            }
        }
        assert_close(&batch.k, &k);
        assert_close(&batch.d, &d);
    }

    #[test]
    fn volume_weighted_stochastic_rsi_batch_single_param_matches_single() {
        let source = sample_source();
        let volume = sample_volume();
        let sweep = VolumeWeightedStochasticRsiBatchRange {
            rsi_length: (14, 14, 0),
            stoch_length: (14, 14, 0),
            k_length: (3, 3, 0),
            d_length: (3, 3, 0),
            ma_type: "VWMA".to_string(),
        };
        let batch = volume_weighted_stochastic_rsi_batch_with_kernel(
            &source,
            &volume,
            &sweep,
            Kernel::ScalarBatch,
        )
        .expect("batch");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, source.len());
        let single =
            volume_weighted_stochastic_rsi(&VolumeWeightedStochasticRsiInput::from_slices(
                &source,
                &volume,
                VolumeWeightedStochasticRsiParams {
                    rsi_length: Some(14),
                    stoch_length: Some(14),
                    k_length: Some(3),
                    d_length: Some(3),
                    ma_type: Some("VWMA".to_string()),
                },
            ))
            .expect("single");
        assert_close(&batch.k, &single.k);
        assert_close(&batch.d, &single.d);
    }

    #[test]
    fn volume_weighted_stochastic_rsi_rejects_invalid_ma_type() {
        let source = sample_source();
        let volume = sample_volume();
        let input = VolumeWeightedStochasticRsiInput::from_slices(
            &source,
            &volume,
            VolumeWeightedStochasticRsiParams {
                rsi_length: Some(14),
                stoch_length: Some(14),
                k_length: Some(3),
                d_length: Some(3),
                ma_type: Some("BAD".to_string()),
            },
        );
        let err = volume_weighted_stochastic_rsi(&input).expect_err("invalid ma type");
        assert!(matches!(
            err,
            VolumeWeightedStochasticRsiError::InvalidMaType { .. }
        ));
    }

    #[test]
    fn volume_weighted_stochastic_rsi_dispatch_d_output_matches_direct() {
        let source = sample_source();
        let volume = sample_volume();
        let params = [
            ParamKV {
                key: "rsi_length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "stoch_length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "k_length",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "d_length",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "ma_type",
                value: ParamValue::EnumString("WSMA"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "volume_weighted_stochastic_rsi",
            output_id: Some("d"),
            data: IndicatorDataRef::CloseVolume {
                close: &source,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })
        .expect("dispatch");
        let direct =
            volume_weighted_stochastic_rsi(&VolumeWeightedStochasticRsiInput::from_slices(
                &source,
                &volume,
                VolumeWeightedStochasticRsiParams {
                    rsi_length: Some(14),
                    stoch_length: Some(14),
                    k_length: Some(3),
                    d_length: Some(3),
                    ma_type: Some("WSMA".to_string()),
                },
            ))
            .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, source.len());
        let values = out.values_f64.expect("f64 values");
        assert_close(&values, &direct.d);
    }
}
