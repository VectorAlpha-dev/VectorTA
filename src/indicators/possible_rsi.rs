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

use crate::indicators::rsi::{rsi_into_slice, RsiInput, RsiParams};
use crate::indicators::rsx::{rsx_into_slice, RsxInput, RsxParams};
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

const DEFAULT_SOURCE: &str = "close";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PossibleRsiMode {
    Rsx,
    Regular,
    Slow,
    Rapid,
    Harris,
    Cutler,
    EhlersSmoothed,
}

impl PossibleRsiMode {
    #[inline(always)]
    fn from_str(value: &str) -> Result<Self, PossibleRsiError> {
        if value.eq_ignore_ascii_case("rsx") {
            return Ok(Self::Rsx);
        }
        if value.eq_ignore_ascii_case("regular") {
            return Ok(Self::Regular);
        }
        if value.eq_ignore_ascii_case("slow") {
            return Ok(Self::Slow);
        }
        if value.eq_ignore_ascii_case("rapid") {
            return Ok(Self::Rapid);
        }
        if value.eq_ignore_ascii_case("harris") {
            return Ok(Self::Harris);
        }
        if value.eq_ignore_ascii_case("cutler") || value.eq_ignore_ascii_case("cuttler") {
            return Ok(Self::Cutler);
        }
        if value.eq_ignore_ascii_case("ehlers_smoothed")
            || value.eq_ignore_ascii_case("ehlers-smoothed")
            || value.eq_ignore_ascii_case("ehlers smoothed")
        {
            return Ok(Self::EhlersSmoothed);
        }
        Err(PossibleRsiError::InvalidRsiMode {
            rsi_mode: value.to_string(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PossibleRsiNormalizationMode {
    GaussianFisher,
    Softmax,
    RegularNorm,
}

impl PossibleRsiNormalizationMode {
    #[inline(always)]
    fn from_str(value: &str) -> Result<Self, PossibleRsiError> {
        if value.eq_ignore_ascii_case("gaussian_fisher")
            || value.eq_ignore_ascii_case("gaussian")
            || value.eq_ignore_ascii_case("gaussian (fisher)")
            || value.eq_ignore_ascii_case("fisher")
        {
            return Ok(Self::GaussianFisher);
        }
        if value.eq_ignore_ascii_case("softmax") {
            return Ok(Self::Softmax);
        }
        if value.eq_ignore_ascii_case("regular_norm")
            || value.eq_ignore_ascii_case("regular norm")
            || value.eq_ignore_ascii_case("regnorm")
        {
            return Ok(Self::RegularNorm);
        }
        Err(PossibleRsiError::InvalidNormalizationMode {
            normalization_mode: value.to_string(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PossibleRsiSignalType {
    Slope,
    DynamicMiddleCrossover,
    LevelsCrossover,
    ZerolineCrossover,
}

impl PossibleRsiSignalType {
    #[inline(always)]
    fn from_str(value: &str) -> Result<Self, PossibleRsiError> {
        if value.eq_ignore_ascii_case("slope") {
            return Ok(Self::Slope);
        }
        if value.eq_ignore_ascii_case("dynamic_middle_crossover")
            || value.eq_ignore_ascii_case("dynamic middle crossover")
        {
            return Ok(Self::DynamicMiddleCrossover);
        }
        if value.eq_ignore_ascii_case("levels_crossover")
            || value.eq_ignore_ascii_case("levels crossover")
        {
            return Ok(Self::LevelsCrossover);
        }
        if value.eq_ignore_ascii_case("zeroline_crossover")
            || value.eq_ignore_ascii_case("zeroline crossover")
            || value.eq_ignore_ascii_case("zero_line_crossover")
        {
            return Ok(Self::ZerolineCrossover);
        }
        Err(PossibleRsiError::InvalidSignalType {
            signal_type: value.to_string(),
        })
    }
}

#[derive(Debug, Clone)]
pub enum PossibleRsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PossibleRsiOutput {
    pub value: Vec<f64>,
    pub buy_level: Vec<f64>,
    pub sell_level: Vec<f64>,
    pub middle_level: Vec<f64>,
    pub state: Vec<f64>,
    pub long_signal: Vec<f64>,
    pub short_signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct PossibleRsiParams {
    pub period: Option<usize>,
    pub rsi_mode: Option<String>,
    pub norm_period: Option<usize>,
    pub normalization_mode: Option<String>,
    pub normalization_length: Option<usize>,
    pub nonlag_period: Option<usize>,
    pub dynamic_zone_period: Option<usize>,
    pub buy_probability: Option<f64>,
    pub sell_probability: Option<f64>,
    pub signal_type: Option<String>,
    pub run_highpass: Option<bool>,
    pub highpass_period: Option<usize>,
}

impl Default for PossibleRsiParams {
    fn default() -> Self {
        Self {
            period: Some(32),
            rsi_mode: Some("regular".to_string()),
            norm_period: Some(100),
            normalization_mode: Some("gaussian_fisher".to_string()),
            normalization_length: Some(15),
            nonlag_period: Some(15),
            dynamic_zone_period: Some(20),
            buy_probability: Some(0.2),
            sell_probability: Some(0.2),
            signal_type: Some("zeroline_crossover".to_string()),
            run_highpass: Some(false),
            highpass_period: Some(15),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PossibleRsiInput<'a> {
    pub data: PossibleRsiData<'a>,
    pub params: PossibleRsiParams,
}

impl<'a> PossibleRsiInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: PossibleRsiParams) -> Self {
        Self {
            data: PossibleRsiData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: PossibleRsiParams) -> Self {
        Self {
            data: PossibleRsiData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DEFAULT_SOURCE, PossibleRsiParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PossibleRsiBuilder {
    period: Option<usize>,
    rsi_mode: Option<&'static str>,
    norm_period: Option<usize>,
    normalization_mode: Option<&'static str>,
    normalization_length: Option<usize>,
    nonlag_period: Option<usize>,
    dynamic_zone_period: Option<usize>,
    buy_probability: Option<f64>,
    sell_probability: Option<f64>,
    signal_type: Option<&'static str>,
    run_highpass: Option<bool>,
    highpass_period: Option<usize>,
    kernel: Kernel,
}

impl Default for PossibleRsiBuilder {
    fn default() -> Self {
        Self {
            period: None,
            rsi_mode: None,
            norm_period: None,
            normalization_mode: None,
            normalization_length: None,
            nonlag_period: None,
            dynamic_zone_period: None,
            buy_probability: None,
            sell_probability: None,
            signal_type: None,
            run_highpass: None,
            highpass_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PossibleRsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, value: usize) -> Self {
        self.period = Some(value);
        self
    }

    #[inline(always)]
    pub fn rsi_mode(mut self, value: &'static str) -> Self {
        self.rsi_mode = Some(value);
        self
    }

    #[inline(always)]
    pub fn norm_period(mut self, value: usize) -> Self {
        self.norm_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn normalization_mode(mut self, value: &'static str) -> Self {
        self.normalization_mode = Some(value);
        self
    }

    #[inline(always)]
    pub fn normalization_length(mut self, value: usize) -> Self {
        self.normalization_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn nonlag_period(mut self, value: usize) -> Self {
        self.nonlag_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn dynamic_zone_period(mut self, value: usize) -> Self {
        self.dynamic_zone_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn buy_probability(mut self, value: f64) -> Self {
        self.buy_probability = Some(value);
        self
    }

    #[inline(always)]
    pub fn sell_probability(mut self, value: f64) -> Self {
        self.sell_probability = Some(value);
        self
    }

    #[inline(always)]
    pub fn signal_type(mut self, value: &'static str) -> Self {
        self.signal_type = Some(value);
        self
    }

    #[inline(always)]
    pub fn run_highpass(mut self, value: bool) -> Self {
        self.run_highpass = Some(value);
        self
    }

    #[inline(always)]
    pub fn highpass_period(mut self, value: usize) -> Self {
        self.highpass_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    fn build_params(self) -> PossibleRsiParams {
        PossibleRsiParams {
            period: self.period,
            rsi_mode: self.rsi_mode.map(str::to_string),
            norm_period: self.norm_period,
            normalization_mode: self.normalization_mode.map(str::to_string),
            normalization_length: self.normalization_length,
            nonlag_period: self.nonlag_period,
            dynamic_zone_period: self.dynamic_zone_period,
            buy_probability: self.buy_probability,
            sell_probability: self.sell_probability,
            signal_type: self.signal_type.map(str::to_string),
            run_highpass: self.run_highpass,
            highpass_period: self.highpass_period,
        }
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<PossibleRsiOutput, PossibleRsiError> {
        self.apply_candles(candles, DEFAULT_SOURCE)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<PossibleRsiOutput, PossibleRsiError> {
        possible_rsi_with_kernel(
            &PossibleRsiInput::from_candles(candles, source, self.build_params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<PossibleRsiOutput, PossibleRsiError> {
        possible_rsi_with_kernel(
            &PossibleRsiInput::from_slice(data, self.build_params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<PossibleRsiStream, PossibleRsiError> {
        PossibleRsiStream::try_new(self.build_params())
    }
}

#[derive(Debug, Error)]
pub enum PossibleRsiError {
    #[error("possible_rsi: Input data slice is empty.")]
    EmptyInputData,
    #[error("possible_rsi: All values are NaN.")]
    AllValuesNaN,
    #[error("possible_rsi: Invalid period: {period}")]
    InvalidPeriod { period: usize },
    #[error("possible_rsi: Invalid norm_period: {norm_period}")]
    InvalidNormPeriod { norm_period: usize },
    #[error("possible_rsi: Invalid normalization_length: {normalization_length}")]
    InvalidNormalizationLength { normalization_length: usize },
    #[error("possible_rsi: Invalid nonlag_period: {nonlag_period}")]
    InvalidNonlagPeriod { nonlag_period: usize },
    #[error("possible_rsi: Invalid dynamic_zone_period: {dynamic_zone_period}")]
    InvalidDynamicZonePeriod { dynamic_zone_period: usize },
    #[error("possible_rsi: Invalid highpass_period: {highpass_period}")]
    InvalidHighpassPeriod { highpass_period: usize },
    #[error("possible_rsi: Invalid buy_probability: {buy_probability}")]
    InvalidBuyProbability { buy_probability: f64 },
    #[error("possible_rsi: Invalid sell_probability: {sell_probability}")]
    InvalidSellProbability { sell_probability: f64 },
    #[error("possible_rsi: Invalid RSI mode: {rsi_mode}")]
    InvalidRsiMode { rsi_mode: String },
    #[error("possible_rsi: Invalid normalization_mode: {normalization_mode}")]
    InvalidNormalizationMode { normalization_mode: String },
    #[error("possible_rsi: Invalid signal_type: {signal_type}")]
    InvalidSignalType { signal_type: String },
    #[error("possible_rsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("possible_rsi: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("possible_rsi: Invalid range: {field} start={start} end={end} step={step}")]
    InvalidRange {
        field: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("possible_rsi: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("possible_rsi: Output length mismatch: dst = {dst_len}, expected = {expected_len}")]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("possible_rsi: Invalid input: {msg}")]
    InvalidInput { msg: String },
    #[error("possible_rsi: RSI helper failed: {details}")]
    RsiFailure { details: String },
    #[error("possible_rsi: RSX helper failed: {details}")]
    RsxFailure { details: String },
}

#[derive(Debug, Clone, Copy)]
struct PossibleRsiResolved {
    period: usize,
    rsi_mode: PossibleRsiMode,
    norm_period: usize,
    normalization_mode: PossibleRsiNormalizationMode,
    normalization_length: usize,
    nonlag_period: usize,
    dynamic_zone_period: usize,
    buy_probability: f64,
    sell_probability: f64,
    signal_type: PossibleRsiSignalType,
    run_highpass: bool,
    highpass_period: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct PossibleRsiPoint {
    pub value: f64,
    pub buy_level: f64,
    pub sell_level: f64,
    pub middle_level: f64,
    pub state: f64,
    pub long_signal: f64,
    pub short_signal: f64,
}

#[derive(Debug, Clone)]
pub struct PossibleRsiStream {
    params: PossibleRsiParams,
    history: Vec<f64>,
}

impl PossibleRsiStream {
    #[inline(always)]
    pub fn try_new(params: PossibleRsiParams) -> Result<Self, PossibleRsiError> {
        let _ = resolve_params(&params)?;
        Ok(Self {
            params,
            history: Vec::new(),
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.history.clear();
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<PossibleRsiPoint> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        self.history.push(value);
        let out = possible_rsi(&PossibleRsiInput::from_slice(
            &self.history,
            self.params.clone(),
        ))
        .ok()?;
        let i = self.history.len().saturating_sub(1);
        let point = PossibleRsiPoint {
            value: out.value[i],
            buy_level: out.buy_level[i],
            sell_level: out.sell_level[i],
            middle_level: out.middle_level[i],
            state: out.state[i],
            long_signal: out.long_signal[i],
            short_signal: out.short_signal[i],
        };
        if point.value.is_finite()
            && point.buy_level.is_finite()
            && point.sell_level.is_finite()
            && point.middle_level.is_finite()
            && point.state.is_finite()
        {
            Some(point)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        resolve_params(&self.params)
            .map(estimated_warmup)
            .unwrap_or(0)
    }
}

#[inline(always)]
fn input_slice<'a>(input: &'a PossibleRsiInput<'a>) -> &'a [f64] {
    match &input.data {
        PossibleRsiData::Candles { candles, source } => match *source {
            "open" => candles.open.as_slice(),
            "high" => candles.high.as_slice(),
            "low" => candles.low.as_slice(),
            "close" => candles.close.as_slice(),
            "volume" => candles.volume.as_slice(),
            _ => source_type(candles, source),
        },
        PossibleRsiData::Slice(values) => values,
    }
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for &value in data {
        if value.is_finite() {
            current += 1;
            if current > best {
                best = current;
            }
        } else {
            current = 0;
        }
    }
    best
}

#[inline(always)]
fn nonlag_kernel_len(period: usize) -> usize {
    period.saturating_mul(5).saturating_sub(1)
}

#[inline(always)]
fn estimated_warmup(params: PossibleRsiResolved) -> usize {
    params
        .period
        .saturating_add(params.norm_period.saturating_sub(1))
        .saturating_add(params.normalization_length.saturating_sub(1))
        .saturating_add(nonlag_kernel_len(params.nonlag_period).saturating_sub(1))
        .saturating_add(params.dynamic_zone_period.saturating_sub(1))
}

#[inline(always)]
fn resolve_params(params: &PossibleRsiParams) -> Result<PossibleRsiResolved, PossibleRsiError> {
    let period = params.period.unwrap_or(32);
    if period == 0 {
        return Err(PossibleRsiError::InvalidPeriod { period });
    }
    let norm_period = params.norm_period.unwrap_or(100);
    if norm_period == 0 {
        return Err(PossibleRsiError::InvalidNormPeriod { norm_period });
    }
    let normalization_length = params.normalization_length.unwrap_or(15);
    if normalization_length == 0 {
        return Err(PossibleRsiError::InvalidNormalizationLength {
            normalization_length,
        });
    }
    let nonlag_period = params.nonlag_period.unwrap_or(15);
    if nonlag_period == 0 {
        return Err(PossibleRsiError::InvalidNonlagPeriod { nonlag_period });
    }
    let dynamic_zone_period = params.dynamic_zone_period.unwrap_or(20);
    if dynamic_zone_period == 0 {
        return Err(PossibleRsiError::InvalidDynamicZonePeriod {
            dynamic_zone_period,
        });
    }
    let buy_probability = params.buy_probability.unwrap_or(0.2);
    if !buy_probability.is_finite() || !(0.0..=0.5).contains(&buy_probability) {
        return Err(PossibleRsiError::InvalidBuyProbability { buy_probability });
    }
    let sell_probability = params.sell_probability.unwrap_or(0.2);
    if !sell_probability.is_finite() || !(0.0..=0.5).contains(&sell_probability) {
        return Err(PossibleRsiError::InvalidSellProbability { sell_probability });
    }
    let highpass_period = params.highpass_period.unwrap_or(15);
    if highpass_period == 0 {
        return Err(PossibleRsiError::InvalidHighpassPeriod { highpass_period });
    }
    Ok(PossibleRsiResolved {
        period,
        rsi_mode: PossibleRsiMode::from_str(params.rsi_mode.as_deref().unwrap_or("regular"))?,
        norm_period,
        normalization_mode: PossibleRsiNormalizationMode::from_str(
            params
                .normalization_mode
                .as_deref()
                .unwrap_or("gaussian_fisher"),
        )?,
        normalization_length,
        nonlag_period,
        dynamic_zone_period,
        buy_probability,
        sell_probability,
        signal_type: PossibleRsiSignalType::from_str(
            params
                .signal_type
                .as_deref()
                .unwrap_or("zeroline_crossover"),
        )?,
        run_highpass: params.run_highpass.unwrap_or(false),
        highpass_period,
    })
}
#[inline(always)]
fn validate_common(data: &[f64], params: PossibleRsiResolved) -> Result<(), PossibleRsiError> {
    if data.is_empty() {
        return Err(PossibleRsiError::EmptyInputData);
    }
    let valid = longest_valid_run(data);
    if valid == 0 {
        return Err(PossibleRsiError::AllValuesNaN);
    }
    let needed = estimated_warmup(params).saturating_add(1);
    if valid < needed {
        return Err(PossibleRsiError::NotEnoughValidData { needed, valid });
    }
    Ok(())
}

#[inline(always)]
fn for_each_finite_segment<F>(data: &[f64], mut f: F)
where
    F: FnMut(usize, usize),
{
    let mut start = 0usize;
    while start < data.len() {
        while start < data.len() && !data[start].is_finite() {
            start += 1;
        }
        if start >= data.len() {
            break;
        }
        let mut end = start + 1;
        while end < data.len() && data[end].is_finite() {
            end += 1;
        }
        f(start, end);
        start = end;
    }
}

#[inline(always)]
fn highpass_series(data: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    let a1 = (-1.414 * std::f64::consts::PI / period as f64).exp();
    let b1 = 2.0 * a1 * (1.414 * std::f64::consts::PI / period as f64).cos();
    let c2 = b1;
    let c3 = -a1 * a1;
    let c1 = (1.0 + c2 - c3) / 4.0;
    for_each_finite_segment(data, |start, end| {
        let mut hp1 = 0.0;
        let mut hp2 = 0.0;
        for i in start..end {
            if i - start < 4 {
                out[i] = 0.0;
                hp2 = hp1;
                hp1 = 0.0;
                continue;
            }
            let hp = c1 * (data[i] - 2.0 * data[i - 1] + data[i - 2]) + c2 * hp1 + c3 * hp2;
            out[i] = hp;
            hp2 = hp1;
            hp1 = hp;
        }
    });
    out
}

#[inline(always)]
fn cutler_rsi_series(data: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    for_each_finite_segment(data, |start, end| {
        if end - start <= period {
            return;
        }
        let mut gain = 0.0;
        let mut loss = 0.0;
        for i in (start + 1)..=(start + period) {
            let diff = data[i] - data[i - 1];
            if diff > 0.0 {
                gain += diff;
            } else {
                loss += -diff;
            }
        }
        out[start + period] = if gain + loss == 0.0 {
            50.0
        } else {
            100.0 * gain / (gain + loss)
        };
        for i in (start + period + 1)..end {
            let old_diff = data[i - period] - data[i - period - 1];
            if old_diff > 0.0 {
                gain -= old_diff;
            } else {
                loss -= -old_diff;
            }
            let new_diff = data[i] - data[i - 1];
            if new_diff > 0.0 {
                gain += new_diff;
            } else {
                loss += -new_diff;
            }
            out[i] = if gain + loss == 0.0 {
                50.0
            } else {
                100.0 * gain / (gain + loss)
            };
        }
    });
    out
}

#[inline(always)]
fn harris_rsi_series(data: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    for_each_finite_segment(data, |start, end| {
        if end - start <= period {
            return;
        }
        for i in (start + period)..end {
            let current = data[i];
            let mut up = 0.0;
            let mut down = 0.0;
            for j in 1..=period {
                let diff = current - data[i - j];
                if diff > 0.0 {
                    up += diff;
                } else {
                    down += -diff;
                }
            }
            out[i] = if up + down == 0.0 {
                50.0
            } else {
                100.0 * up / (up + down)
            };
        }
    });
    out
}

#[inline(always)]
fn ehlers_smoothed_rsi_series(data: &[f64], period: usize) -> Vec<f64> {
    let mut smooth = vec![f64::NAN; data.len()];
    for_each_finite_segment(data, |start, end| {
        if end - start < 4 {
            return;
        }
        for i in (start + 3)..end {
            smooth[i] = (data[i] + 2.0 * data[i - 1] + 2.0 * data[i - 2] + data[i - 3]) / 6.0;
        }
    });
    cutler_rsi_series(&smooth, period)
}

#[inline(always)]
fn ema_valid_series(data: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut state = None;
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            state = None;
            continue;
        }
        let next = match state {
            Some(prev) => prev + alpha * (value - prev),
            None => value,
        };
        out[i] = next;
        state = Some(next);
    }
    out
}

#[inline(always)]
fn compute_rsi_series(
    data: &[f64],
    mode: PossibleRsiMode,
    period: usize,
) -> Result<Vec<f64>, PossibleRsiError> {
    match mode {
        PossibleRsiMode::Regular => {
            let mut out = vec![f64::NAN; data.len()];
            let input = RsiInput::from_slice(
                data,
                RsiParams {
                    period: Some(period),
                },
            );
            rsi_into_slice(&mut out, &input, Kernel::Auto).map_err(|e| {
                PossibleRsiError::RsiFailure {
                    details: e.to_string(),
                }
            })?;
            Ok(out)
        }
        PossibleRsiMode::Rsx => {
            let mut out = vec![f64::NAN; data.len()];
            let input = RsxInput::from_slice(
                data,
                RsxParams {
                    period: Some(period),
                },
            );
            rsx_into_slice(&mut out, &input, Kernel::Auto).map_err(|e| {
                PossibleRsiError::RsxFailure {
                    details: e.to_string(),
                }
            })?;
            Ok(out)
        }
        PossibleRsiMode::Cutler | PossibleRsiMode::Rapid => Ok(cutler_rsi_series(data, period)),
        PossibleRsiMode::Slow => Ok(ema_valid_series(
            &compute_rsi_series(data, PossibleRsiMode::Regular, period)?,
            period,
        )),
        PossibleRsiMode::Harris => Ok(harris_rsi_series(data, period)),
        PossibleRsiMode::EhlersSmoothed => Ok(ehlers_smoothed_rsi_series(data, period)),
    }
}

#[inline(always)]
fn rolling_min_max(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mins = vec![f64::NAN; data.len()];
    let mut maxs = vec![f64::NAN; data.len()];
    for_each_finite_segment(data, |start, end| {
        if end - start < period {
            return;
        }
        let mut min_q: std::collections::VecDeque<usize> =
            std::collections::VecDeque::with_capacity(period);
        let mut max_q: std::collections::VecDeque<usize> =
            std::collections::VecDeque::with_capacity(period);
        for i in start..end {
            while let Some(&front) = min_q.front() {
                if front + period <= i {
                    min_q.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&front) = max_q.front() {
                if front + period <= i {
                    max_q.pop_front();
                } else {
                    break;
                }
            }
            while let Some(&back) = min_q.back() {
                if data[back] >= data[i] {
                    min_q.pop_back();
                } else {
                    break;
                }
            }
            while let Some(&back) = max_q.back() {
                if data[back] <= data[i] {
                    max_q.pop_back();
                } else {
                    break;
                }
            }
            min_q.push_back(i);
            max_q.push_back(i);
            if i + 1 >= start + period {
                mins[i] = data[*min_q.front().unwrap()];
                maxs[i] = data[*max_q.front().unwrap()];
            }
        }
    });
    (mins, maxs)
}

#[inline(always)]
fn rolling_mean_std(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let mut means = vec![f64::NAN; data.len()];
    let mut stds = vec![f64::NAN; data.len()];
    for_each_finite_segment(data, |start, end| {
        if end - start < period {
            return;
        }
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for i in start..end {
            let value = data[i];
            sum += value;
            sumsq += value * value;
            if i >= start + period {
                let old = data[i - period];
                sum -= old;
                sumsq -= old * old;
            }
            if i + 1 >= start + period {
                let mean = sum / period as f64;
                let mut var = sumsq / period as f64 - mean * mean;
                if var < 0.0 {
                    var = 0.0;
                }
                means[i] = mean;
                stds[i] = var.sqrt();
            }
        }
    });
    (means, stds)
}

#[inline(always)]
fn fisher_transform_series(data: &[f64], period: usize) -> Vec<f64> {
    let (mins, maxs) = rolling_min_max(data, period);
    let mut out = vec![f64::NAN; data.len()];
    let mut prev_value = 0.0;
    let mut prev_fish = 0.0;
    let mut seeded = false;
    for i in 0..data.len() {
        let src = data[i];
        let low = mins[i];
        let high = maxs[i];
        if !src.is_finite()
            || !low.is_finite()
            || !high.is_finite()
            || (high - low).abs() <= f64::EPSILON
        {
            seeded = false;
            prev_value = 0.0;
            prev_fish = 0.0;
            continue;
        }
        let mut value = 0.66 * ((src - low) / (high - low) - 0.5)
            + 0.67 * if seeded { prev_value } else { 0.0 };
        if value > 0.99 {
            value = 0.999;
        }
        if value < -0.99 {
            value = -0.999;
        }
        let fish =
            0.5 * ((1.0 + value) / (1.0 - value)).ln() + 0.5 * if seeded { prev_fish } else { 0.0 };
        out[i] = fish;
        prev_value = value;
        prev_fish = fish;
        seeded = true;
    }
    out
}

#[inline(always)]
fn softmax_series(data: &[f64], period: usize) -> Vec<f64> {
    let (means, stds) = rolling_mean_std(data, period);
    let mut out = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        if !data[i].is_finite()
            || !means[i].is_finite()
            || !stds[i].is_finite()
            || stds[i] <= f64::EPSILON
        {
            continue;
        }
        let z = (data[i] - means[i]) / stds[i];
        let exp = (-z).exp();
        out[i] = (1.0 - exp) / (1.0 + exp);
    }
    out
}

#[inline(always)]
fn regular_norm_series(data: &[f64], period: usize) -> Vec<f64> {
    let (means, stds) = rolling_mean_std(data, period);
    let mut out = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        if !data[i].is_finite()
            || !means[i].is_finite()
            || !stds[i].is_finite()
            || stds[i] <= f64::EPSILON
        {
            continue;
        }
        out[i] = (data[i] - means[i]) / (stds[i] * 3.0);
    }
    out
}

#[inline(always)]
fn normalize_min_max(data: &[f64], period: usize) -> Vec<f64> {
    let (mins, maxs) = rolling_min_max(data, period);
    let mut out = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        if !data[i].is_finite()
            || !mins[i].is_finite()
            || !maxs[i].is_finite()
            || (maxs[i] - mins[i]).abs() <= f64::EPSILON
        {
            continue;
        }
        out[i] = 100.0 * (data[i] - mins[i]) / (maxs[i] - mins[i]);
    }
    out
}

#[inline(always)]
fn apply_secondary_normalization(
    data: &[f64],
    mode: PossibleRsiNormalizationMode,
    period: usize,
) -> Vec<f64> {
    match mode {
        PossibleRsiNormalizationMode::GaussianFisher => fisher_transform_series(data, period),
        PossibleRsiNormalizationMode::Softmax => softmax_series(data, period),
        PossibleRsiNormalizationMode::RegularNorm => regular_norm_series(data, period),
    }
}

#[inline(always)]
fn build_nonlag_weights(period: usize) -> (Vec<f64>, f64) {
    let cycle = 4.0;
    let coeff = 3.0 * std::f64::consts::PI;
    let phase = period as f64 - 1.0;
    let len = (period as f64 * cycle + phase) as usize;
    let mut weights = vec![0.0; len];
    let mut weight_sum = 0.0;
    for k in 0..len {
        let t = if phase > 1.0 && (k as f64) <= phase - 1.0 {
            k as f64 / (phase - 1.0)
        } else {
            1.0 + (k as f64 - phase + 1.0) * (2.0 * cycle - 1.0) / (cycle * period as f64 - 1.0)
        };
        let beta = (std::f64::consts::PI * t).cos();
        let mut g = 1.0 / (coeff * t + 1.0);
        if t <= 0.5 {
            g = 1.0;
        }
        let weight = g * beta;
        weights[k] = weight;
        weight_sum += weight;
    }
    (weights, weight_sum)
}

#[inline(always)]
fn nonlag_ma_series(data: &[f64], period: usize) -> Vec<f64> {
    let (weights, weight_sum) = build_nonlag_weights(period);
    let len = weights.len();
    let mut out = vec![f64::NAN; data.len()];
    for_each_finite_segment(data, |start, end| {
        if end - start < len {
            return;
        }
        for i in (start + len - 1)..end {
            let mut sum = 0.0;
            let mut valid = true;
            for (k, &weight) in weights.iter().enumerate() {
                let value = data[i - k];
                if !value.is_finite() {
                    valid = false;
                    break;
                }
                sum += weight * value;
            }
            if valid {
                out[i] = sum / weight_sum;
            }
        }
    });
    out
}

#[inline(always)]
fn percentile_nearest_rank(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    let rank = (probability * n as f64).ceil();
    let index = rank.max(1.0) as usize - 1;
    sorted[index.min(n - 1)]
}

#[inline(always)]
fn rolling_percentile_series(data: &[f64], period: usize, probability: f64) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    for_each_finite_segment(data, |start, end| {
        if end - start < period {
            return;
        }
        let mut sorted = data[start..(start + period)].to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out[start + period - 1] = percentile_nearest_rank(&sorted, probability);

        for i in (start + period)..end {
            let old = data[i - period];
            let remove_idx = sorted
                .binary_search_by(|probe| {
                    probe.partial_cmp(&old).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|idx| idx.min(sorted.len() - 1));
            sorted.remove(remove_idx);

            let new_value = data[i];
            let insert_idx = sorted
                .binary_search_by(|probe| {
                    probe
                        .partial_cmp(&new_value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|idx| idx);
            sorted.insert(insert_idx, new_value);
            out[i] = percentile_nearest_rank(&sorted, probability);
        }
    });
    out
}

#[inline(always)]
fn crossover(a_prev: f64, a: f64, b_prev: f64, b: f64) -> f64 {
    if a_prev.is_finite()
        && a.is_finite()
        && b_prev.is_finite()
        && b.is_finite()
        && a_prev <= b_prev
        && a > b
    {
        1.0
    } else {
        0.0
    }
}

#[inline(always)]
fn crossunder(a_prev: f64, a: f64, b_prev: f64, b: f64) -> f64 {
    if a_prev.is_finite()
        && a.is_finite()
        && b_prev.is_finite()
        && b.is_finite()
        && a_prev >= b_prev
        && a < b
    {
        1.0
    } else {
        0.0
    }
}

fn compute_possible_rsi_output(
    data: &[f64],
    params: PossibleRsiResolved,
) -> Result<PossibleRsiOutput, PossibleRsiError> {
    let (value, buy_level, sell_level, middle_level) = compute_possible_rsi_levels(data, params)?;
    let mut state = vec![f64::NAN; data.len()];
    let mut long_signal = vec![0.0; data.len()];
    let mut short_signal = vec![0.0; data.len()];

    fill_possible_rsi_signal_outputs(
        &value,
        &buy_level,
        &sell_level,
        &middle_level,
        params.signal_type,
        &mut state,
        &mut long_signal,
        &mut short_signal,
    );

    Ok(PossibleRsiOutput {
        value,
        buy_level,
        sell_level,
        middle_level,
        state,
        long_signal,
        short_signal,
    })
}

#[inline(always)]
fn compute_possible_rsi_levels(
    data: &[f64],
    params: PossibleRsiResolved,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), PossibleRsiError> {
    let source_storage;
    let source = if params.run_highpass {
        source_storage = highpass_series(data, params.highpass_period);
        source_storage.as_slice()
    } else {
        data
    };
    let rsi = compute_rsi_series(&source, params.rsi_mode, params.period)?;
    let scaled = normalize_min_max(&rsi, params.norm_period);
    let normalized = apply_secondary_normalization(
        &scaled,
        params.normalization_mode,
        params.normalization_length,
    );
    let value = nonlag_ma_series(&normalized, params.nonlag_period);
    let buy_level =
        rolling_percentile_series(&value, params.dynamic_zone_period, params.buy_probability);
    let sell_level = rolling_percentile_series(
        &value,
        params.dynamic_zone_period,
        1.0 - params.sell_probability,
    );
    let middle_level = rolling_percentile_series(&value, params.dynamic_zone_period, 0.5);

    Ok((value, buy_level, sell_level, middle_level))
}

#[inline(always)]
fn fill_possible_rsi_signal_outputs(
    value: &[f64],
    buy_level: &[f64],
    sell_level: &[f64],
    middle_level: &[f64],
    signal_type: PossibleRsiSignalType,
    state: &mut [f64],
    long_signal: &mut [f64],
    short_signal: &mut [f64],
) {
    state.fill(f64::NAN);
    long_signal.fill(0.0);
    short_signal.fill(0.0);

    for i in 0..value.len() {
        if !value[i].is_finite() {
            continue;
        }
        let signal_value = match signal_type {
            PossibleRsiSignalType::Slope => {
                if i == 0 || !value[i - 1].is_finite() {
                    continue;
                }
                value[i - 1]
            }
            PossibleRsiSignalType::DynamicMiddleCrossover => {
                if !middle_level[i].is_finite() {
                    continue;
                }
                middle_level[i]
            }
            PossibleRsiSignalType::LevelsCrossover => {
                if !buy_level[i].is_finite() || !sell_level[i].is_finite() {
                    continue;
                }
                f64::NAN
            }
            PossibleRsiSignalType::ZerolineCrossover => 0.0,
        };

        state[i] = match signal_type {
            PossibleRsiSignalType::Slope
            | PossibleRsiSignalType::DynamicMiddleCrossover
            | PossibleRsiSignalType::ZerolineCrossover => {
                if value[i] < signal_value {
                    -1.0
                } else if value[i] > signal_value {
                    1.0
                } else {
                    0.0
                }
            }
            PossibleRsiSignalType::LevelsCrossover => {
                if value[i] < buy_level[i] {
                    -1.0
                } else if value[i] > sell_level[i] {
                    1.0
                } else {
                    0.0
                }
            }
        };

        if i == 0 {
            continue;
        }

        match signal_type {
            PossibleRsiSignalType::Slope => {
                long_signal[i] = crossover(
                    value[i - 1],
                    value[i],
                    if i > 1 { value[i - 2] } else { value[i - 1] },
                    value[i - 1],
                );
                short_signal[i] = crossunder(
                    value[i - 1],
                    value[i],
                    if i > 1 { value[i - 2] } else { value[i - 1] },
                    value[i - 1],
                );
            }
            PossibleRsiSignalType::DynamicMiddleCrossover => {
                long_signal[i] =
                    crossover(value[i - 1], value[i], middle_level[i - 1], middle_level[i]);
                short_signal[i] =
                    crossunder(value[i - 1], value[i], middle_level[i - 1], middle_level[i]);
            }
            PossibleRsiSignalType::LevelsCrossover => {
                long_signal[i] =
                    crossover(value[i - 1], value[i], sell_level[i - 1], sell_level[i]);
                short_signal[i] =
                    crossunder(value[i - 1], value[i], buy_level[i - 1], buy_level[i]);
            }
            PossibleRsiSignalType::ZerolineCrossover => {
                long_signal[i] = crossover(value[i - 1], value[i], 0.0, 0.0);
                short_signal[i] = crossunder(value[i - 1], value[i], 0.0, 0.0);
            }
        }
    }
}

pub fn possible_rsi(input: &PossibleRsiInput) -> Result<PossibleRsiOutput, PossibleRsiError> {
    possible_rsi_with_kernel(input, Kernel::Auto)
}

pub fn possible_rsi_with_kernel(
    input: &PossibleRsiInput,
    kernel: Kernel,
) -> Result<PossibleRsiOutput, PossibleRsiError> {
    let data = input_slice(input);
    let params = resolve_params(&input.params)?;
    validate_common(data, params)?;
    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    compute_possible_rsi_output(data, params)
}

pub fn possible_rsi_into_slice(
    dst_value: &mut [f64],
    dst_buy_level: &mut [f64],
    dst_sell_level: &mut [f64],
    dst_middle_level: &mut [f64],
    dst_state: &mut [f64],
    dst_long_signal: &mut [f64],
    dst_short_signal: &mut [f64],
    input: &PossibleRsiInput,
    kernel: Kernel,
) -> Result<(), PossibleRsiError> {
    let data = input_slice(input);
    let params = resolve_params(&input.params)?;
    validate_common(data, params)?;
    if dst_value.len() != data.len()
        || dst_buy_level.len() != data.len()
        || dst_sell_level.len() != data.len()
        || dst_middle_level.len() != data.len()
        || dst_state.len() != data.len()
        || dst_long_signal.len() != data.len()
        || dst_short_signal.len() != data.len()
    {
        return Err(PossibleRsiError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_value
                .len()
                .min(dst_buy_level.len())
                .min(dst_sell_level.len())
                .min(dst_middle_level.len())
                .min(dst_state.len())
                .min(dst_long_signal.len())
                .min(dst_short_signal.len()),
        });
    }
    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let (value, buy_level, sell_level, middle_level) = compute_possible_rsi_levels(data, params)?;
    dst_value.copy_from_slice(&value);
    dst_buy_level.copy_from_slice(&buy_level);
    dst_sell_level.copy_from_slice(&sell_level);
    dst_middle_level.copy_from_slice(&middle_level);
    fill_possible_rsi_signal_outputs(
        &value,
        &buy_level,
        &sell_level,
        &middle_level,
        params.signal_type,
        dst_state,
        dst_long_signal,
        dst_short_signal,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn possible_rsi_into(
    input: &PossibleRsiInput,
    dst_value: &mut [f64],
    dst_buy_level: &mut [f64],
    dst_sell_level: &mut [f64],
    dst_middle_level: &mut [f64],
    dst_state: &mut [f64],
    dst_long_signal: &mut [f64],
    dst_short_signal: &mut [f64],
) -> Result<(), PossibleRsiError> {
    possible_rsi_into_slice(
        dst_value,
        dst_buy_level,
        dst_sell_level,
        dst_middle_level,
        dst_state,
        dst_long_signal,
        dst_short_signal,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, Copy)]
pub struct PossibleRsiBatchRange {
    pub period: (usize, usize, usize),
    pub norm_period: (usize, usize, usize),
    pub normalization_length: (usize, usize, usize),
    pub nonlag_period: (usize, usize, usize),
    pub dynamic_zone_period: (usize, usize, usize),
    pub buy_probability: (f64, f64, f64),
    pub sell_probability: (f64, f64, f64),
    pub highpass_period: (usize, usize, usize),
}

impl Default for PossibleRsiBatchRange {
    fn default() -> Self {
        Self {
            period: (32, 32, 0),
            norm_period: (100, 100, 0),
            normalization_length: (15, 15, 0),
            nonlag_period: (15, 15, 0),
            dynamic_zone_period: (20, 20, 0),
            buy_probability: (0.2, 0.2, 0.0),
            sell_probability: (0.2, 0.2, 0.0),
            highpass_period: (15, 15, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PossibleRsiBatchOutput {
    pub value: Vec<f64>,
    pub buy_level: Vec<f64>,
    pub sell_level: Vec<f64>,
    pub middle_level: Vec<f64>,
    pub state: Vec<f64>,
    pub long_signal: Vec<f64>,
    pub short_signal: Vec<f64>,
    pub combos: Vec<PossibleRsiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct PossibleRsiBatchBuilder {
    range: PossibleRsiBatchRange,
    rsi_mode: Option<&'static str>,
    normalization_mode: Option<&'static str>,
    signal_type: Option<&'static str>,
    run_highpass: Option<bool>,
    kernel: Kernel,
}

impl Default for PossibleRsiBatchBuilder {
    fn default() -> Self {
        Self {
            range: PossibleRsiBatchRange::default(),
            rsi_mode: None,
            normalization_mode: None,
            signal_type: None,
            run_highpass: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PossibleRsiBatchBuilder {
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
    pub fn rsi_mode(mut self, value: &'static str) -> Self {
        self.rsi_mode = Some(value);
        self
    }

    #[inline(always)]
    pub fn normalization_mode(mut self, value: &'static str) -> Self {
        self.normalization_mode = Some(value);
        self
    }

    #[inline(always)]
    pub fn signal_type(mut self, value: &'static str) -> Self {
        self.signal_type = Some(value);
        self
    }

    #[inline(always)]
    pub fn run_highpass(mut self, value: bool) -> Self {
        self.run_highpass = Some(value);
        self
    }

    #[inline(always)]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn norm_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.norm_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn normalization_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.normalization_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn nonlag_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.nonlag_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn dynamic_zone_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.dynamic_zone_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn buy_probability_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.buy_probability = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn sell_probability_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.sell_probability = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn highpass_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.highpass_period = (start, end, step);
        self
    }

    #[inline(always)]
    fn base_params(self) -> PossibleRsiParams {
        PossibleRsiParams {
            period: None,
            rsi_mode: self.rsi_mode.map(str::to_string),
            norm_period: None,
            normalization_mode: self.normalization_mode.map(str::to_string),
            normalization_length: None,
            nonlag_period: None,
            dynamic_zone_period: None,
            buy_probability: None,
            sell_probability: None,
            signal_type: self.signal_type.map(str::to_string),
            run_highpass: self.run_highpass,
            highpass_period: None,
        }
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<PossibleRsiBatchOutput, PossibleRsiError> {
        possible_rsi_batch_with_kernel(data, &self.range, &self.base_params(), self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<PossibleRsiBatchOutput, PossibleRsiError> {
        possible_rsi_batch_with_kernel(
            source_type(candles, source),
            &self.range,
            &self.base_params(),
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_usize_range(
    field: &'static str,
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, PossibleRsiError> {
    if start == 0 || end == 0 {
        return Err(PossibleRsiError::InvalidRange {
            field,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(PossibleRsiError::InvalidRange {
            field,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end {
            break;
        }
        let next = current.saturating_add(step);
        if next <= current {
            return Err(PossibleRsiError::InvalidRange {
                field,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current = next.min(end);
        if current == *out.last().unwrap() {
            break;
        }
    }
    Ok(out)
}

#[inline(always)]
fn expand_float_range(
    field: &'static str,
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, PossibleRsiError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(PossibleRsiError::InvalidRange {
            field,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(PossibleRsiError::InvalidRange {
            field,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end || (end - current).abs() <= 1e-12 {
            break;
        }
        let next = current + step;
        if next <= current {
            return Err(PossibleRsiError::InvalidRange {
                field,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current = if next > end { end } else { next };
    }
    Ok(out)
}

fn expand_grid_checked(
    range: &PossibleRsiBatchRange,
    base: &PossibleRsiParams,
) -> Result<Vec<PossibleRsiParams>, PossibleRsiError> {
    let periods = expand_usize_range("period", range.period.0, range.period.1, range.period.2)?;
    let norm_periods = expand_usize_range(
        "norm_period",
        range.norm_period.0,
        range.norm_period.1,
        range.norm_period.2,
    )?;
    let normalization_lengths = expand_usize_range(
        "normalization_length",
        range.normalization_length.0,
        range.normalization_length.1,
        range.normalization_length.2,
    )?;
    let nonlag_periods = expand_usize_range(
        "nonlag_period",
        range.nonlag_period.0,
        range.nonlag_period.1,
        range.nonlag_period.2,
    )?;
    let dynamic_zone_periods = expand_usize_range(
        "dynamic_zone_period",
        range.dynamic_zone_period.0,
        range.dynamic_zone_period.1,
        range.dynamic_zone_period.2,
    )?;
    let buy_probabilities = expand_float_range(
        "buy_probability",
        range.buy_probability.0,
        range.buy_probability.1,
        range.buy_probability.2,
    )?;
    let sell_probabilities = expand_float_range(
        "sell_probability",
        range.sell_probability.0,
        range.sell_probability.1,
        range.sell_probability.2,
    )?;
    let highpass_periods = expand_usize_range(
        "highpass_period",
        range.highpass_period.0,
        range.highpass_period.1,
        range.highpass_period.2,
    )?;

    let mut combos = Vec::new();
    for &period in &periods {
        for &norm_period in &norm_periods {
            for &normalization_length in &normalization_lengths {
                for &nonlag_period in &nonlag_periods {
                    for &dynamic_zone_period in &dynamic_zone_periods {
                        for &buy_probability in &buy_probabilities {
                            for &sell_probability in &sell_probabilities {
                                for &highpass_period in &highpass_periods {
                                    combos.push(PossibleRsiParams {
                                        period: Some(period),
                                        rsi_mode: base.rsi_mode.clone(),
                                        norm_period: Some(norm_period),
                                        normalization_mode: base.normalization_mode.clone(),
                                        normalization_length: Some(normalization_length),
                                        nonlag_period: Some(nonlag_period),
                                        dynamic_zone_period: Some(dynamic_zone_period),
                                        buy_probability: Some(buy_probability),
                                        sell_probability: Some(sell_probability),
                                        signal_type: base.signal_type.clone(),
                                        run_highpass: base.run_highpass,
                                        highpass_period: Some(highpass_period),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(combos)
}

pub fn expand_grid_possible_rsi(
    range: &PossibleRsiBatchRange,
    base: &PossibleRsiParams,
) -> Vec<PossibleRsiParams> {
    expand_grid_checked(range, base).unwrap_or_default()
}

#[inline(always)]
fn alloc_matrix(rows: usize, cols: usize, warmups: &[usize]) -> Vec<f64> {
    let mut matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut matrix, cols, warmups);
    let mut out = unsafe {
        Vec::from_raw_parts(
            matrix.as_mut_ptr() as *mut f64,
            matrix.len(),
            matrix.capacity(),
        )
    };
    std::mem::forget(matrix);
    out
}

pub fn possible_rsi_batch_with_kernel(
    data: &[f64],
    range: &PossibleRsiBatchRange,
    base: &PossibleRsiParams,
    kernel: Kernel,
) -> Result<PossibleRsiBatchOutput, PossibleRsiError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(PossibleRsiError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(range, base)?;
    if data.is_empty() {
        return Err(PossibleRsiError::EmptyInputData);
    }
    if longest_valid_run(data) == 0 {
        return Err(PossibleRsiError::AllValuesNaN);
    }

    let rows = combos.len();
    let cols = data.len();
    let warmups = combos
        .iter()
        .map(|params| {
            resolve_params(params)
                .map(estimated_warmup)
                .unwrap_or(cols)
                .min(cols)
        })
        .collect::<Vec<_>>();
    let mut value = alloc_matrix(rows, cols, &warmups);
    let mut buy_level = alloc_matrix(rows, cols, &warmups);
    let mut sell_level = alloc_matrix(rows, cols, &warmups);
    let mut middle_level = alloc_matrix(rows, cols, &warmups);
    let mut state = alloc_matrix(rows, cols, &warmups);
    let mut long_signal = alloc_matrix(rows, cols, &warmups);
    let mut short_signal = alloc_matrix(rows, cols, &warmups);

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize,
                  dst_value: &mut [f64],
                  dst_buy: &mut [f64],
                  dst_sell: &mut [f64],
                  dst_middle: &mut [f64],
                  dst_state: &mut [f64],
                  dst_long: &mut [f64],
                  dst_short: &mut [f64]| {
        if let Ok(out) = possible_rsi(&PossibleRsiInput::from_slice(data, combos[row].clone())) {
            dst_value.copy_from_slice(&out.value);
            dst_buy.copy_from_slice(&out.buy_level);
            dst_sell.copy_from_slice(&out.sell_level);
            dst_middle.copy_from_slice(&out.middle_level);
            dst_state.copy_from_slice(&out.state);
            dst_long.copy_from_slice(&out.long_signal);
            dst_short.copy_from_slice(&out.short_signal);
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        value
            .par_chunks_mut(cols)
            .zip(buy_level.par_chunks_mut(cols))
            .zip(sell_level.par_chunks_mut(cols))
            .zip(middle_level.par_chunks_mut(cols))
            .zip(state.par_chunks_mut(cols))
            .zip(long_signal.par_chunks_mut(cols))
            .zip(short_signal.par_chunks_mut(cols))
            .enumerate()
            .for_each(
                |(
                    row,
                    (
                        (((((dst_value, dst_buy), dst_sell), dst_middle), dst_state), dst_long),
                        dst_short,
                    ),
                )| {
                    worker(
                        row, dst_value, dst_buy, dst_sell, dst_middle, dst_state, dst_long,
                        dst_short,
                    );
                },
            );
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (
            row,
            ((((((dst_value, dst_buy), dst_sell), dst_middle), dst_state), dst_long), dst_short),
        ) in value
            .chunks_mut(cols)
            .zip(buy_level.chunks_mut(cols))
            .zip(sell_level.chunks_mut(cols))
            .zip(middle_level.chunks_mut(cols))
            .zip(state.chunks_mut(cols))
            .zip(long_signal.chunks_mut(cols))
            .zip(short_signal.chunks_mut(cols))
            .enumerate()
        {
            worker(
                row, dst_value, dst_buy, dst_sell, dst_middle, dst_state, dst_long, dst_short,
            );
        }
    }

    Ok(PossibleRsiBatchOutput {
        value,
        buy_level,
        sell_level,
        middle_level,
        state,
        long_signal,
        short_signal,
        combos,
        rows,
        cols,
    })
}
pub fn possible_rsi_batch_slice(
    data: &[f64],
    range: &PossibleRsiBatchRange,
    base: &PossibleRsiParams,
    kernel: Kernel,
) -> Result<PossibleRsiBatchOutput, PossibleRsiError> {
    possible_rsi_batch_with_kernel(data, range, base, kernel)
}

pub fn possible_rsi_batch_par_slice(
    data: &[f64],
    range: &PossibleRsiBatchRange,
    base: &PossibleRsiParams,
    kernel: Kernel,
) -> Result<PossibleRsiBatchOutput, PossibleRsiError> {
    possible_rsi_batch_with_kernel(data, range, base, kernel)
}

#[cfg(feature = "python")]
#[pyfunction(name = "possible_rsi")]
#[pyo3(signature = (data, period=32, rsi_mode="regular", norm_period=100, normalization_mode="gaussian_fisher", normalization_length=15, nonlag_period=15, dynamic_zone_period=20, buy_probability=0.2, sell_probability=0.2, signal_type="zeroline_crossover", run_highpass=false, highpass_period=15, kernel=None))]
pub fn possible_rsi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    rsi_mode: &str,
    norm_period: usize,
    normalization_mode: &str,
    normalization_length: usize,
    nonlag_period: usize,
    dynamic_zone_period: usize,
    buy_probability: f64,
    sell_probability: f64,
    signal_type: &str,
    run_highpass: bool,
    highpass_period: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = PossibleRsiInput::from_slice(
        data,
        PossibleRsiParams {
            period: Some(period),
            rsi_mode: Some(rsi_mode.to_string()),
            norm_period: Some(norm_period),
            normalization_mode: Some(normalization_mode.to_string()),
            normalization_length: Some(normalization_length),
            nonlag_period: Some(nonlag_period),
            dynamic_zone_period: Some(dynamic_zone_period),
            buy_probability: Some(buy_probability),
            sell_probability: Some(sell_probability),
            signal_type: Some(signal_type.to_string()),
            run_highpass: Some(run_highpass),
            highpass_period: Some(highpass_period),
        },
    );
    let out = py
        .allow_threads(|| possible_rsi_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.value.into_pyarray(py),
        out.buy_level.into_pyarray(py),
        out.sell_level.into_pyarray(py),
        out.middle_level.into_pyarray(py),
        out.state.into_pyarray(py),
        out.long_signal.into_pyarray(py),
        out.short_signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "PossibleRsiStream")]
pub struct PossibleRsiStreamPy {
    stream: PossibleRsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PossibleRsiStreamPy {
    #[new]
    #[pyo3(signature = (period=32, rsi_mode="regular", norm_period=100, normalization_mode="gaussian_fisher", normalization_length=15, nonlag_period=15, dynamic_zone_period=20, buy_probability=0.2, sell_probability=0.2, signal_type="zeroline_crossover", run_highpass=false, highpass_period=15))]
    fn new(
        period: usize,
        rsi_mode: &str,
        norm_period: usize,
        normalization_mode: &str,
        normalization_length: usize,
        nonlag_period: usize,
        dynamic_zone_period: usize,
        buy_probability: f64,
        sell_probability: f64,
        signal_type: &str,
        run_highpass: bool,
        highpass_period: usize,
    ) -> PyResult<Self> {
        let stream = PossibleRsiStream::try_new(PossibleRsiParams {
            period: Some(period),
            rsi_mode: Some(rsi_mode.to_string()),
            norm_period: Some(norm_period),
            normalization_mode: Some(normalization_mode.to_string()),
            normalization_length: Some(normalization_length),
            nonlag_period: Some(nonlag_period),
            dynamic_zone_period: Some(dynamic_zone_period),
            buy_probability: Some(buy_probability),
            sell_probability: Some(sell_probability),
            signal_type: Some(signal_type.to_string()),
            run_highpass: Some(run_highpass),
            highpass_period: Some(highpass_period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64, f64, f64, f64)> {
        self.stream.update(value).map(|point| {
            (
                point.value,
                point.buy_level,
                point.sell_level,
                point.middle_level,
                point.state,
                point.long_signal,
                point.short_signal,
            )
        })
    }

    fn reset(&mut self) {
        self.stream.reset();
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "possible_rsi_batch")]
#[pyo3(signature = (data, period_range=(32, 32, 0), rsi_mode="regular", norm_period_range=(100, 100, 0), normalization_mode="gaussian_fisher", normalization_length_range=(15, 15, 0), nonlag_period_range=(15, 15, 0), dynamic_zone_period_range=(20, 20, 0), buy_probability_range=(0.2, 0.2, 0.0), sell_probability_range=(0.2, 0.2, 0.0), signal_type="zeroline_crossover", run_highpass=false, highpass_period=15, kernel=None))]
pub fn possible_rsi_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    rsi_mode: &str,
    norm_period_range: (usize, usize, usize),
    normalization_mode: &str,
    normalization_length_range: (usize, usize, usize),
    nonlag_period_range: (usize, usize, usize),
    dynamic_zone_period_range: (usize, usize, usize),
    buy_probability_range: (f64, f64, f64),
    sell_probability_range: (f64, f64, f64),
    signal_type: &str,
    run_highpass: bool,
    highpass_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            possible_rsi_batch_with_kernel(
                data,
                &PossibleRsiBatchRange {
                    period: period_range,
                    norm_period: norm_period_range,
                    normalization_length: normalization_length_range,
                    nonlag_period: nonlag_period_range,
                    dynamic_zone_period: dynamic_zone_period_range,
                    buy_probability: buy_probability_range,
                    sell_probability: sell_probability_range,
                    highpass_period: (highpass_period, highpass_period, 0),
                },
                &PossibleRsiParams {
                    period: None,
                    rsi_mode: Some(rsi_mode.to_string()),
                    norm_period: None,
                    normalization_mode: Some(normalization_mode.to_string()),
                    normalization_length: None,
                    nonlag_period: None,
                    dynamic_zone_period: None,
                    buy_probability: None,
                    sell_probability: None,
                    signal_type: Some(signal_type.to_string()),
                    run_highpass: Some(run_highpass),
                    highpass_period: Some(highpass_period),
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "value",
        output
            .value
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "buy_level",
        output
            .buy_level
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "sell_level",
        output
            .sell_level
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "middle_level",
        output
            .middle_level
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "state",
        output
            .state
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "long_signal",
        output
            .long_signal
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "short_signal",
        output
            .short_signal
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "periods",
        output
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(32) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "norm_periods",
        output
            .combos
            .iter()
            .map(|combo| combo.norm_period.unwrap_or(100) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "normalization_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.normalization_length.unwrap_or(15) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "nonlag_periods",
        output
            .combos
            .iter()
            .map(|combo| combo.nonlag_period.unwrap_or(15) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "dynamic_zone_periods",
        output
            .combos
            .iter()
            .map(|combo| combo.dynamic_zone_period.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "buy_probabilities",
        output
            .combos
            .iter()
            .map(|combo| combo.buy_probability.unwrap_or(0.2))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "sell_probabilities",
        output
            .combos
            .iter()
            .map(|combo| combo.sell_probability.unwrap_or(0.2))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "rsi_modes",
        output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .rsi_mode
                    .clone()
                    .unwrap_or_else(|| "regular".to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "normalization_modes",
        output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .normalization_mode
                    .clone()
                    .unwrap_or_else(|| "gaussian_fisher".to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "signal_types",
        output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .signal_type
                    .clone()
                    .unwrap_or_else(|| "zeroline_crossover".to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "run_highpass",
        output
            .combos
            .iter()
            .map(|combo| combo.run_highpass.unwrap_or(false))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_possible_rsi_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(possible_rsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(possible_rsi_batch_py, m)?)?;
    m.add_class::<PossibleRsiStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PossibleRsiBatchConfig {
    pub period_range: Vec<usize>,
    pub rsi_mode: Option<String>,
    pub norm_period_range: Vec<usize>,
    pub normalization_mode: Option<String>,
    pub normalization_length_range: Vec<usize>,
    pub nonlag_period_range: Vec<usize>,
    pub dynamic_zone_period_range: Vec<usize>,
    pub buy_probability_range: Vec<f64>,
    pub sell_probability_range: Vec<f64>,
    pub signal_type: Option<String>,
    pub run_highpass: Option<bool>,
    pub highpass_period: Option<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = possible_rsi_js)]
pub fn possible_rsi_js(
    data: &[f64],
    period: usize,
    rsi_mode: &str,
    norm_period: usize,
    normalization_mode: &str,
    normalization_length: usize,
    nonlag_period: usize,
    dynamic_zone_period: usize,
    buy_probability: f64,
    sell_probability: f64,
    signal_type: &str,
    run_highpass: bool,
    highpass_period: usize,
) -> Result<JsValue, JsValue> {
    let input = PossibleRsiInput::from_slice(
        data,
        PossibleRsiParams {
            period: Some(period),
            rsi_mode: Some(rsi_mode.to_string()),
            norm_period: Some(norm_period),
            normalization_mode: Some(normalization_mode.to_string()),
            normalization_length: Some(normalization_length),
            nonlag_period: Some(nonlag_period),
            dynamic_zone_period: Some(dynamic_zone_period),
            buy_probability: Some(buy_probability),
            sell_probability: Some(sell_probability),
            signal_type: Some(signal_type.to_string()),
            run_highpass: Some(run_highpass),
            highpass_period: Some(highpass_period),
        },
    );
    let out = possible_rsi_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("value"),
        &serde_wasm_bindgen::to_value(&out.value).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("buy_level"),
        &serde_wasm_bindgen::to_value(&out.buy_level).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("sell_level"),
        &serde_wasm_bindgen::to_value(&out.sell_level).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("middle_level"),
        &serde_wasm_bindgen::to_value(&out.middle_level).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("state"),
        &serde_wasm_bindgen::to_value(&out.state).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("long_signal"),
        &serde_wasm_bindgen::to_value(&out.long_signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("short_signal"),
        &serde_wasm_bindgen::to_value(&out.short_signal).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = possible_rsi_batch_js)]
pub fn possible_rsi_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: PossibleRsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.period_range.len() != 3
        || config.norm_period_range.len() != 3
        || config.normalization_length_range.len() != 3
        || config.nonlag_period_range.len() != 3
        || config.dynamic_zone_period_range.len() != 3
        || config.buy_probability_range.len() != 3
        || config.sell_probability_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }
    let highpass_period = config.highpass_period.unwrap_or(15);
    let out = possible_rsi_batch_with_kernel(
        data,
        &PossibleRsiBatchRange {
            period: (
                config.period_range[0],
                config.period_range[1],
                config.period_range[2],
            ),
            norm_period: (
                config.norm_period_range[0],
                config.norm_period_range[1],
                config.norm_period_range[2],
            ),
            normalization_length: (
                config.normalization_length_range[0],
                config.normalization_length_range[1],
                config.normalization_length_range[2],
            ),
            nonlag_period: (
                config.nonlag_period_range[0],
                config.nonlag_period_range[1],
                config.nonlag_period_range[2],
            ),
            dynamic_zone_period: (
                config.dynamic_zone_period_range[0],
                config.dynamic_zone_period_range[1],
                config.dynamic_zone_period_range[2],
            ),
            buy_probability: (
                config.buy_probability_range[0],
                config.buy_probability_range[1],
                config.buy_probability_range[2],
            ),
            sell_probability: (
                config.sell_probability_range[0],
                config.sell_probability_range[1],
                config.sell_probability_range[2],
            ),
            highpass_period: (highpass_period, highpass_period, 0),
        },
        &PossibleRsiParams {
            period: None,
            rsi_mode: Some(config.rsi_mode.unwrap_or_else(|| "regular".to_string())),
            norm_period: None,
            normalization_mode: Some(
                config
                    .normalization_mode
                    .unwrap_or_else(|| "gaussian_fisher".to_string()),
            ),
            normalization_length: None,
            nonlag_period: None,
            dynamic_zone_period: None,
            buy_probability: None,
            sell_probability: None,
            signal_type: Some(
                config
                    .signal_type
                    .unwrap_or_else(|| "zeroline_crossover".to_string()),
            ),
            run_highpass: Some(config.run_highpass.unwrap_or(false)),
            highpass_period: Some(highpass_period),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("value"),
        &serde_wasm_bindgen::to_value(&out.value).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("buy_level"),
        &serde_wasm_bindgen::to_value(&out.buy_level).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("sell_level"),
        &serde_wasm_bindgen::to_value(&out.sell_level).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("middle_level"),
        &serde_wasm_bindgen::to_value(&out.middle_level).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("state"),
        &serde_wasm_bindgen::to_value(&out.state).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("long_signal"),
        &serde_wasm_bindgen::to_value(&out.long_signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("short_signal"),
        &serde_wasm_bindgen::to_value(&out.short_signal).unwrap(),
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
pub fn possible_rsi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(7 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn possible_rsi_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 7 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn possible_rsi_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    rsi_mode: &str,
    norm_period: usize,
    normalization_mode: &str,
    normalization_length: usize,
    nonlag_period: usize,
    dynamic_zone_period: usize,
    buy_probability: f64,
    sell_probability: f64,
    signal_type: &str,
    run_highpass: bool,
    highpass_period: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to possible_rsi_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 7 * len);
        let (dst_value, rest) = out.split_at_mut(len);
        let (dst_buy_level, rest) = rest.split_at_mut(len);
        let (dst_sell_level, rest) = rest.split_at_mut(len);
        let (dst_middle_level, rest) = rest.split_at_mut(len);
        let (dst_state, rest) = rest.split_at_mut(len);
        let (dst_long_signal, dst_short_signal) = rest.split_at_mut(len);
        let input = PossibleRsiInput::from_slice(
            data,
            PossibleRsiParams {
                period: Some(period),
                rsi_mode: Some(rsi_mode.to_string()),
                norm_period: Some(norm_period),
                normalization_mode: Some(normalization_mode.to_string()),
                normalization_length: Some(normalization_length),
                nonlag_period: Some(nonlag_period),
                dynamic_zone_period: Some(dynamic_zone_period),
                buy_probability: Some(buy_probability),
                sell_probability: Some(sell_probability),
                signal_type: Some(signal_type.to_string()),
                run_highpass: Some(run_highpass),
                highpass_period: Some(highpass_period),
            },
        );
        possible_rsi_into_slice(
            dst_value,
            dst_buy_level,
            dst_sell_level,
            dst_middle_level,
            dst_state,
            dst_long_signal,
            dst_short_signal,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn possible_rsi_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    rsi_mode: &str,
    norm_period_start: usize,
    norm_period_end: usize,
    norm_period_step: usize,
    normalization_mode: &str,
    normalization_length_start: usize,
    normalization_length_end: usize,
    normalization_length_step: usize,
    nonlag_period_start: usize,
    nonlag_period_end: usize,
    nonlag_period_step: usize,
    dynamic_zone_period_start: usize,
    dynamic_zone_period_end: usize,
    dynamic_zone_period_step: usize,
    buy_probability_start: f64,
    buy_probability_end: f64,
    buy_probability_step: f64,
    sell_probability_start: f64,
    sell_probability_end: f64,
    sell_probability_step: f64,
    signal_type: &str,
    run_highpass: bool,
    highpass_period: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to possible_rsi_batch_into",
        ));
    }
    let batch = unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        possible_rsi_batch_with_kernel(
            data,
            &PossibleRsiBatchRange {
                period: (period_start, period_end, period_step),
                norm_period: (norm_period_start, norm_period_end, norm_period_step),
                normalization_length: (
                    normalization_length_start,
                    normalization_length_end,
                    normalization_length_step,
                ),
                nonlag_period: (nonlag_period_start, nonlag_period_end, nonlag_period_step),
                dynamic_zone_period: (
                    dynamic_zone_period_start,
                    dynamic_zone_period_end,
                    dynamic_zone_period_step,
                ),
                buy_probability: (
                    buy_probability_start,
                    buy_probability_end,
                    buy_probability_step,
                ),
                sell_probability: (
                    sell_probability_start,
                    sell_probability_end,
                    sell_probability_step,
                ),
                highpass_period: (highpass_period, highpass_period, 0),
            },
            &PossibleRsiParams {
                period: None,
                rsi_mode: Some(rsi_mode.to_string()),
                norm_period: None,
                normalization_mode: Some(normalization_mode.to_string()),
                normalization_length: None,
                nonlag_period: None,
                dynamic_zone_period: None,
                buy_probability: None,
                sell_probability: None,
                signal_type: Some(signal_type.to_string()),
                run_highpass: Some(run_highpass),
                highpass_period: Some(highpass_period),
            },
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?
    };
    let rows = batch.rows;
    let total = rows
        .checked_mul(len)
        .and_then(|value| value.checked_mul(7))
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in possible_rsi_batch_into"))?;
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let field_len = rows * len;
        let (dst_value, rest) = out.split_at_mut(field_len);
        let (dst_buy_level, rest) = rest.split_at_mut(field_len);
        let (dst_sell_level, rest) = rest.split_at_mut(field_len);
        let (dst_middle_level, rest) = rest.split_at_mut(field_len);
        let (dst_state, rest) = rest.split_at_mut(field_len);
        let (dst_long_signal, dst_short_signal) = rest.split_at_mut(field_len);
        dst_value.copy_from_slice(&batch.value);
        dst_buy_level.copy_from_slice(&batch.buy_level);
        dst_sell_level.copy_from_slice(&batch.sell_level);
        dst_middle_level.copy_from_slice(&batch.middle_level);
        dst_state.copy_from_slice(&batch.state);
        dst_long_signal.copy_from_slice(&batch.long_signal);
        dst_short_signal.copy_from_slice(&batch.short_signal);
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn possible_rsi_output_into_js(
    data: &[f64],
    period: usize,
    rsi_mode: &str,
    norm_period: usize,
    normalization_mode: &str,
    normalization_length: usize,
    nonlag_period: usize,
    dynamic_zone_period: usize,
    buy_probability: f64,
    sell_probability: f64,
    signal_type: &str,
    run_highpass: bool,
    highpass_period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = possible_rsi_js(
        data,
        period,
        rsi_mode,
        norm_period,
        normalization_mode,
        normalization_length,
        nonlag_period,
        dynamic_zone_period,
        buy_probability,
        sell_probability,
        signal_type,
        run_highpass,
        highpass_period,
    )?;
    crate::write_wasm_object_f64_outputs("possible_rsi_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn possible_rsi_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = possible_rsi_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs("possible_rsi_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, compute_cpu_batch, IndicatorBatchRequest, IndicatorComputeRequest,
        IndicatorDataRef, IndicatorParamSet, IndicatorSeries, ParamKV, ParamValue,
    };

    fn sample_close(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.07 + (x * 0.13).sin() * 1.4 + (x * 0.037).cos() * 0.9
            })
            .collect()
    }

    fn assert_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "left={a} right={b}");
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn possible_rsi_output_contract() -> Result<(), Box<dyn Error>> {
        let data = sample_close(384);
        let out = possible_rsi(&PossibleRsiInput::from_slice(
            &data,
            PossibleRsiParams::default(),
        ))?;
        assert_eq!(out.value.len(), data.len());
        assert_eq!(out.buy_level.len(), data.len());
        assert_eq!(out.sell_level.len(), data.len());
        assert_eq!(out.middle_level.len(), data.len());
        assert_eq!(out.state.len(), data.len());
        assert_eq!(out.long_signal.len(), data.len());
        assert_eq!(out.short_signal.len(), data.len());
        assert!(out.value.iter().any(|v| v.is_finite()));
        assert!(out.value.last().is_some_and(|v| v.is_finite()));
        assert!(out.buy_level.last().is_some_and(|v| v.is_finite()));
        assert!(out.sell_level.last().is_some_and(|v| v.is_finite()));
        assert!(out.middle_level.last().is_some_and(|v| v.is_finite()));
        assert!(out.state.last().is_some_and(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn possible_rsi_invalid_period_rejected() {
        let data = sample_close(64);
        let err = possible_rsi(&PossibleRsiInput::from_slice(
            &data,
            PossibleRsiParams {
                period: Some(0),
                ..PossibleRsiParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(err, PossibleRsiError::InvalidPeriod { period: 0 }));
    }

    #[test]
    fn possible_rsi_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = sample_close(360);
        let params = PossibleRsiParams {
            period: Some(28),
            rsi_mode: Some("cutler".to_string()),
            norm_period: Some(90),
            normalization_mode: Some("softmax".to_string()),
            normalization_length: Some(12),
            nonlag_period: Some(11),
            dynamic_zone_period: Some(18),
            buy_probability: Some(0.2),
            sell_probability: Some(0.2),
            signal_type: Some("levels_crossover".to_string()),
            run_highpass: Some(true),
            highpass_period: Some(13),
        };
        let batch = possible_rsi(&PossibleRsiInput::from_slice(&data, params.clone()))?;
        let mut stream = PossibleRsiStream::try_new(params)?;
        let mut value = Vec::with_capacity(data.len());
        let mut buy_level = Vec::with_capacity(data.len());
        let mut sell_level = Vec::with_capacity(data.len());
        let mut middle_level = Vec::with_capacity(data.len());
        let mut state = Vec::with_capacity(data.len());
        let mut long_signal = Vec::with_capacity(data.len());
        let mut short_signal = Vec::with_capacity(data.len());
        for &sample in &data {
            if let Some(point) = stream.update(sample) {
                value.push(point.value);
                buy_level.push(point.buy_level);
                sell_level.push(point.sell_level);
                middle_level.push(point.middle_level);
                state.push(point.state);
                long_signal.push(point.long_signal);
                short_signal.push(point.short_signal);
            } else {
                value.push(f64::NAN);
                buy_level.push(f64::NAN);
                sell_level.push(f64::NAN);
                middle_level.push(f64::NAN);
                state.push(f64::NAN);
                long_signal.push(0.0);
                short_signal.push(0.0);
            }
        }
        let start = state
            .iter()
            .position(|v| v.is_finite())
            .expect("stream should eventually emit finite state");
        assert_close(&value[start..], &batch.value[start..], 1e-12);
        assert_close(&buy_level[start..], &batch.buy_level[start..], 1e-12);
        assert_close(&sell_level[start..], &batch.sell_level[start..], 1e-12);
        assert_close(&middle_level[start..], &batch.middle_level[start..], 1e-12);
        assert_close(&state[start..], &batch.state[start..], 1e-12);
        Ok(())
    }

    #[test]
    fn possible_rsi_dispatch_returns_selected_output() -> Result<(), Box<dyn Error>> {
        let data = sample_close(256);
        let params = [
            ParamKV {
                key: "period",
                value: ParamValue::Int(28),
            },
            ParamKV {
                key: "rsi_mode",
                value: ParamValue::EnumString("regular"),
            },
            ParamKV {
                key: "signal_type",
                value: ParamValue::EnumString("zeroline_crossover"),
            },
        ];
        let req = IndicatorComputeRequest {
            indicator_id: "possible_rsi",
            data: IndicatorDataRef::Slice { values: &data },
            params: &params,
            output_id: Some("state"),
            kernel: Kernel::Auto,
        };
        let out = compute_cpu(req)?;
        let values = match out.series {
            IndicatorSeries::F64(values) => values,
            other => panic!("unexpected series type: {other:?}"),
        };
        let direct = possible_rsi(&PossibleRsiInput::from_slice(
            &data,
            PossibleRsiParams {
                period: Some(28),
                rsi_mode: Some("regular".to_string()),
                signal_type: Some("zeroline_crossover".to_string()),
                ..PossibleRsiParams::default()
            },
        ))?;
        assert_close(&values, &direct.state, 1e-12);
        Ok(())
    }

    #[test]
    fn possible_rsi_batch_dispatch_returns_selected_output() -> Result<(), Box<dyn Error>> {
        let data = sample_close(320);
        let params = [
            ParamKV {
                key: "period",
                value: ParamValue::Int(28),
            },
            ParamKV {
                key: "rsi_mode",
                value: ParamValue::EnumString("regular"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];
        let req = IndicatorBatchRequest {
            indicator_id: "possible_rsi",
            output_id: Some("value"),
            combos: &combos,
            data: IndicatorDataRef::Slice { values: &data },
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req)?;
        let values = out.values_f64.expect("f64 output");
        let direct = possible_rsi(&PossibleRsiInput::from_slice(
            &data,
            PossibleRsiParams {
                period: Some(28),
                rsi_mode: Some("regular".to_string()),
                ..PossibleRsiParams::default()
            },
        ))?;
        assert_close(&values, &direct.value, 1e-12);
        Ok(())
    }
}

#[derive(Debug, Clone)]
#[cfg(any())]
mod duplicate_impl {
    use super::*;

    struct HighPassState {
        period: usize,
        index: usize,
        prev_src1: f64,
        prev_src2: f64,
        prev_hp1: f64,
        prev_hp2: f64,
    }

    impl HighPassState {
        #[inline(always)]
        fn new(period: usize) -> Self {
            Self {
                period,
                index: 0,
                prev_src1: 0.0,
                prev_src2: 0.0,
                prev_hp1: 0.0,
                prev_hp2: 0.0,
            }
        }

        #[inline(always)]
        fn reset(&mut self) {
            self.index = 0;
            self.prev_src1 = 0.0;
            self.prev_src2 = 0.0;
            self.prev_hp1 = 0.0;
            self.prev_hp2 = 0.0;
        }

        #[inline(always)]
        fn update(&mut self, src: f64) -> f64 {
            let idx = self.index;
            self.index = self.index.saturating_add(1);
            if idx < 4 {
                self.prev_src2 = self.prev_src1;
                self.prev_src1 = src;
                self.prev_hp2 = self.prev_hp1;
                self.prev_hp1 = 0.0;
                return 0.0;
            }

            let a1 = (-1.414 * std::f64::consts::PI / self.period as f64).exp();
            let b1 = 2.0 * a1 * (1.414 * std::f64::consts::PI / self.period as f64).cos();
            let c2 = b1;
            let c3 = -(a1 * a1);
            let c1 = (1.0 + c2 - c3) * 0.25;
            let hp = c1 * (src - 2.0 * self.prev_src1 + self.prev_src2)
                + c2 * self.prev_hp1
                + c3 * self.prev_hp2;
            self.prev_src2 = self.prev_src1;
            self.prev_src1 = src;
            self.prev_hp2 = self.prev_hp1;
            self.prev_hp1 = hp;
            hp
        }
    }

    #[derive(Debug, Clone)]
    struct CutlerRsiState {
        period: usize,
        prev: Option<f64>,
        gains: Vec<f64>,
        losses: Vec<f64>,
        sum_gain: f64,
        sum_loss: f64,
        index: usize,
        count: usize,
    }

    impl CutlerRsiState {
        #[inline(always)]
        fn new(period: usize) -> Self {
            Self {
                period,
                prev: None,
                gains: vec![0.0; period],
                losses: vec![0.0; period],
                sum_gain: 0.0,
                sum_loss: 0.0,
                index: 0,
                count: 0,
            }
        }

        #[inline(always)]
        fn reset(&mut self) {
            self.prev = None;
            self.gains.fill(0.0);
            self.losses.fill(0.0);
            self.sum_gain = 0.0;
            self.sum_loss = 0.0;
            self.index = 0;
            self.count = 0;
        }

        #[inline(always)]
        fn update(&mut self, value: f64) -> Option<f64> {
            let prev = match self.prev {
                Some(prev) => prev,
                None => {
                    self.prev = Some(value);
                    return None;
                }
            };
            self.prev = Some(value);
            let delta = value - prev;
            let gain = delta.max(0.0);
            let loss = (-delta).max(0.0);
            if self.count == self.period {
                self.sum_gain -= self.gains[self.index];
                self.sum_loss -= self.losses[self.index];
            } else {
                self.count += 1;
            }
            self.gains[self.index] = gain;
            self.losses[self.index] = loss;
            self.sum_gain += gain;
            self.sum_loss += loss;
            self.index = (self.index + 1) % self.period;
            if self.count < self.period {
                return None;
            }
            let denom = self.sum_gain + self.sum_loss;
            Some(if denom.abs() <= f64::EPSILON {
                50.0
            } else {
                100.0 * self.sum_gain / denom
            })
        }
    }

    #[derive(Debug, Clone)]
    enum PossibleRsiEngine {
        Regular(RsiStream),
        Rsx(RsxStream),
        Cutler(CutlerRsiState),
    }

    impl PossibleRsiEngine {
        #[inline(always)]
        fn new(mode: PossibleRsiMode, period: usize) -> Result<Self, PossibleRsiError> {
            Ok(match mode {
                PossibleRsiMode::Regular => Self::Regular(
                    RsiStream::try_new(RsiParams {
                        period: Some(period),
                    })
                    .map_err(|e| PossibleRsiError::InvalidInput { msg: e.to_string() })?,
                ),
                PossibleRsiMode::Rsx => Self::Rsx(
                    RsxStream::try_new(RsxParams {
                        period: Some(period),
                    })
                    .map_err(|e| PossibleRsiError::InvalidInput { msg: e.to_string() })?,
                ),
                PossibleRsiMode::Cutler => Self::Cutler(CutlerRsiState::new(period)),
            })
        }

        #[inline(always)]
        fn update(&mut self, value: f64) -> Option<f64> {
            match self {
                Self::Regular(inner) => inner.update(value),
                Self::Rsx(inner) => inner.update(value),
                Self::Cutler(inner) => inner.update(value),
            }
        }

        #[inline(always)]
        fn reset(&mut self) {
            match self {
                Self::Regular(inner) => {
                    *inner = RsiStream::try_new(RsiParams {
                        period: Some(inner.period),
                    })
                    .expect("valid RSI params");
                }
                Self::Rsx(inner) => {
                    *inner = RsxStream::try_new(RsxParams {
                        period: Some(inner.period),
                    })
                    .expect("valid RSX params");
                }
                Self::Cutler(inner) => inner.reset(),
            }
        }
    }

    #[derive(Debug, Clone)]
    struct RollingWindow {
        values: Vec<f64>,
        index: usize,
        count: usize,
    }

    impl RollingWindow {
        #[inline(always)]
        fn new(period: usize) -> Self {
            Self {
                values: vec![0.0; period.max(1)],
                index: 0,
                count: 0,
            }
        }

        #[inline(always)]
        fn reset(&mut self) {
            self.values.fill(0.0);
            self.index = 0;
            self.count = 0;
        }

        #[inline(always)]
        fn push(&mut self, value: f64) {
            self.values[self.index] = value;
            self.index = (self.index + 1) % self.values.len();
            if self.count < self.values.len() {
                self.count += 1;
            }
        }

        #[inline(always)]
        fn len(&self) -> usize {
            self.values.len()
        }

        #[inline(always)]
        fn is_full(&self) -> bool {
            self.count == self.values.len()
        }

        #[inline(always)]
        fn get_recent(&self, age: usize) -> Option<f64> {
            if age >= self.count {
                return None;
            }
            let len = self.values.len();
            let newest = if self.index == 0 {
                len - 1
            } else {
                self.index - 1
            };
            let idx = (newest + len - age) % len;
            Some(self.values[idx])
        }

        #[inline(always)]
        fn min_max(&self) -> Option<(f64, f64)> {
            if !self.is_full() {
                return None;
            }
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for &value in &self.values {
                if value < min_v {
                    min_v = value;
                }
                if value > max_v {
                    max_v = value;
                }
            }
            Some((min_v, max_v))
        }

        #[inline(always)]
        fn mean_std(&self) -> Option<(f64, f64)> {
            if !self.is_full() {
                return None;
            }
            let n = self.values.len() as f64;
            let mut sum = 0.0;
            let mut sumsq = 0.0;
            for &value in &self.values {
                sum += value;
                sumsq += value * value;
            }
            let mean = sum / n;
            let mut var = sumsq / n - mean * mean;
            if var < 0.0 {
                var = 0.0;
            }
            Some((mean, var.sqrt()))
        }

        #[inline(always)]
        fn quantile_nearest_rank(&self, q: f64) -> Option<f64> {
            if !self.is_full() {
                return None;
            }
            let mut scratch = self.values.clone();
            let n = scratch.len();
            let rank = ((q * n as f64).ceil() as usize).clamp(1, n);
            let idx = rank - 1;
            scratch.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
            Some(scratch[idx])
        }
    }

    #[derive(Debug, Clone)]
    struct FisherState {
        window: RollingWindow,
        value_state: f64,
        fish_state: f64,
    }

    impl FisherState {
        #[inline(always)]
        fn new(period: usize) -> Self {
            Self {
                window: RollingWindow::new(period),
                value_state: 0.0,
                fish_state: 0.0,
            }
        }

        #[inline(always)]
        fn reset(&mut self) {
            self.window.reset();
            self.value_state = 0.0;
            self.fish_state = 0.0;
        }

        #[inline(always)]
        fn update(&mut self, value: f64) -> Option<f64> {
            self.window.push(value);
            let (low, high) = self.window.min_max()?;
            let range = high - low;
            let normalized = if range.abs() <= f64::EPSILON {
                0.0
            } else {
                (value - low) / range - 0.5
            };
            let mut next = 0.66 * normalized + 0.67 * self.value_state;
            next = next.clamp(-0.999, 0.999);
            self.value_state = next;
            let fish = 0.5 * ((1.0 + next) / (1.0 - next)).ln() + 0.5 * self.fish_state;
            self.fish_state = fish;
            Some(fish)
        }
    }

    #[derive(Debug, Clone)]
    struct NonLagState {
        weights: Vec<f64>,
        weight_sum: f64,
        window: RollingWindow,
    }

    impl NonLagState {
        #[inline(always)]
        fn new(period: usize) -> Self {
            let cycle = 4.0;
            let coeff = 3.0 * std::f64::consts::PI;
            let phase = period as f64 - 1.0;
            let len = ((period as f64) * cycle + phase) as usize;
            let mut weights = vec![0.0; len.max(1)];
            let mut weight_sum = 0.0;
            for k in 0..weights.len() {
                let t = if k as f64 <= phase - 1.0 {
                    if phase <= 1.0 {
                        0.0
                    } else {
                        k as f64 / (phase - 1.0)
                    }
                } else {
                    1.0 + (k as f64 - phase + 1.0) * (2.0 * cycle - 1.0)
                        / (cycle * period as f64 - 1.0)
                };
                let beta = (std::f64::consts::PI * t).cos();
                let mut g = 1.0 / (coeff * t + 1.0);
                if t <= 0.5 {
                    g = 1.0;
                }
                let weight = g * beta;
                weights[k] = weight;
                weight_sum += weight;
            }
            Self {
                weight_sum,
                window: RollingWindow::new(weights.len()),
                weights,
            }
        }

        #[inline(always)]
        fn reset(&mut self) {
            self.window.reset();
        }

        #[inline(always)]
        fn update(&mut self, value: f64) -> f64 {
            self.window.push(value);
            let mut sum = 0.0;
            for k in 0..self.weights.len() {
                if let Some(sample) = self.window.get_recent(k) {
                    sum += self.weights[k] * sample;
                }
            }
            sum / self.weight_sum
        }
    }

    #[derive(Debug, Clone)]
    pub struct PossibleRsiStream {
        resolved: PossibleRsiResolved,
        highpass: Option<HighPassState>,
        engine: PossibleRsiEngine,
        norm_window: RollingWindow,
        normalization_window: RollingWindow,
        fisher: Option<FisherState>,
        nonlag: NonLagState,
        dz_window: RollingWindow,
        prev_value: Option<f64>,
        prev_prev_value: Option<f64>,
        prev_buy: Option<f64>,
        prev_sell: Option<f64>,
        prev_middle: Option<f64>,
        prev_state: f64,
    }

    impl PossibleRsiStream {
        #[inline(always)]
        pub fn try_new(params: PossibleRsiParams) -> Result<Self, PossibleRsiError> {
            let resolved = resolve_params(&params)?;
            Ok(Self {
                highpass: if resolved.run_highpass {
                    Some(HighPassState::new(resolved.highpass_period))
                } else {
                    None
                },
                engine: PossibleRsiEngine::new(resolved.rsi_mode, resolved.period)?,
                norm_window: RollingWindow::new(resolved.norm_period),
                normalization_window: RollingWindow::new(resolved.normalization_length),
                fisher: if matches!(
                    resolved.normalization_mode,
                    PossibleRsiNormalizationMode::GaussianFisher
                ) {
                    Some(FisherState::new(resolved.normalization_length))
                } else {
                    None
                },
                nonlag: NonLagState::new(resolved.nonlag_period),
                dz_window: RollingWindow::new(resolved.dynamic_zone_period),
                prev_value: None,
                prev_prev_value: None,
                prev_buy: None,
                prev_sell: None,
                prev_middle: None,
                prev_state: 0.0,
                resolved,
            })
        }

        #[inline(always)]
        pub fn reset(&mut self) {
            if let Some(highpass) = &mut self.highpass {
                highpass.reset();
            }
            self.engine.reset();
            self.norm_window.reset();
            self.normalization_window.reset();
            if let Some(fisher) = &mut self.fisher {
                fisher.reset();
            }
            self.nonlag.reset();
            self.dz_window.reset();
            self.prev_value = None;
            self.prev_prev_value = None;
            self.prev_buy = None;
            self.prev_sell = None;
            self.prev_middle = None;
            self.prev_state = 0.0;
        }

        #[inline(always)]
        pub fn update(&mut self, value: f64) -> Option<PossibleRsiPoint> {
            if !value.is_finite() {
                self.reset();
                return None;
            }
            let filtered = if let Some(highpass) = &mut self.highpass {
                highpass.update(value)
            } else {
                value
            };
            let rsi = self.engine.update(filtered)?;
            if !rsi.is_finite() {
                self.prev_prev_value = self.prev_value;
                self.prev_value = Some(f64::NAN);
                return None;
            }
            self.norm_window.push(rsi);
            let (fmin, fmax) = self.norm_window.min_max()?;
            let range = fmax - fmin;
            if range.abs() <= f64::EPSILON {
                self.prev_prev_value = self.prev_value;
                self.prev_value = Some(f64::NAN);
                return None;
            }
            let base = 100.0 * (rsi - fmin) / range;
            let normalized = match self.resolved.normalization_mode {
                PossibleRsiNormalizationMode::GaussianFisher => {
                    self.fisher.as_mut().and_then(|state| state.update(base))?
                }
                PossibleRsiNormalizationMode::Softmax => {
                    self.normalization_window.push(base);
                    let (mean, stdev) = self.normalization_window.mean_std()?;
                    if stdev.abs() <= f64::EPSILON {
                        0.0
                    } else {
                        let z = (base - mean) / stdev;
                        (1.0 - (-z).exp()) / (1.0 + (-z).exp())
                    }
                }
                PossibleRsiNormalizationMode::RegularNorm => {
                    self.normalization_window.push(base);
                    let (mean, stdev) = self.normalization_window.mean_std()?;
                    if stdev.abs() <= f64::EPSILON {
                        0.0
                    } else {
                        (base - mean) / (stdev * 3.0)
                    }
                }
            };

            let value = self.nonlag.update(normalized);
            self.dz_window.push(value);
            let buy_level = self
                .dz_window
                .quantile_nearest_rank(self.resolved.buy_probability)?;
            let sell_level = self
                .dz_window
                .quantile_nearest_rank(1.0 - self.resolved.sell_probability)?;
            let middle_level = self.dz_window.quantile_nearest_rank(0.5)?;
            let state = match self.resolved.signal_type {
                PossibleRsiSignalType::Slope => {
                    if let Some(prev) = self.prev_value {
                        if value < prev {
                            -1.0
                        } else if value > prev {
                            1.0
                        } else {
                            self.prev_state
                        }
                    } else {
                        0.0
                    }
                }
                PossibleRsiSignalType::DynamicMiddleCrossover => {
                    if value < middle_level {
                        -1.0
                    } else if value > middle_level {
                        1.0
                    } else {
                        self.prev_state
                    }
                }
                PossibleRsiSignalType::LevelsCrossover => {
                    if value < buy_level {
                        -1.0
                    } else if value > sell_level {
                        1.0
                    } else {
                        self.prev_state
                    }
                }
                PossibleRsiSignalType::ZerolineCrossover => {
                    if value < 0.0 {
                        -1.0
                    } else if value > 0.0 {
                        1.0
                    } else {
                        self.prev_state
                    }
                }
            };
            let (long_signal, short_signal) =
                self.crossover_signals(value, buy_level, sell_level, middle_level);

            self.prev_prev_value = self.prev_value;
            self.prev_value = Some(value);
            self.prev_buy = Some(buy_level);
            self.prev_sell = Some(sell_level);
            self.prev_middle = Some(middle_level);
            self.prev_state = state;

            Some(PossibleRsiPoint {
                value,
                buy_level,
                sell_level,
                middle_level,
                state,
                long_signal,
                short_signal,
            })
        }

        #[inline(always)]
        fn crossover_signals(
            &self,
            value: f64,
            buy_level: f64,
            sell_level: f64,
            middle_level: f64,
        ) -> (f64, f64) {
            let Some(prev_value) = self.prev_value else {
                return (0.0, 0.0);
            };
            match self.resolved.signal_type {
                PossibleRsiSignalType::Slope => {
                    let Some(prev_prev_value) = self.prev_prev_value else {
                        return (0.0, 0.0);
                    };
                    (
                        if prev_value <= prev_prev_value && value > prev_value {
                            1.0
                        } else {
                            0.0
                        },
                        if prev_value >= prev_prev_value && value < prev_value {
                            1.0
                        } else {
                            0.0
                        },
                    )
                }
                PossibleRsiSignalType::DynamicMiddleCrossover => {
                    let prev_middle = self.prev_middle.unwrap_or(middle_level);
                    (
                        if prev_value <= prev_middle && value > middle_level {
                            1.0
                        } else {
                            0.0
                        },
                        if prev_value >= prev_middle && value < middle_level {
                            1.0
                        } else {
                            0.0
                        },
                    )
                }
                PossibleRsiSignalType::LevelsCrossover => {
                    let prev_buy = self.prev_buy.unwrap_or(buy_level);
                    let prev_sell = self.prev_sell.unwrap_or(sell_level);
                    (
                        if prev_value <= prev_sell && value > sell_level {
                            1.0
                        } else {
                            0.0
                        },
                        if prev_value >= prev_buy && value < buy_level {
                            1.0
                        } else {
                            0.0
                        },
                    )
                }
                PossibleRsiSignalType::ZerolineCrossover => (
                    if prev_value <= 0.0 && value > 0.0 {
                        1.0
                    } else {
                        0.0
                    },
                    if prev_value >= 0.0 && value < 0.0 {
                        1.0
                    } else {
                        0.0
                    },
                ),
            }
        }
    }

    #[inline(always)]
    fn longest_valid_run(data: &[f64]) -> usize {
        let mut best = 0usize;
        let mut current = 0usize;
        for &value in data {
            if value.is_finite() {
                current += 1;
                if current > best {
                    best = current;
                }
            } else {
                current = 0;
            }
        }
        best
    }

    #[inline(always)]
    fn resolve_params(params: &PossibleRsiParams) -> Result<PossibleRsiResolved, PossibleRsiError> {
        let period = params.period.unwrap_or(32);
        if period == 0 {
            return Err(PossibleRsiError::InvalidPeriod { period });
        }
        let norm_period = params.norm_period.unwrap_or(100);
        if norm_period == 0 {
            return Err(PossibleRsiError::InvalidNormPeriod { norm_period });
        }
        let normalization_length = params.normalization_length.unwrap_or(15);
        if normalization_length == 0 {
            return Err(PossibleRsiError::InvalidNormalizationLength {
                normalization_length,
            });
        }
        let nonlag_period = params.nonlag_period.unwrap_or(15);
        if nonlag_period == 0 {
            return Err(PossibleRsiError::InvalidNonlagPeriod { nonlag_period });
        }
        let dynamic_zone_period = params.dynamic_zone_period.unwrap_or(20);
        if dynamic_zone_period == 0 {
            return Err(PossibleRsiError::InvalidDynamicZonePeriod {
                dynamic_zone_period,
            });
        }
        let highpass_period = params.highpass_period.unwrap_or(15);
        if highpass_period == 0 {
            return Err(PossibleRsiError::InvalidHighpassPeriod { highpass_period });
        }
        let buy_probability = params.buy_probability.unwrap_or(0.2);
        if !buy_probability.is_finite() || !(0.0..=0.5).contains(&buy_probability) {
            return Err(PossibleRsiError::InvalidBuyProbability { buy_probability });
        }
        let sell_probability = params.sell_probability.unwrap_or(0.2);
        if !sell_probability.is_finite() || !(0.0..=0.5).contains(&sell_probability) {
            return Err(PossibleRsiError::InvalidSellProbability { sell_probability });
        }
        Ok(PossibleRsiResolved {
            period,
            rsi_mode: PossibleRsiMode::from_str(params.rsi_mode.as_deref().unwrap_or("regular"))?,
            norm_period,
            normalization_mode: PossibleRsiNormalizationMode::from_str(
                params
                    .normalization_mode
                    .as_deref()
                    .unwrap_or("gaussian_fisher"),
            )?,
            normalization_length,
            nonlag_period,
            dynamic_zone_period,
            buy_probability,
            sell_probability,
            signal_type: PossibleRsiSignalType::from_str(
                params
                    .signal_type
                    .as_deref()
                    .unwrap_or("zeroline_crossover"),
            )?,
            run_highpass: params.run_highpass.unwrap_or(false),
            highpass_period,
        })
    }

    #[inline(always)]
    fn input_slice<'a>(input: &'a PossibleRsiInput<'a>) -> &'a [f64] {
        match &input.data {
            PossibleRsiData::Candles { candles, source } => source_type(candles, source),
            PossibleRsiData::Slice(data) => data,
        }
    }

    #[inline(always)]
    fn validate_common(
        data: &[f64],
        params: &PossibleRsiParams,
    ) -> Result<PossibleRsiResolved, PossibleRsiError> {
        if data.is_empty() {
            return Err(PossibleRsiError::EmptyInputData);
        }
        let resolved = resolve_params(params)?;
        let valid = longest_valid_run(data);
        if valid == 0 {
            return Err(PossibleRsiError::AllValuesNaN);
        }
        let needed = resolved
            .period
            .saturating_add(resolved.norm_period)
            .saturating_add(resolved.normalization_length)
            .saturating_add(resolved.dynamic_zone_period);
        if valid < needed {
            return Err(PossibleRsiError::NotEnoughValidData { needed, valid });
        }
        Ok(resolved)
    }

    #[inline(always)]
    fn fill_outputs(
        data: &[f64],
        stream: &mut PossibleRsiStream,
        value: &mut [f64],
        buy_level: &mut [f64],
        sell_level: &mut [f64],
        middle_level: &mut [f64],
        state: &mut [f64],
        long_signal: &mut [f64],
        short_signal: &mut [f64],
    ) {
        for i in 0..data.len() {
            match stream.update(data[i]) {
                Some(point) => {
                    value[i] = point.value;
                    buy_level[i] = point.buy_level;
                    sell_level[i] = point.sell_level;
                    middle_level[i] = point.middle_level;
                    state[i] = point.state;
                    long_signal[i] = point.long_signal;
                    short_signal[i] = point.short_signal;
                }
                None => {
                    value[i] = f64::NAN;
                    buy_level[i] = f64::NAN;
                    sell_level[i] = f64::NAN;
                    middle_level[i] = f64::NAN;
                    state[i] = f64::NAN;
                    long_signal[i] = f64::NAN;
                    short_signal[i] = f64::NAN;
                }
            }
        }
    }

    #[inline]
    pub fn possible_rsi(input: &PossibleRsiInput) -> Result<PossibleRsiOutput, PossibleRsiError> {
        possible_rsi_with_kernel(input, Kernel::Auto)
    }

    pub fn possible_rsi_with_kernel(
        input: &PossibleRsiInput,
        kernel: Kernel,
    ) -> Result<PossibleRsiOutput, PossibleRsiError> {
        let data = input_slice(input);
        let _resolved = validate_common(data, &input.params)?;
        let _chosen = match kernel {
            Kernel::Auto => detect_best_kernel(),
            other => other,
        };
        let mut value = alloc_with_nan_prefix(data.len(), 0);
        let mut buy_level = alloc_with_nan_prefix(data.len(), 0);
        let mut sell_level = alloc_with_nan_prefix(data.len(), 0);
        let mut middle_level = alloc_with_nan_prefix(data.len(), 0);
        let mut state = alloc_with_nan_prefix(data.len(), 0);
        let mut long_signal = alloc_with_nan_prefix(data.len(), 0);
        let mut short_signal = alloc_with_nan_prefix(data.len(), 0);
        value.fill(f64::NAN);
        buy_level.fill(f64::NAN);
        sell_level.fill(f64::NAN);
        middle_level.fill(f64::NAN);
        state.fill(f64::NAN);
        long_signal.fill(f64::NAN);
        short_signal.fill(f64::NAN);
        let mut stream = PossibleRsiStream::try_new(input.params.clone())?;
        fill_outputs(
            data,
            &mut stream,
            &mut value,
            &mut buy_level,
            &mut sell_level,
            &mut middle_level,
            &mut state,
            &mut long_signal,
            &mut short_signal,
        );
        Ok(PossibleRsiOutput {
            value,
            buy_level,
            sell_level,
            middle_level,
            state,
            long_signal,
            short_signal,
        })
    }

    pub fn possible_rsi_into_slice(
        dst_value: &mut [f64],
        dst_buy_level: &mut [f64],
        dst_sell_level: &mut [f64],
        dst_middle_level: &mut [f64],
        dst_state: &mut [f64],
        dst_long_signal: &mut [f64],
        dst_short_signal: &mut [f64],
        input: &PossibleRsiInput,
        kernel: Kernel,
    ) -> Result<(), PossibleRsiError> {
        let data = input_slice(input);
        let _resolved = validate_common(data, &input.params)?;
        if dst_value.len() != data.len()
            || dst_buy_level.len() != data.len()
            || dst_sell_level.len() != data.len()
            || dst_middle_level.len() != data.len()
            || dst_state.len() != data.len()
            || dst_long_signal.len() != data.len()
            || dst_short_signal.len() != data.len()
        {
            return Err(PossibleRsiError::OutputLengthMismatch {
                expected: data.len(),
                got: dst_value.len(),
            });
        }
        let _chosen = match kernel {
            Kernel::Auto => detect_best_kernel(),
            other => other,
        };
        let mut stream = PossibleRsiStream::try_new(input.params.clone())?;
        fill_outputs(
            data,
            &mut stream,
            dst_value,
            dst_buy_level,
            dst_sell_level,
            dst_middle_level,
            dst_state,
            dst_long_signal,
            dst_short_signal,
        );
        Ok(())
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    pub fn possible_rsi_into(
        input: &PossibleRsiInput,
        dst_value: &mut [f64],
        dst_buy_level: &mut [f64],
        dst_sell_level: &mut [f64],
        dst_middle_level: &mut [f64],
        dst_state: &mut [f64],
        dst_long_signal: &mut [f64],
        dst_short_signal: &mut [f64],
    ) -> Result<(), PossibleRsiError> {
        possible_rsi_into_slice(
            dst_value,
            dst_buy_level,
            dst_sell_level,
            dst_middle_level,
            dst_state,
            dst_long_signal,
            dst_short_signal,
            input,
            Kernel::Auto,
        )
    }

    #[derive(Debug, Clone, Copy)]
    pub struct PossibleRsiBatchRange {
        pub period: (usize, usize, usize),
        pub norm_period: (usize, usize, usize),
        pub normalization_length: (usize, usize, usize),
        pub nonlag_period: (usize, usize, usize),
        pub dynamic_zone_period: (usize, usize, usize),
        pub buy_probability: (f64, f64, f64),
        pub sell_probability: (f64, f64, f64),
    }

    impl Default for PossibleRsiBatchRange {
        fn default() -> Self {
            Self {
                period: (32, 32, 0),
                norm_period: (100, 100, 0),
                normalization_length: (15, 15, 0),
                nonlag_period: (15, 15, 0),
                dynamic_zone_period: (20, 20, 0),
                buy_probability: (0.2, 0.2, 0.0),
                sell_probability: (0.2, 0.2, 0.0),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct PossibleRsiBatchOutput {
        pub value: Vec<f64>,
        pub buy_level: Vec<f64>,
        pub sell_level: Vec<f64>,
        pub middle_level: Vec<f64>,
        pub state: Vec<f64>,
        pub long_signal: Vec<f64>,
        pub short_signal: Vec<f64>,
        pub combos: Vec<PossibleRsiParams>,
        pub rows: usize,
        pub cols: usize,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct PossibleRsiBatchBuilder {
        range: PossibleRsiBatchRange,
        fixed: PossibleRsiParams,
        kernel: Kernel,
    }

    impl Default for PossibleRsiBatchBuilder {
        fn default() -> Self {
            Self {
                range: PossibleRsiBatchRange::default(),
                fixed: PossibleRsiParams::default(),
                kernel: Kernel::Auto,
            }
        }
    }

    impl PossibleRsiBatchBuilder {
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
        pub fn rsi_mode(mut self, value: &'static str) -> Self {
            self.fixed.rsi_mode = Some(value.to_string());
            self
        }

        #[inline(always)]
        pub fn normalization_mode(mut self, value: &'static str) -> Self {
            self.fixed.normalization_mode = Some(value.to_string());
            self
        }

        #[inline(always)]
        pub fn signal_type(mut self, value: &'static str) -> Self {
            self.fixed.signal_type = Some(value.to_string());
            self
        }

        #[inline(always)]
        pub fn run_highpass(mut self, value: bool) -> Self {
            self.fixed.run_highpass = Some(value);
            self
        }

        #[inline(always)]
        pub fn highpass_period(mut self, value: usize) -> Self {
            self.fixed.highpass_period = Some(value);
            self
        }

        #[inline(always)]
        pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
            self.range.period = (start, end, step);
            self
        }

        #[inline(always)]
        pub fn apply_slice(self, data: &[f64]) -> Result<PossibleRsiBatchOutput, PossibleRsiError> {
            possible_rsi_batch_with_kernel(data, &self.range, &self.fixed, self.kernel)
        }
    }

    #[inline(always)]
    fn expand_axis_usize(
        field: &'static str,
        range: (usize, usize, usize),
    ) -> Result<Vec<usize>, PossibleRsiError> {
        let (start, end, step) = range;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start > end {
            return Err(PossibleRsiError::InvalidRange {
                field,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        let mut values = Vec::new();
        let mut current = start;
        loop {
            values.push(current);
            if current >= end {
                break;
            }
            let next = current.saturating_add(step);
            if next <= current {
                return Err(PossibleRsiError::InvalidRange {
                    field,
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            current = next.min(end);
            if current == *values.last().unwrap() {
                break;
            }
        }
        Ok(values)
    }

    #[inline(always)]
    fn expand_axis_f64(
        field: &'static str,
        range: (f64, f64, f64),
    ) -> Result<Vec<f64>, PossibleRsiError> {
        let (start, end, step) = range;
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(PossibleRsiError::InvalidRange {
                field,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        if step.abs() <= f64::EPSILON || (start - end).abs() <= f64::EPSILON {
            return Ok(vec![start]);
        }
        if start > end {
            return Err(PossibleRsiError::InvalidRange {
                field,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        let mut values = Vec::new();
        let mut current = start;
        while current <= end + 1e-12 {
            values.push(current.min(end));
            current += step;
            if step <= 0.0 {
                break;
            }
        }
        Ok(values)
    }

    fn expand_grid_checked(
        range: &PossibleRsiBatchRange,
        fixed: &PossibleRsiParams,
    ) -> Result<Vec<PossibleRsiParams>, PossibleRsiError> {
        let periods = expand_axis_usize("period", range.period)?;
        let norm_periods = expand_axis_usize("norm_period", range.norm_period)?;
        let normalization_lengths =
            expand_axis_usize("normalization_length", range.normalization_length)?;
        let nonlag_periods = expand_axis_usize("nonlag_period", range.nonlag_period)?;
        let dynamic_zone_periods =
            expand_axis_usize("dynamic_zone_period", range.dynamic_zone_period)?;
        let buy_probabilities = expand_axis_f64("buy_probability", range.buy_probability)?;
        let sell_probabilities = expand_axis_f64("sell_probability", range.sell_probability)?;
        let mut out = Vec::new();
        for &period in &periods {
            for &norm_period in &norm_periods {
                for &normalization_length in &normalization_lengths {
                    for &nonlag_period in &nonlag_periods {
                        for &dynamic_zone_period in &dynamic_zone_periods {
                            for &buy_probability in &buy_probabilities {
                                for &sell_probability in &sell_probabilities {
                                    out.push(PossibleRsiParams {
                                        period: Some(period),
                                        rsi_mode: fixed.rsi_mode.clone(),
                                        norm_period: Some(norm_period),
                                        normalization_mode: fixed.normalization_mode.clone(),
                                        normalization_length: Some(normalization_length),
                                        nonlag_period: Some(nonlag_period),
                                        dynamic_zone_period: Some(dynamic_zone_period),
                                        buy_probability: Some(buy_probability),
                                        sell_probability: Some(sell_probability),
                                        signal_type: fixed.signal_type.clone(),
                                        run_highpass: fixed.run_highpass,
                                        highpass_period: fixed.highpass_period,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn expand_grid_possible_rsi(
        range: &PossibleRsiBatchRange,
        fixed: &PossibleRsiParams,
    ) -> Vec<PossibleRsiParams> {
        expand_grid_checked(range, fixed).unwrap_or_default()
    }

    pub fn possible_rsi_batch_with_kernel(
        data: &[f64],
        sweep: &PossibleRsiBatchRange,
        fixed: &PossibleRsiParams,
        kernel: Kernel,
    ) -> Result<PossibleRsiBatchOutput, PossibleRsiError> {
        match kernel {
            Kernel::Auto
            | Kernel::Scalar
            | Kernel::ScalarBatch
            | Kernel::Avx2
            | Kernel::Avx2Batch
            | Kernel::Avx512
            | Kernel::Avx512Batch => {}
            other => return Err(PossibleRsiError::InvalidKernelForBatch(other)),
        }
        let combos = expand_grid_checked(sweep, fixed)?;
        let rows = combos.len();
        let cols = data.len();
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| PossibleRsiError::InvalidInput {
                msg: "possible_rsi rows*cols overflow".to_string(),
            })?;
        let _chosen = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };

        let mut value_mu = make_uninit_matrix(rows, cols);
        let mut buy_mu = make_uninit_matrix(rows, cols);
        let mut sell_mu = make_uninit_matrix(rows, cols);
        let mut middle_mu = make_uninit_matrix(rows, cols);
        let mut state_mu = make_uninit_matrix(rows, cols);
        let mut long_mu = make_uninit_matrix(rows, cols);
        let mut short_mu = make_uninit_matrix(rows, cols);
        let warmups = vec![0usize; rows];
        init_matrix_prefixes(&mut value_mu, cols, &warmups);
        init_matrix_prefixes(&mut buy_mu, cols, &warmups);
        init_matrix_prefixes(&mut sell_mu, cols, &warmups);
        init_matrix_prefixes(&mut middle_mu, cols, &warmups);
        init_matrix_prefixes(&mut state_mu, cols, &warmups);
        init_matrix_prefixes(&mut long_mu, cols, &warmups);
        init_matrix_prefixes(&mut short_mu, cols, &warmups);

        let mut value = unsafe {
            Vec::from_raw_parts(
                value_mu.as_mut_ptr() as *mut f64,
                value_mu.len(),
                value_mu.capacity(),
            )
        };
        let mut buy_level = unsafe {
            Vec::from_raw_parts(
                buy_mu.as_mut_ptr() as *mut f64,
                buy_mu.len(),
                buy_mu.capacity(),
            )
        };
        let mut sell_level = unsafe {
            Vec::from_raw_parts(
                sell_mu.as_mut_ptr() as *mut f64,
                sell_mu.len(),
                sell_mu.capacity(),
            )
        };
        let mut middle_level = unsafe {
            Vec::from_raw_parts(
                middle_mu.as_mut_ptr() as *mut f64,
                middle_mu.len(),
                middle_mu.capacity(),
            )
        };
        let mut state = unsafe {
            Vec::from_raw_parts(
                state_mu.as_mut_ptr() as *mut f64,
                state_mu.len(),
                state_mu.capacity(),
            )
        };
        let mut long_signal = unsafe {
            Vec::from_raw_parts(
                long_mu.as_mut_ptr() as *mut f64,
                long_mu.len(),
                long_mu.capacity(),
            )
        };
        let mut short_signal = unsafe {
            Vec::from_raw_parts(
                short_mu.as_mut_ptr() as *mut f64,
                short_mu.len(),
                short_mu.capacity(),
            )
        };
        std::mem::forget(value_mu);
        std::mem::forget(buy_mu);
        std::mem::forget(sell_mu);
        std::mem::forget(middle_mu);
        std::mem::forget(state_mu);
        std::mem::forget(long_mu);
        std::mem::forget(short_mu);
        debug_assert_eq!(value.len(), total);

        let worker = |row: usize,
                      dst_value: &mut [f64],
                      dst_buy: &mut [f64],
                      dst_sell: &mut [f64],
                      dst_middle: &mut [f64],
                      dst_state: &mut [f64],
                      dst_long: &mut [f64],
                      dst_short: &mut [f64]| {
            dst_value.fill(f64::NAN);
            dst_buy.fill(f64::NAN);
            dst_sell.fill(f64::NAN);
            dst_middle.fill(f64::NAN);
            dst_state.fill(f64::NAN);
            dst_long.fill(f64::NAN);
            dst_short.fill(f64::NAN);
            let mut stream = PossibleRsiStream::try_new(combos[row].clone()).expect("valid params");
            fill_outputs(
                data,
                &mut stream,
                dst_value,
                dst_buy,
                dst_sell,
                dst_middle,
                dst_state,
                dst_long,
                dst_short,
            );
        };

        #[cfg(not(target_arch = "wasm32"))]
        if rows > 1 {
            value
                .par_chunks_mut(cols)
                .zip(buy_level.par_chunks_mut(cols))
                .zip(sell_level.par_chunks_mut(cols))
                .zip(middle_level.par_chunks_mut(cols))
                .zip(state.par_chunks_mut(cols))
                .zip(long_signal.par_chunks_mut(cols))
                .zip(short_signal.par_chunks_mut(cols))
                .enumerate()
                .for_each(
                    |(
                        row,
                        (
                            (((((dst_value, dst_buy), dst_sell), dst_middle), dst_state), dst_long),
                            dst_short,
                        ),
                    )| {
                        worker(
                            row, dst_value, dst_buy, dst_sell, dst_middle, dst_state, dst_long,
                            dst_short,
                        );
                    },
                );
        } else {
            for (
                row,
                (
                    (((((dst_value, dst_buy), dst_sell), dst_middle), dst_state), dst_long),
                    dst_short,
                ),
            ) in value
                .chunks_mut(cols)
                .zip(buy_level.chunks_mut(cols))
                .zip(sell_level.chunks_mut(cols))
                .zip(middle_level.chunks_mut(cols))
                .zip(state.chunks_mut(cols))
                .zip(long_signal.chunks_mut(cols))
                .zip(short_signal.chunks_mut(cols))
                .enumerate()
            {
                worker(
                    row, dst_value, dst_buy, dst_sell, dst_middle, dst_state, dst_long, dst_short,
                );
            }
        }

        Ok(PossibleRsiBatchOutput {
            value,
            buy_level,
            sell_level,
            middle_level,
            state,
            long_signal,
            short_signal,
            combos,
            rows,
            cols,
        })
    }
}
