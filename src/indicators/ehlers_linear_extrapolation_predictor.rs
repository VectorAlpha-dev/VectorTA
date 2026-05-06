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
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::f64::consts::PI;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_HIGH_PASS_LENGTH: usize = 125;
const DEFAULT_LOW_PASS_LENGTH: usize = 12;
const DEFAULT_GAIN: f64 = 0.7;
const DEFAULT_BARS_FORWARD: usize = 5;
const HISTORY_LENGTH: usize = 10;
const MAX_BARS_FORWARD: usize = 10;
const FLOAT_TOL: f64 = 1e-12;
const DEFAULT_SIGNAL_MODE: &str = "predict_filter_crosses";

impl<'a> AsRef<[f64]> for EhlersLinearExtrapolationPredictorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersLinearExtrapolationPredictorData::Slice(slice) => slice,
            EhlersLinearExtrapolationPredictorData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhlersLinearExtrapolationPredictorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersLinearExtrapolationPredictorOutput {
    pub prediction: Vec<f64>,
    pub filter: Vec<f64>,
    pub state: Vec<f64>,
    pub go_long: Vec<f64>,
    pub go_short: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersLinearExtrapolationPredictorParams {
    pub high_pass_length: Option<usize>,
    pub low_pass_length: Option<usize>,
    pub gain: Option<f64>,
    pub bars_forward: Option<usize>,
    pub signal_mode: Option<String>,
}

impl Default for EhlersLinearExtrapolationPredictorParams {
    fn default() -> Self {
        Self {
            high_pass_length: Some(DEFAULT_HIGH_PASS_LENGTH),
            low_pass_length: Some(DEFAULT_LOW_PASS_LENGTH),
            gain: Some(DEFAULT_GAIN),
            bars_forward: Some(DEFAULT_BARS_FORWARD),
            signal_mode: Some(DEFAULT_SIGNAL_MODE.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersLinearExtrapolationPredictorInput<'a> {
    pub data: EhlersLinearExtrapolationPredictorData<'a>,
    pub params: EhlersLinearExtrapolationPredictorParams,
}

impl<'a> EhlersLinearExtrapolationPredictorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EhlersLinearExtrapolationPredictorParams,
    ) -> Self {
        Self {
            data: EhlersLinearExtrapolationPredictorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: EhlersLinearExtrapolationPredictorParams) -> Self {
        Self {
            data: EhlersLinearExtrapolationPredictorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            EhlersLinearExtrapolationPredictorParams::default(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignalMode {
    PredictFilterCrosses,
    PredictMiddleCrosses,
    FilterMiddleCrosses,
}

impl SignalMode {
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::PredictFilterCrosses => "predict_filter_crosses",
            Self::PredictMiddleCrosses => "predict_middle_crosses",
            Self::FilterMiddleCrosses => "filter_middle_crosses",
        }
    }

    #[inline]
    fn parse(value: &str) -> Option<Self> {
        let normalized = value
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect::<String>();
        match normalized.as_str() {
            "predictfiltercrosses" | "sm02" => Some(Self::PredictFilterCrosses),
            "predictmiddlecrosses" | "sm03" => Some(Self::PredictMiddleCrosses),
            "filtermiddlecrosses" | "sm04" => Some(Self::FilterMiddleCrosses),
            _ => None,
        }
    }

    #[inline]
    fn state(self, prediction: f64, filter: f64) -> f64 {
        let lhs = match self {
            Self::PredictFilterCrosses | Self::PredictMiddleCrosses => prediction,
            Self::FilterMiddleCrosses => filter,
        };
        let rhs = match self {
            Self::PredictFilterCrosses => filter,
            Self::PredictMiddleCrosses | Self::FilterMiddleCrosses => 0.0,
        };
        signum_with_tol(lhs - rhs)
    }

    #[inline]
    fn go_long(self, prev_prediction: f64, prev_filter: f64, prediction: f64, filter: f64) -> f64 {
        let (prev_lhs, prev_rhs, lhs, rhs) = match self {
            Self::PredictFilterCrosses => (prev_prediction, prev_filter, prediction, filter),
            Self::PredictMiddleCrosses => (prev_prediction, 0.0, prediction, 0.0),
            Self::FilterMiddleCrosses => (prev_filter, 0.0, filter, 0.0),
        };
        if prev_lhs <= prev_rhs && lhs > rhs {
            1.0
        } else {
            0.0
        }
    }

    #[inline]
    fn go_short(self, prev_prediction: f64, prev_filter: f64, prediction: f64, filter: f64) -> f64 {
        let (prev_lhs, prev_rhs, lhs, rhs) = match self {
            Self::PredictFilterCrosses => (prev_prediction, prev_filter, prediction, filter),
            Self::PredictMiddleCrosses => (prev_prediction, 0.0, prediction, 0.0),
            Self::FilterMiddleCrosses => (prev_filter, 0.0, filter, 0.0),
        };
        if prev_lhs >= prev_rhs && lhs < rhs {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersLinearExtrapolationPredictorBuilder {
    high_pass_length: Option<usize>,
    low_pass_length: Option<usize>,
    gain: Option<f64>,
    bars_forward: Option<usize>,
    signal_mode: Option<SignalMode>,
    kernel: Kernel,
}

impl Default for EhlersLinearExtrapolationPredictorBuilder {
    fn default() -> Self {
        Self {
            high_pass_length: None,
            low_pass_length: None,
            gain: None,
            bars_forward: None,
            signal_mode: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersLinearExtrapolationPredictorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn high_pass_length(mut self, high_pass_length: usize) -> Self {
        self.high_pass_length = Some(high_pass_length);
        self
    }

    #[inline]
    pub fn low_pass_length(mut self, low_pass_length: usize) -> Self {
        self.low_pass_length = Some(low_pass_length);
        self
    }

    #[inline]
    pub fn gain(mut self, gain: f64) -> Self {
        self.gain = Some(gain);
        self
    }

    #[inline]
    pub fn bars_forward(mut self, bars_forward: usize) -> Self {
        self.bars_forward = Some(bars_forward);
        self
    }

    #[inline]
    pub fn signal_mode(
        mut self,
        signal_mode: &str,
    ) -> Result<Self, EhlersLinearExtrapolationPredictorError> {
        self.signal_mode = Some(parse_signal_mode(signal_mode)?);
        Ok(self)
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<EhlersLinearExtrapolationPredictorOutput, EhlersLinearExtrapolationPredictorError>
    {
        let input = EhlersLinearExtrapolationPredictorInput::from_candles(
            candles,
            source,
            EhlersLinearExtrapolationPredictorParams {
                high_pass_length: self.high_pass_length,
                low_pass_length: self.low_pass_length,
                gain: self.gain,
                bars_forward: self.bars_forward,
                signal_mode: Some(
                    self.signal_mode
                        .unwrap_or(SignalMode::PredictFilterCrosses)
                        .as_str()
                        .to_string(),
                ),
            },
        );
        ehlers_linear_extrapolation_predictor_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersLinearExtrapolationPredictorOutput, EhlersLinearExtrapolationPredictorError>
    {
        let input = EhlersLinearExtrapolationPredictorInput::from_slice(
            data,
            EhlersLinearExtrapolationPredictorParams {
                high_pass_length: self.high_pass_length,
                low_pass_length: self.low_pass_length,
                gain: self.gain,
                bars_forward: self.bars_forward,
                signal_mode: Some(
                    self.signal_mode
                        .unwrap_or(SignalMode::PredictFilterCrosses)
                        .as_str()
                        .to_string(),
                ),
            },
        );
        ehlers_linear_extrapolation_predictor_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<EhlersLinearExtrapolationPredictorStream, EhlersLinearExtrapolationPredictorError>
    {
        EhlersLinearExtrapolationPredictorStream::try_new(
            EhlersLinearExtrapolationPredictorParams {
                high_pass_length: self.high_pass_length,
                low_pass_length: self.low_pass_length,
                gain: self.gain,
                bars_forward: self.bars_forward,
                signal_mode: Some(
                    self.signal_mode
                        .unwrap_or(SignalMode::PredictFilterCrosses)
                        .as_str()
                        .to_string(),
                ),
            },
        )
    }
}

#[derive(Debug, Error)]
pub enum EhlersLinearExtrapolationPredictorError {
    #[error("ehlers_linear_extrapolation_predictor: Input data slice is empty.")]
    EmptyInputData,
    #[error("ehlers_linear_extrapolation_predictor: All values are NaN.")]
    AllValuesNaN,
    #[error("ehlers_linear_extrapolation_predictor: Invalid high_pass_length: {high_pass_length}")]
    InvalidHighPassLength { high_pass_length: usize },
    #[error("ehlers_linear_extrapolation_predictor: Invalid low_pass_length: {low_pass_length}")]
    InvalidLowPassLength { low_pass_length: usize },
    #[error("ehlers_linear_extrapolation_predictor: Invalid gain: {gain}")]
    InvalidGain { gain: f64 },
    #[error("ehlers_linear_extrapolation_predictor: Invalid bars_forward: {bars_forward}")]
    InvalidBarsForward { bars_forward: usize },
    #[error(
        "ehlers_linear_extrapolation_predictor: Invalid signal_mode: {signal_mode}. Supported: predict_filter_crosses, predict_middle_crosses, filter_middle_crosses"
    )]
    InvalidSignalMode { signal_mode: String },
    #[error(
        "ehlers_linear_extrapolation_predictor: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "ehlers_linear_extrapolation_predictor: Output length mismatch: expected = {expected}, prediction = {prediction_got}, filter = {filter_got}, state = {state_got}, go_long = {go_long_got}, go_short = {go_short_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        prediction_got: usize,
        filter_got: usize,
        state_got: usize,
        go_long_got: usize,
        go_short_got: usize,
    },
    #[error(
        "ehlers_linear_extrapolation_predictor: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ehlers_linear_extrapolation_predictor: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
struct ResolvedParams {
    high_pass_length: usize,
    low_pass_length: usize,
    gain: f64,
    bars_forward: usize,
    signal_mode: SignalMode,
    hp_c1: f64,
    hp_c2: f64,
    hp_c3: f64,
}

#[derive(Debug, Clone)]
pub struct EhlersLinearExtrapolationPredictorStream {
    params: ResolvedParams,
    source_count: usize,
    prev_source_1: f64,
    prev_source_2: f64,
    hp_prev_1: f64,
    hp_prev_2: f64,
    hp_history: VecDeque<f64>,
    hann_weights: Vec<f64>,
    hann_weight_sum: f64,
    filter_history: VecDeque<f64>,
    prev_prediction: Option<f64>,
    prev_filter: Option<f64>,
}

impl EhlersLinearExtrapolationPredictorStream {
    pub fn try_new(
        params: EhlersLinearExtrapolationPredictorParams,
    ) -> Result<Self, EhlersLinearExtrapolationPredictorError> {
        let params = resolve_params(&params)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        let low_pass_length = params.low_pass_length;
        let (hann_weights, hann_weight_sum) = hann_weights(low_pass_length);
        Self {
            params,
            source_count: 0,
            prev_source_1: 0.0,
            prev_source_2: 0.0,
            hp_prev_1: 0.0,
            hp_prev_2: 0.0,
            hp_history: VecDeque::with_capacity(low_pass_length),
            hann_weights,
            hann_weight_sum,
            filter_history: VecDeque::with_capacity(HISTORY_LENGTH),
            prev_prediction: None,
            prev_filter: None,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new_resolved(self.params.clone());
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        warmup_period(self.params.low_pass_length)
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        self.source_count += 1;
        let hp = if self.source_count <= 4 {
            0.0
        } else {
            self.params.hp_c1 * (value - 2.0 * self.prev_source_1 + self.prev_source_2)
                + self.params.hp_c2 * self.hp_prev_1
                + self.params.hp_c3 * self.hp_prev_2
        };

        self.prev_source_2 = self.prev_source_1;
        self.prev_source_1 = value;
        self.hp_prev_2 = self.hp_prev_1;
        self.hp_prev_1 = hp;

        if self.hp_history.len() == self.params.low_pass_length {
            self.hp_history.pop_back();
        }
        self.hp_history.push_front(hp);

        if self.source_count < 4 + self.params.low_pass_length - 1
            || self.hp_history.len() < self.params.low_pass_length
        {
            return None;
        }

        let filter = self
            .hann_weights
            .iter()
            .zip(self.hp_history.iter())
            .map(|(coef, sample)| coef * sample)
            .sum::<f64>()
            / self.hann_weight_sum;

        if self.filter_history.len() == HISTORY_LENGTH {
            self.filter_history.pop_front();
        }
        self.filter_history.push_back(filter);
        if self.filter_history.len() < HISTORY_LENGTH {
            return None;
        }

        let prediction = extrapolate_prediction(
            &self.filter_history,
            self.params.bars_forward,
            self.params.gain,
        );
        let state = self.params.signal_mode.state(prediction, filter);

        let (go_long, go_short) = if let (Some(prev_prediction), Some(prev_filter)) =
            (self.prev_prediction, self.prev_filter)
        {
            (
                self.params
                    .signal_mode
                    .go_long(prev_prediction, prev_filter, prediction, filter),
                self.params
                    .signal_mode
                    .go_short(prev_prediction, prev_filter, prediction, filter),
            )
        } else {
            (0.0, 0.0)
        };

        self.prev_prediction = Some(prediction);
        self.prev_filter = Some(filter);

        Some((prediction, filter, state, go_long, go_short))
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersLinearExtrapolationPredictorBatchRange {
    pub high_pass_length: (usize, usize, usize),
    pub low_pass_length: (usize, usize, usize),
    pub gain: (f64, f64, f64),
    pub bars_forward: (usize, usize, usize),
    pub signal_mode: Option<String>,
}

impl Default for EhlersLinearExtrapolationPredictorBatchRange {
    fn default() -> Self {
        Self {
            high_pass_length: (DEFAULT_HIGH_PASS_LENGTH, DEFAULT_HIGH_PASS_LENGTH, 0),
            low_pass_length: (DEFAULT_LOW_PASS_LENGTH, DEFAULT_LOW_PASS_LENGTH, 0),
            gain: (DEFAULT_GAIN, DEFAULT_GAIN, 0.0),
            bars_forward: (DEFAULT_BARS_FORWARD, DEFAULT_BARS_FORWARD, 0),
            signal_mode: Some(DEFAULT_SIGNAL_MODE.to_string()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EhlersLinearExtrapolationPredictorBatchOutput {
    pub prediction: Vec<f64>,
    pub filter: Vec<f64>,
    pub state: Vec<f64>,
    pub go_long: Vec<f64>,
    pub go_short: Vec<f64>,
    pub combos: Vec<EhlersLinearExtrapolationPredictorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct EhlersLinearExtrapolationPredictorBatchBuilder {
    range: EhlersLinearExtrapolationPredictorBatchRange,
    kernel: Kernel,
}

impl EhlersLinearExtrapolationPredictorBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn high_pass_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.high_pass_length = (start, end, step);
        self
    }

    #[inline]
    pub fn low_pass_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.low_pass_length = (start, end, step);
        self
    }

    #[inline]
    pub fn gain_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.gain = (start, end, step);
        self
    }

    #[inline]
    pub fn bars_forward_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.bars_forward = (start, end, step);
        self
    }

    #[inline]
    pub fn signal_mode(
        mut self,
        signal_mode: &str,
    ) -> Result<Self, EhlersLinearExtrapolationPredictorError> {
        self.range.signal_mode = Some(parse_signal_mode(signal_mode)?.as_str().to_string());
        Ok(self)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<
        EhlersLinearExtrapolationPredictorBatchOutput,
        EhlersLinearExtrapolationPredictorError,
    > {
        ehlers_linear_extrapolation_predictor_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<
        EhlersLinearExtrapolationPredictorBatchOutput,
        EhlersLinearExtrapolationPredictorError,
    > {
        self.apply_slice(source_type(candles, source))
    }
}

#[inline]
pub fn ehlers_linear_extrapolation_predictor(
    input: &EhlersLinearExtrapolationPredictorInput,
) -> Result<EhlersLinearExtrapolationPredictorOutput, EhlersLinearExtrapolationPredictorError> {
    ehlers_linear_extrapolation_predictor_with_kernel(input, Kernel::Scalar)
}

#[inline]
fn signum_with_tol(value: f64) -> f64 {
    if value > FLOAT_TOL {
        1.0
    } else if value < -FLOAT_TOL {
        -1.0
    } else {
        0.0
    }
}

#[inline]
fn parse_signal_mode(value: &str) -> Result<SignalMode, EhlersLinearExtrapolationPredictorError> {
    SignalMode::parse(value).ok_or_else(|| {
        EhlersLinearExtrapolationPredictorError::InvalidSignalMode {
            signal_mode: value.to_string(),
        }
    })
}

#[inline]
fn resolve_params(
    params: &EhlersLinearExtrapolationPredictorParams,
) -> Result<ResolvedParams, EhlersLinearExtrapolationPredictorError> {
    let high_pass_length = params.high_pass_length.unwrap_or(DEFAULT_HIGH_PASS_LENGTH);
    if high_pass_length == 0 {
        return Err(
            EhlersLinearExtrapolationPredictorError::InvalidHighPassLength { high_pass_length },
        );
    }

    let low_pass_length = params.low_pass_length.unwrap_or(DEFAULT_LOW_PASS_LENGTH);
    if low_pass_length == 0 {
        return Err(
            EhlersLinearExtrapolationPredictorError::InvalidLowPassLength { low_pass_length },
        );
    }

    let gain = params.gain.unwrap_or(DEFAULT_GAIN);
    if !gain.is_finite() {
        return Err(EhlersLinearExtrapolationPredictorError::InvalidGain { gain });
    }

    let bars_forward = params.bars_forward.unwrap_or(DEFAULT_BARS_FORWARD);
    if bars_forward > MAX_BARS_FORWARD {
        return Err(EhlersLinearExtrapolationPredictorError::InvalidBarsForward { bars_forward });
    }

    let signal_mode =
        parse_signal_mode(params.signal_mode.as_deref().unwrap_or(DEFAULT_SIGNAL_MODE))?;

    let angle = 1.414 * PI / high_pass_length as f64;
    let a1 = (-angle).exp();
    let b1 = 2.0 * a1 * angle.cos();
    let hp_c2 = b1;
    let hp_c3 = -a1 * a1;
    let hp_c1 = (1.0 + hp_c2 - hp_c3) * 0.25;

    Ok(ResolvedParams {
        high_pass_length,
        low_pass_length,
        gain,
        bars_forward,
        signal_mode,
        hp_c1,
        hp_c2,
        hp_c3,
    })
}

#[inline]
fn warmup_period(low_pass_length: usize) -> usize {
    low_pass_length + 11
}

#[inline]
fn first_valid_value(data: &[f64]) -> usize {
    data.iter()
        .position(|v| v.is_finite())
        .unwrap_or(data.len())
}

#[inline]
fn count_valid_values(data: &[f64]) -> usize {
    data.iter().filter(|&&v| v.is_finite()).count()
}

#[inline]
fn hann_weights(period: usize) -> (Vec<f64>, f64) {
    let mut weights = Vec::with_capacity(period);
    let pix2 = 2.0 * PI / (period + 1) as f64;
    let mut sum = 0.0;
    for count in 1..=period {
        let coef = 1.0 - (count as f64 * pix2).cos();
        weights.push(coef);
        sum += coef;
    }
    (weights, sum)
}

#[inline]
fn extrapolate_prediction(history: &VecDeque<f64>, bars_forward: usize, gain: f64) -> f64 {
    let current = history[HISTORY_LENGTH - 1];
    if bars_forward == 0 {
        return current * gain;
    }
    let prev = history[HISTORY_LENGTH - 2];
    let step = current - prev;
    (current + bars_forward as f64 * step) * gain
}

#[inline(always)]
fn write_predictor_nan(
    i: usize,
    prediction_out: &mut [f64],
    filter_out: &mut [f64],
    state_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) {
    prediction_out[i] = f64::NAN;
    filter_out[i] = f64::NAN;
    state_out[i] = f64::NAN;
    go_long_out[i] = f64::NAN;
    go_short_out[i] = f64::NAN;
}

#[inline]
fn row_from_slice_lowpass12(
    data: &[f64],
    params: ResolvedParams,
    prediction_out: &mut [f64],
    filter_out: &mut [f64],
    state_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) {
    let pix2 = 2.0 * PI / (DEFAULT_LOW_PASS_LENGTH + 1) as f64;
    let mut weights = [0.0; DEFAULT_LOW_PASS_LENGTH];
    let mut weight_sum = 0.0;
    for count in 1..=DEFAULT_LOW_PASS_LENGTH {
        let coef = 1.0 - (count as f64 * pix2).cos();
        weights[count - 1] = coef;
        weight_sum += coef;
    }

    let mut source_count = 0usize;
    let mut prev_source_1 = 0.0;
    let mut prev_source_2 = 0.0;
    let mut hp_prev_1 = 0.0;
    let mut hp_prev_2 = 0.0;
    let mut hp_history = [0.0; DEFAULT_LOW_PASS_LENGTH];
    let mut hp_len = 0usize;
    let mut filter_history = [0.0; HISTORY_LENGTH];
    let mut filter_len = 0usize;
    let mut prev_prediction = 0.0;
    let mut prev_filter = 0.0;
    let mut have_prev = false;

    for i in 0..data.len() {
        let value = data[i];
        if !value.is_finite() {
            source_count = 0;
            prev_source_1 = 0.0;
            prev_source_2 = 0.0;
            hp_prev_1 = 0.0;
            hp_prev_2 = 0.0;
            hp_len = 0;
            filter_len = 0;
            have_prev = false;
            write_predictor_nan(
                i,
                prediction_out,
                filter_out,
                state_out,
                go_long_out,
                go_short_out,
            );
            continue;
        }

        source_count += 1;
        let hp = if source_count <= 4 {
            0.0
        } else {
            params.hp_c1 * (value - 2.0 * prev_source_1 + prev_source_2)
                + params.hp_c2 * hp_prev_1
                + params.hp_c3 * hp_prev_2
        };

        prev_source_2 = prev_source_1;
        prev_source_1 = value;
        hp_prev_2 = hp_prev_1;
        hp_prev_1 = hp;

        let mut j = DEFAULT_LOW_PASS_LENGTH - 1;
        while j > 0 {
            hp_history[j] = hp_history[j - 1];
            j -= 1;
        }
        hp_history[0] = hp;
        if hp_len < DEFAULT_LOW_PASS_LENGTH {
            hp_len += 1;
        }

        if source_count < 4 + DEFAULT_LOW_PASS_LENGTH - 1 || hp_len < DEFAULT_LOW_PASS_LENGTH {
            write_predictor_nan(
                i,
                prediction_out,
                filter_out,
                state_out,
                go_long_out,
                go_short_out,
            );
            continue;
        }

        let mut filter = 0.0;
        for j in 0..DEFAULT_LOW_PASS_LENGTH {
            filter += weights[j] * hp_history[j];
        }
        filter /= weight_sum;

        if filter_len < HISTORY_LENGTH {
            filter_history[filter_len] = filter;
            filter_len += 1;
        } else {
            for j in 0..(HISTORY_LENGTH - 1) {
                filter_history[j] = filter_history[j + 1];
            }
            filter_history[HISTORY_LENGTH - 1] = filter;
        }

        if filter_len < HISTORY_LENGTH {
            write_predictor_nan(
                i,
                prediction_out,
                filter_out,
                state_out,
                go_long_out,
                go_short_out,
            );
            continue;
        }

        let current = filter_history[HISTORY_LENGTH - 1];
        let previous = filter_history[HISTORY_LENGTH - 2];
        let prediction =
            (current + params.bars_forward as f64 * (current - previous)) * params.gain;
        let filter = current;
        let state = params.signal_mode.state(prediction, filter);
        let (go_long, go_short) = if have_prev {
            (
                params
                    .signal_mode
                    .go_long(prev_prediction, prev_filter, prediction, filter),
                params
                    .signal_mode
                    .go_short(prev_prediction, prev_filter, prediction, filter),
            )
        } else {
            (0.0, 0.0)
        };

        prev_prediction = prediction;
        prev_filter = filter;
        have_prev = true;

        prediction_out[i] = prediction;
        filter_out[i] = filter;
        state_out[i] = state;
        go_long_out[i] = go_long;
        go_short_out[i] = go_short;
    }
}

#[inline]
fn prepare<'a>(
    input: &'a EhlersLinearExtrapolationPredictorInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, usize, Kernel), EhlersLinearExtrapolationPredictorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(EhlersLinearExtrapolationPredictorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(EhlersLinearExtrapolationPredictorError::AllValuesNaN);
    }
    let params = resolve_params(&input.params)?;
    let needed = warmup_period(params.low_pass_length) + 1;
    let valid = count_valid_values(data);
    if valid < needed {
        return Err(EhlersLinearExtrapolationPredictorError::NotEnoughValidData { needed, valid });
    }
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    Ok((data, params, first, chosen))
}

#[inline]
fn row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    prediction_out: &mut [f64],
    filter_out: &mut [f64],
    state_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) {
    if params.low_pass_length == DEFAULT_LOW_PASS_LENGTH {
        row_from_slice_lowpass12(
            data,
            params,
            prediction_out,
            filter_out,
            state_out,
            go_long_out,
            go_short_out,
        );
        return;
    }

    let mut stream = EhlersLinearExtrapolationPredictorStream::new_resolved(params);
    for i in 0..data.len() {
        match stream.update(data[i]) {
            Some((prediction, filter, state, go_long, go_short)) => {
                prediction_out[i] = prediction;
                filter_out[i] = filter;
                state_out[i] = state;
                go_long_out[i] = go_long;
                go_short_out[i] = go_short;
            }
            None => {
                prediction_out[i] = f64::NAN;
                filter_out[i] = f64::NAN;
                state_out[i] = f64::NAN;
                go_long_out[i] = f64::NAN;
                go_short_out[i] = f64::NAN;
            }
        }
    }
}

#[inline]
pub fn ehlers_linear_extrapolation_predictor_with_kernel(
    input: &EhlersLinearExtrapolationPredictorInput,
    kernel: Kernel,
) -> Result<EhlersLinearExtrapolationPredictorOutput, EhlersLinearExtrapolationPredictorError> {
    let (data, params, _first, _chosen) = prepare(input, kernel)?;
    let mut prediction = alloc_uninit_f64(data.len());
    let mut filter = alloc_uninit_f64(data.len());
    let mut state = alloc_uninit_f64(data.len());
    let mut go_long = alloc_uninit_f64(data.len());
    let mut go_short = alloc_uninit_f64(data.len());
    row_from_slice(
        data,
        params,
        &mut prediction,
        &mut filter,
        &mut state,
        &mut go_long,
        &mut go_short,
    );
    Ok(EhlersLinearExtrapolationPredictorOutput {
        prediction,
        filter,
        state,
        go_long,
        go_short,
    })
}

#[inline]
pub fn ehlers_linear_extrapolation_predictor_into_slices(
    prediction_out: &mut [f64],
    filter_out: &mut [f64],
    state_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
    input: &EhlersLinearExtrapolationPredictorInput,
    kernel: Kernel,
) -> Result<(), EhlersLinearExtrapolationPredictorError> {
    let (data, params, _first, _chosen) = prepare(input, kernel)?;
    let len = data.len();
    if prediction_out.len() != len
        || filter_out.len() != len
        || state_out.len() != len
        || go_long_out.len() != len
        || go_short_out.len() != len
    {
        return Err(
            EhlersLinearExtrapolationPredictorError::OutputLengthMismatch {
                expected: len,
                prediction_got: prediction_out.len(),
                filter_got: filter_out.len(),
                state_got: state_out.len(),
                go_long_got: go_long_out.len(),
                go_short_got: go_short_out.len(),
            },
        );
    }
    row_from_slice(
        data,
        params,
        prediction_out,
        filter_out,
        state_out,
        go_long_out,
        go_short_out,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ehlers_linear_extrapolation_predictor_into(
    input: &EhlersLinearExtrapolationPredictorInput,
    prediction_out: &mut [f64],
    filter_out: &mut [f64],
    state_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) -> Result<(), EhlersLinearExtrapolationPredictorError> {
    ehlers_linear_extrapolation_predictor_into_slices(
        prediction_out,
        filter_out,
        state_out,
        go_long_out,
        go_short_out,
        input,
        Kernel::Auto,
    )
}

#[inline]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, EhlersLinearExtrapolationPredictorError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(EhlersLinearExtrapolationPredictorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_add(step);
            if next == x || next > end {
                break;
            }
            x = next;
        }
    } else {
        let mut x = start;
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step);
            if next == x || next < end {
                break;
            }
            x = next;
        }
    }
    if out.is_empty() {
        return Err(EhlersLinearExtrapolationPredictorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline]
fn expand_axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, EhlersLinearExtrapolationPredictorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(EhlersLinearExtrapolationPredictorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() < 1e-12 || step.abs() < 1e-12 {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let st = step.abs();
        let mut x = start;
        while x <= end + 1e-12 {
            out.push(x);
            x += st;
        }
    } else {
        let st = -step.abs();
        let mut x = start;
        while x >= end - 1e-12 {
            out.push(x);
            x += st;
        }
    }
    if out.is_empty() {
        return Err(EhlersLinearExtrapolationPredictorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline]
fn expand_grid_ehlers_linear_extrapolation_predictor(
    range: &EhlersLinearExtrapolationPredictorBatchRange,
) -> Result<Vec<EhlersLinearExtrapolationPredictorParams>, EhlersLinearExtrapolationPredictorError>
{
    let high_pass_lengths = expand_axis_usize(range.high_pass_length)?;
    let low_pass_lengths = expand_axis_usize(range.low_pass_length)?;
    let gains = expand_axis_f64(range.gain)?;
    let bars_forwards = expand_axis_usize(range.bars_forward)?;
    let signal_mode =
        parse_signal_mode(range.signal_mode.as_deref().unwrap_or(DEFAULT_SIGNAL_MODE))?;

    let mut out = Vec::with_capacity(
        high_pass_lengths.len() * low_pass_lengths.len() * gains.len() * bars_forwards.len(),
    );
    for &high_pass_length in &high_pass_lengths {
        if high_pass_length == 0 {
            return Err(
                EhlersLinearExtrapolationPredictorError::InvalidHighPassLength { high_pass_length },
            );
        }
        for &low_pass_length in &low_pass_lengths {
            if low_pass_length == 0 {
                return Err(
                    EhlersLinearExtrapolationPredictorError::InvalidLowPassLength {
                        low_pass_length,
                    },
                );
            }
            for &gain in &gains {
                if !gain.is_finite() {
                    return Err(EhlersLinearExtrapolationPredictorError::InvalidGain { gain });
                }
                for &bars_forward in &bars_forwards {
                    if bars_forward > MAX_BARS_FORWARD {
                        return Err(
                            EhlersLinearExtrapolationPredictorError::InvalidBarsForward {
                                bars_forward,
                            },
                        );
                    }
                    out.push(EhlersLinearExtrapolationPredictorParams {
                        high_pass_length: Some(high_pass_length),
                        low_pass_length: Some(low_pass_length),
                        gain: Some(gain),
                        bars_forward: Some(bars_forward),
                        signal_mode: Some(signal_mode.as_str().to_string()),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn ehlers_linear_extrapolation_predictor_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersLinearExtrapolationPredictorBatchRange,
    kernel: Kernel,
) -> Result<EhlersLinearExtrapolationPredictorBatchOutput, EhlersLinearExtrapolationPredictorError>
{
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(EhlersLinearExtrapolationPredictorError::InvalidKernelForBatch(other)),
    };
    ehlers_linear_extrapolation_predictor_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn ehlers_linear_extrapolation_predictor_batch_slice(
    data: &[f64],
    sweep: &EhlersLinearExtrapolationPredictorBatchRange,
    kernel: Kernel,
) -> Result<EhlersLinearExtrapolationPredictorBatchOutput, EhlersLinearExtrapolationPredictorError>
{
    ehlers_linear_extrapolation_predictor_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn ehlers_linear_extrapolation_predictor_batch_par_slice(
    data: &[f64],
    sweep: &EhlersLinearExtrapolationPredictorBatchRange,
    kernel: Kernel,
) -> Result<EhlersLinearExtrapolationPredictorBatchOutput, EhlersLinearExtrapolationPredictorError>
{
    ehlers_linear_extrapolation_predictor_batch_inner(data, sweep, kernel, true)
}

fn ehlers_linear_extrapolation_predictor_batch_inner(
    data: &[f64],
    sweep: &EhlersLinearExtrapolationPredictorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<EhlersLinearExtrapolationPredictorBatchOutput, EhlersLinearExtrapolationPredictorError>
{
    let combos = expand_grid_ehlers_linear_extrapolation_predictor(sweep)?;
    if data.is_empty() {
        return Err(EhlersLinearExtrapolationPredictorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(EhlersLinearExtrapolationPredictorError::AllValuesNaN);
    }
    let valid = count_valid_values(data);
    let max_needed = combos
        .iter()
        .map(|combo| warmup_period(combo.low_pass_length.unwrap_or(DEFAULT_LOW_PASS_LENGTH)) + 1)
        .max()
        .unwrap_or(0);
    if valid < max_needed {
        return Err(
            EhlersLinearExtrapolationPredictorError::NotEnoughValidData {
                needed: max_needed,
                valid,
            },
        );
    }

    let rows = combos.len();
    let cols = data.len();
    let total = rows.checked_mul(cols).ok_or(
        EhlersLinearExtrapolationPredictorError::OutputLengthMismatch {
            expected: usize::MAX,
            prediction_got: 0,
            filter_got: 0,
            state_got: 0,
            go_long_got: 0,
            go_short_got: 0,
        },
    )?;
    let warmups = combos
        .iter()
        .map(|combo| {
            first + warmup_period(combo.low_pass_length.unwrap_or(DEFAULT_LOW_PASS_LENGTH))
        })
        .collect::<Vec<_>>();

    let mut pred_mu = make_uninit_matrix(rows, cols);
    let mut filter_mu = make_uninit_matrix(rows, cols);
    let mut state_mu = make_uninit_matrix(rows, cols);
    let mut long_mu = make_uninit_matrix(rows, cols);
    let mut short_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut pred_mu, cols, &warmups);
    init_matrix_prefixes(&mut filter_mu, cols, &warmups);
    init_matrix_prefixes(&mut state_mu, cols, &warmups);
    init_matrix_prefixes(&mut long_mu, cols, &warmups);
    init_matrix_prefixes(&mut short_mu, cols, &warmups);

    let mut pred_guard = ManuallyDrop::new(pred_mu);
    let mut filter_guard = ManuallyDrop::new(filter_mu);
    let mut state_guard = ManuallyDrop::new(state_mu);
    let mut long_guard = ManuallyDrop::new(long_mu);
    let mut short_guard = ManuallyDrop::new(short_mu);

    let pred_out =
        unsafe { std::slice::from_raw_parts_mut(pred_guard.as_mut_ptr() as *mut f64, total) };
    let filter_out =
        unsafe { std::slice::from_raw_parts_mut(filter_guard.as_mut_ptr() as *mut f64, total) };
    let state_out =
        unsafe { std::slice::from_raw_parts_mut(state_guard.as_mut_ptr() as *mut f64, total) };
    let long_out =
        unsafe { std::slice::from_raw_parts_mut(long_guard.as_mut_ptr() as *mut f64, total) };
    let short_out =
        unsafe { std::slice::from_raw_parts_mut(short_guard.as_mut_ptr() as *mut f64, total) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        pred_out
            .par_chunks_mut(cols)
            .zip(filter_out.par_chunks_mut(cols))
            .zip(state_out.par_chunks_mut(cols))
            .zip(long_out.par_chunks_mut(cols))
            .zip(short_out.par_chunks_mut(cols))
            .zip(combos.par_iter())
            .for_each(
                |(((((dst_pred, dst_filter), dst_state), dst_long), dst_short), combo)| {
                    let params = resolve_params(combo).unwrap();
                    row_from_slice(
                        data, params, dst_pred, dst_filter, dst_state, dst_long, dst_short,
                    );
                },
            );

        #[cfg(target_arch = "wasm32")]
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let params = resolve_params(&combos[row])?;
            row_from_slice(
                data,
                params,
                &mut pred_out[start..end],
                &mut filter_out[start..end],
                &mut state_out[start..end],
                &mut long_out[start..end],
                &mut short_out[start..end],
            );
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let params = resolve_params(&combos[row])?;
            row_from_slice(
                data,
                params,
                &mut pred_out[start..end],
                &mut filter_out[start..end],
                &mut state_out[start..end],
                &mut long_out[start..end],
                &mut short_out[start..end],
            );
        }
    }

    let prediction = unsafe {
        Vec::from_raw_parts(
            pred_guard.as_mut_ptr() as *mut f64,
            pred_guard.len(),
            pred_guard.capacity(),
        )
    };
    let filter = unsafe {
        Vec::from_raw_parts(
            filter_guard.as_mut_ptr() as *mut f64,
            filter_guard.len(),
            filter_guard.capacity(),
        )
    };
    let state = unsafe {
        Vec::from_raw_parts(
            state_guard.as_mut_ptr() as *mut f64,
            state_guard.len(),
            state_guard.capacity(),
        )
    };
    let go_long = unsafe {
        Vec::from_raw_parts(
            long_guard.as_mut_ptr() as *mut f64,
            long_guard.len(),
            long_guard.capacity(),
        )
    };
    let go_short = unsafe {
        Vec::from_raw_parts(
            short_guard.as_mut_ptr() as *mut f64,
            short_guard.len(),
            short_guard.capacity(),
        )
    };

    Ok(EhlersLinearExtrapolationPredictorBatchOutput {
        prediction,
        filter,
        state,
        go_long,
        go_short,
        combos,
        rows,
        cols,
    })
}

pub fn ehlers_linear_extrapolation_predictor_batch_inner_into(
    data: &[f64],
    sweep: &EhlersLinearExtrapolationPredictorBatchRange,
    kernel: Kernel,
    prediction_out: &mut [f64],
    filter_out: &mut [f64],
    state_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) -> Result<Vec<EhlersLinearExtrapolationPredictorParams>, EhlersLinearExtrapolationPredictorError>
{
    let out = ehlers_linear_extrapolation_predictor_batch_inner(data, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if prediction_out.len() != total
        || filter_out.len() != total
        || state_out.len() != total
        || go_long_out.len() != total
        || go_short_out.len() != total
    {
        return Err(
            EhlersLinearExtrapolationPredictorError::OutputLengthMismatch {
                expected: total,
                prediction_got: prediction_out.len(),
                filter_got: filter_out.len(),
                state_got: state_out.len(),
                go_long_got: go_long_out.len(),
                go_short_got: go_short_out.len(),
            },
        );
    }
    prediction_out.copy_from_slice(&out.prediction);
    filter_out.copy_from_slice(&out.filter);
    state_out.copy_from_slice(&out.state);
    go_long_out.copy_from_slice(&out.go_long);
    go_short_out.copy_from_slice(&out.go_short);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_linear_extrapolation_predictor")]
#[pyo3(signature = (data, high_pass_length=None, low_pass_length=None, gain=None, bars_forward=None, signal_mode=None, kernel=None))]
pub fn ehlers_linear_extrapolation_predictor_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    high_pass_length: Option<usize>,
    low_pass_length: Option<usize>,
    gain: Option<f64>,
    bars_forward: Option<usize>,
    signal_mode: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = EhlersLinearExtrapolationPredictorInput::from_slice(
        data,
        EhlersLinearExtrapolationPredictorParams {
            high_pass_length,
            low_pass_length,
            gain,
            bars_forward,
            signal_mode: signal_mode.map(|v| v.to_string()),
        },
    );
    let out = py
        .allow_threads(|| ehlers_linear_extrapolation_predictor_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.prediction.into_pyarray(py),
        out.filter.into_pyarray(py),
        out.state.into_pyarray(py),
        out.go_long.into_pyarray(py),
        out.go_short.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersLinearExtrapolationPredictorStream")]
pub struct EhlersLinearExtrapolationPredictorStreamPy {
    inner: EhlersLinearExtrapolationPredictorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersLinearExtrapolationPredictorStreamPy {
    #[new]
    #[pyo3(signature = (high_pass_length=DEFAULT_HIGH_PASS_LENGTH, low_pass_length=DEFAULT_LOW_PASS_LENGTH, gain=DEFAULT_GAIN, bars_forward=DEFAULT_BARS_FORWARD, signal_mode=DEFAULT_SIGNAL_MODE))]
    fn new(
        high_pass_length: usize,
        low_pass_length: usize,
        gain: f64,
        bars_forward: usize,
        signal_mode: &str,
    ) -> PyResult<Self> {
        let inner = EhlersLinearExtrapolationPredictorStream::try_new(
            EhlersLinearExtrapolationPredictorParams {
                high_pass_length: Some(high_pass_length),
                low_pass_length: Some(low_pass_length),
                gain: Some(gain),
                bars_forward: Some(bars_forward),
                signal_mode: Some(signal_mode.to_string()),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64, f64)> {
        self.inner.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_linear_extrapolation_predictor_batch")]
#[pyo3(signature = (
    data,
    high_pass_length_range=(DEFAULT_HIGH_PASS_LENGTH, DEFAULT_HIGH_PASS_LENGTH, 0),
    low_pass_length_range=(DEFAULT_LOW_PASS_LENGTH, DEFAULT_LOW_PASS_LENGTH, 0),
    gain_range=(DEFAULT_GAIN, DEFAULT_GAIN, 0.0),
    bars_forward_range=(DEFAULT_BARS_FORWARD, DEFAULT_BARS_FORWARD, 0),
    signal_mode=DEFAULT_SIGNAL_MODE,
    kernel=None
))]
pub fn ehlers_linear_extrapolation_predictor_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    high_pass_length_range: (usize, usize, usize),
    low_pass_length_range: (usize, usize, usize),
    gain_range: (f64, f64, f64),
    bars_forward_range: (usize, usize, usize),
    signal_mode: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = EhlersLinearExtrapolationPredictorBatchRange {
        high_pass_length: high_pass_length_range,
        low_pass_length: low_pass_length_range,
        gain: gain_range,
        bars_forward: bars_forward_range,
        signal_mode: Some(signal_mode.to_string()),
    };
    let combos = expand_grid_ehlers_linear_extrapolation_predictor(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let prediction_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let filter_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let state_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let long_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let short_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let prediction_slice = unsafe { prediction_arr.as_slice_mut()? };
    let filter_slice = unsafe { filter_arr.as_slice_mut()? };
    let state_slice = unsafe { state_arr.as_slice_mut()? };
    let long_slice = unsafe { long_arr.as_slice_mut()? };
    let short_slice = unsafe { short_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            ehlers_linear_extrapolation_predictor_batch_inner_into(
                data,
                &sweep,
                batch.to_non_batch(),
                prediction_slice,
                filter_slice,
                state_slice,
                long_slice,
                short_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("prediction", prediction_arr.reshape((rows, cols))?)?;
    dict.set_item("filter", filter_arr.reshape((rows, cols))?)?;
    dict.set_item("state", state_arr.reshape((rows, cols))?)?;
    dict.set_item("go_long", long_arr.reshape((rows, cols))?)?;
    dict.set_item("go_short", short_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "high_pass_lengths",
        combos
            .iter()
            .map(|p| p.high_pass_length.unwrap_or(DEFAULT_HIGH_PASS_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "low_pass_lengths",
        combos
            .iter()
            .map(|p| p.low_pass_length.unwrap_or(DEFAULT_LOW_PASS_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "gains",
        combos
            .iter()
            .map(|p| p.gain.unwrap_or(DEFAULT_GAIN))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "bars_forwards",
        combos
            .iter()
            .map(|p| p.bars_forward.unwrap_or(DEFAULT_BARS_FORWARD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_modes",
        combos
            .iter()
            .map(|p| {
                p.signal_mode
                    .clone()
                    .unwrap_or_else(|| DEFAULT_SIGNAL_MODE.to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ehlers_linear_extrapolation_predictor_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        ehlers_linear_extrapolation_predictor_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        ehlers_linear_extrapolation_predictor_batch_py,
        module
    )?)?;
    module.add_class::<EhlersLinearExtrapolationPredictorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_linear_extrapolation_predictor_js")]
pub fn ehlers_linear_extrapolation_predictor_js(
    data: &[f64],
    high_pass_length: usize,
    low_pass_length: usize,
    gain: f64,
    bars_forward: usize,
    signal_mode: &str,
) -> Result<JsValue, JsValue> {
    let input = EhlersLinearExtrapolationPredictorInput::from_slice(
        data,
        EhlersLinearExtrapolationPredictorParams {
            high_pass_length: Some(high_pass_length),
            low_pass_length: Some(low_pass_length),
            gain: Some(gain),
            bars_forward: Some(bars_forward),
            signal_mode: Some(signal_mode.to_string()),
        },
    );
    let out = ehlers_linear_extrapolation_predictor(&input)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Object::new();

    macro_rules! set_arr {
        ($name:literal, $values:expr) => {{
            let arr = js_sys::Float64Array::new_with_length($values.len() as u32);
            arr.copy_from(&$values);
            js_sys::Reflect::set(&result, &JsValue::from_str($name), &arr)?;
        }};
    }

    set_arr!("prediction", out.prediction);
    set_arr!("filter", out.filter);
    set_arr!("state", out.state);
    set_arr!("go_long", out.go_long);
    set_arr!("go_short", out.go_short);
    Ok(result.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_linear_extrapolation_predictor_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_linear_extrapolation_predictor_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_linear_extrapolation_predictor_into(
    data_ptr: *const f64,
    prediction_ptr: *mut f64,
    filter_ptr: *mut f64,
    state_ptr: *mut f64,
    go_long_ptr: *mut f64,
    go_short_ptr: *mut f64,
    len: usize,
    high_pass_length: usize,
    low_pass_length: usize,
    gain: f64,
    bars_forward: usize,
    signal_mode: &str,
) -> Result<(), JsValue> {
    if data_ptr.is_null()
        || prediction_ptr.is_null()
        || filter_ptr.is_null()
        || state_ptr.is_null()
        || go_long_ptr.is_null()
        || go_short_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = EhlersLinearExtrapolationPredictorInput::from_slice(
            data,
            EhlersLinearExtrapolationPredictorParams {
                high_pass_length: Some(high_pass_length),
                low_pass_length: Some(low_pass_length),
                gain: Some(gain),
                bars_forward: Some(bars_forward),
                signal_mode: Some(signal_mode.to_string()),
            },
        );
        let alias = data_ptr == prediction_ptr
            || data_ptr == filter_ptr
            || data_ptr == state_ptr
            || data_ptr == go_long_ptr
            || data_ptr == go_short_ptr
            || prediction_ptr == filter_ptr
            || prediction_ptr == state_ptr
            || prediction_ptr == go_long_ptr
            || prediction_ptr == go_short_ptr
            || filter_ptr == state_ptr
            || filter_ptr == go_long_ptr
            || filter_ptr == go_short_ptr
            || state_ptr == go_long_ptr
            || state_ptr == go_short_ptr
            || go_long_ptr == go_short_ptr;
        if alias {
            let mut prediction_tmp = vec![0.0; len];
            let mut filter_tmp = vec![0.0; len];
            let mut state_tmp = vec![0.0; len];
            let mut long_tmp = vec![0.0; len];
            let mut short_tmp = vec![0.0; len];
            ehlers_linear_extrapolation_predictor_into_slices(
                &mut prediction_tmp,
                &mut filter_tmp,
                &mut state_tmp,
                &mut long_tmp,
                &mut short_tmp,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(prediction_ptr, len).copy_from_slice(&prediction_tmp);
            std::slice::from_raw_parts_mut(filter_ptr, len).copy_from_slice(&filter_tmp);
            std::slice::from_raw_parts_mut(state_ptr, len).copy_from_slice(&state_tmp);
            std::slice::from_raw_parts_mut(go_long_ptr, len).copy_from_slice(&long_tmp);
            std::slice::from_raw_parts_mut(go_short_ptr, len).copy_from_slice(&short_tmp);
        } else {
            ehlers_linear_extrapolation_predictor_into_slices(
                std::slice::from_raw_parts_mut(prediction_ptr, len),
                std::slice::from_raw_parts_mut(filter_ptr, len),
                std::slice::from_raw_parts_mut(state_ptr, len),
                std::slice::from_raw_parts_mut(go_long_ptr, len),
                std::slice::from_raw_parts_mut(go_short_ptr, len),
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersLinearExtrapolationPredictorBatchConfig {
    pub high_pass_length_range: (usize, usize, usize),
    pub low_pass_length_range: (usize, usize, usize),
    pub gain_range: Option<(f64, f64, f64)>,
    pub bars_forward_range: Option<(usize, usize, usize)>,
    pub signal_mode: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersLinearExtrapolationPredictorBatchJsOutput {
    pub prediction: Vec<f64>,
    pub filter: Vec<f64>,
    pub state: Vec<f64>,
    pub go_long: Vec<f64>,
    pub go_short: Vec<f64>,
    pub high_pass_lengths: Vec<usize>,
    pub low_pass_lengths: Vec<usize>,
    pub gains: Vec<f64>,
    pub bars_forwards: Vec<usize>,
    pub signal_modes: Vec<String>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_linear_extrapolation_predictor_batch_js")]
pub fn ehlers_linear_extrapolation_predictor_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EhlersLinearExtrapolationPredictorBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = EhlersLinearExtrapolationPredictorBatchRange {
        high_pass_length: config.high_pass_length_range,
        low_pass_length: config.low_pass_length_range,
        gain: config
            .gain_range
            .unwrap_or((DEFAULT_GAIN, DEFAULT_GAIN, 0.0)),
        bars_forward: config.bars_forward_range.unwrap_or((
            DEFAULT_BARS_FORWARD,
            DEFAULT_BARS_FORWARD,
            0,
        )),
        signal_mode: Some(
            config
                .signal_mode
                .unwrap_or_else(|| DEFAULT_SIGNAL_MODE.to_string()),
        ),
    };
    let out = ehlers_linear_extrapolation_predictor_batch_inner(
        data,
        &sweep,
        detect_best_batch_kernel().to_non_batch(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersLinearExtrapolationPredictorBatchJsOutput {
        high_pass_lengths: out
            .combos
            .iter()
            .map(|p| p.high_pass_length.unwrap_or(DEFAULT_HIGH_PASS_LENGTH))
            .collect(),
        low_pass_lengths: out
            .combos
            .iter()
            .map(|p| p.low_pass_length.unwrap_or(DEFAULT_LOW_PASS_LENGTH))
            .collect(),
        gains: out
            .combos
            .iter()
            .map(|p| p.gain.unwrap_or(DEFAULT_GAIN))
            .collect(),
        bars_forwards: out
            .combos
            .iter()
            .map(|p| p.bars_forward.unwrap_or(DEFAULT_BARS_FORWARD))
            .collect(),
        signal_modes: out
            .combos
            .iter()
            .map(|p| {
                p.signal_mode
                    .clone()
                    .unwrap_or_else(|| DEFAULT_SIGNAL_MODE.to_string())
            })
            .collect(),
        prediction: out.prediction,
        filter: out.filter,
        state: out.state,
        go_long: out.go_long,
        go_short: out.go_short,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_linear_extrapolation_predictor_batch_into(
    data_ptr: *const f64,
    prediction_ptr: *mut f64,
    filter_ptr: *mut f64,
    state_ptr: *mut f64,
    go_long_ptr: *mut f64,
    go_short_ptr: *mut f64,
    len: usize,
    high_pass_start: usize,
    high_pass_end: usize,
    high_pass_step: usize,
    low_pass_start: usize,
    low_pass_end: usize,
    low_pass_step: usize,
    gain_start: f64,
    gain_end: f64,
    gain_step: f64,
    bars_forward_start: usize,
    bars_forward_end: usize,
    bars_forward_step: usize,
    signal_mode: &str,
) -> Result<usize, JsValue> {
    if data_ptr.is_null()
        || prediction_ptr.is_null()
        || filter_ptr.is_null()
        || state_ptr.is_null()
        || go_long_ptr.is_null()
        || go_short_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let sweep = EhlersLinearExtrapolationPredictorBatchRange {
        high_pass_length: (high_pass_start, high_pass_end, high_pass_step),
        low_pass_length: (low_pass_start, low_pass_end, low_pass_step),
        gain: (gain_start, gain_end, gain_step),
        bars_forward: (bars_forward_start, bars_forward_end, bars_forward_step),
        signal_mode: Some(signal_mode.to_string()),
    };
    let combos = expand_grid_ehlers_linear_extrapolation_predictor(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        ehlers_linear_extrapolation_predictor_batch_inner_into(
            data,
            &sweep,
            detect_best_batch_kernel().to_non_batch(),
            std::slice::from_raw_parts_mut(prediction_ptr, total),
            std::slice::from_raw_parts_mut(filter_ptr, total),
            std::slice::from_raw_parts_mut(state_ptr, total),
            std::slice::from_raw_parts_mut(go_long_ptr, total),
            std::slice::from_raw_parts_mut(go_short_ptr, total),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_linear_extrapolation_predictor_output_into_js(
    data: &[f64],
    high_pass_length: usize,
    low_pass_length: usize,
    gain: f64,
    bars_forward: usize,
    signal_mode: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_linear_extrapolation_predictor_js(
        data,
        high_pass_length,
        low_pass_length,
        gain,
        bars_forward,
        signal_mode,
    )?;
    crate::write_wasm_object_f64_outputs(
        "ehlers_linear_extrapolation_predictor_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_linear_extrapolation_predictor_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_linear_extrapolation_predictor_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ehlers_linear_extrapolation_predictor_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn ehlers_linear_extrapolation_predictor_constant_contract() -> Result<(), Box<dyn Error>> {
        let data = [100.0; 24];
        let input = EhlersLinearExtrapolationPredictorInput::from_slice(
            &data,
            EhlersLinearExtrapolationPredictorParams {
                high_pass_length: Some(8),
                low_pass_length: Some(4),
                gain: Some(0.7),
                bars_forward: Some(3),
                signal_mode: Some("predict_filter_crosses".to_string()),
            },
        );
        let out = ehlers_linear_extrapolation_predictor(&input)?;
        assert_eq!(out.prediction.len(), data.len());
        assert!(out.prediction[..15].iter().all(|v| v.is_nan()));
        for i in 15..data.len() {
            assert!(out.prediction[i].abs() <= 1e-12);
            assert!(out.filter[i].abs() <= 1e-12);
            assert!(out.state[i].abs() <= 1e-12);
            assert!(out.go_long[i].abs() <= 1e-12);
            assert!(out.go_short[i].abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn ehlers_linear_extrapolation_predictor_rejects_invalid_bars_forward() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let input = EhlersLinearExtrapolationPredictorInput::from_slice(
            &data,
            EhlersLinearExtrapolationPredictorParams {
                high_pass_length: Some(8),
                low_pass_length: Some(4),
                gain: Some(0.7),
                bars_forward: Some(11),
                signal_mode: Some("predict_filter_crosses".to_string()),
            },
        );
        let err = ehlers_linear_extrapolation_predictor(&input).unwrap_err();
        assert!(matches!(
            err,
            EhlersLinearExtrapolationPredictorError::InvalidBarsForward { bars_forward: 11 }
        ));
    }

    #[test]
    fn ehlers_linear_extrapolation_predictor_stream_matches_batch_with_reset(
    ) -> Result<(), Box<dyn Error>> {
        let data = [
            100.0,
            100.5,
            101.0,
            100.7,
            100.2,
            99.8,
            99.5,
            99.7,
            100.0,
            100.4,
            100.8,
            101.0,
            100.9,
            100.7,
            100.4,
            100.1,
            99.9,
            f64::NAN,
            100.0,
            100.3,
            100.6,
            100.8,
            100.7,
            100.5,
            100.2,
            100.0,
            99.8,
            99.9,
            100.1,
            100.4,
            100.7,
            100.9,
            101.0,
            100.8,
            100.5,
        ];
        let params = EhlersLinearExtrapolationPredictorParams {
            high_pass_length: Some(8),
            low_pass_length: Some(4),
            gain: Some(0.7),
            bars_forward: Some(3),
            signal_mode: Some("predict_middle_crosses".to_string()),
        };
        let batch = ehlers_linear_extrapolation_predictor(
            &EhlersLinearExtrapolationPredictorInput::from_slice(&data, params.clone()),
        )?;
        let mut stream = EhlersLinearExtrapolationPredictorStream::try_new(params)?;
        let mut prediction = Vec::with_capacity(data.len());
        let mut filter = Vec::with_capacity(data.len());
        let mut state = Vec::with_capacity(data.len());
        let mut go_long = Vec::with_capacity(data.len());
        let mut go_short = Vec::with_capacity(data.len());

        for &value in &data {
            if let Some((p, f, s, gl, gs)) = stream.update(value) {
                prediction.push(p);
                filter.push(f);
                state.push(s);
                go_long.push(gl);
                go_short.push(gs);
            } else {
                prediction.push(f64::NAN);
                filter.push(f64::NAN);
                state.push(f64::NAN);
                go_long.push(f64::NAN);
                go_short.push(f64::NAN);
            }
        }

        assert_eq!(stream.get_warmup_period(), 15);
        assert_eq!(batch.prediction.len(), prediction.len());
        for (lhs, rhs) in batch.prediction.iter().zip(prediction.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        for (lhs, rhs) in batch.filter.iter().zip(filter.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        for (lhs, rhs) in batch.state.iter().zip(state.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        for (lhs, rhs) in batch.go_long.iter().zip(go_long.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        for (lhs, rhs) in batch.go_short.iter().zip(go_short.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        Ok(())
    }
}
