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
use std::collections::VecDeque;
use std::error::Error;
use thiserror::Error;

const DEFAULT_LOOKBACK_PERIOD: usize = 12;
const DEFAULT_CONFIRMATION_PERIOD: usize = 3;
const DEFAULT_USE_VOLUME_CONFIRMATION: bool = true;
const DEFAULT_TREND_MA_PERIOD: usize = 50;
const DEFAULT_TREND_MA_TYPE: &str = "EMA";
const DEFAULT_MA_STEP_PERIOD: usize = 33;
const VOLUME_SMA_PERIOD: usize = 20;
const OUTPUTS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrendMaKind {
    Sma,
    Ema,
    Wma,
    Vwma,
}

#[derive(Debug, Clone)]
pub enum ReversalSignalsData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct ReversalSignalsOutput {
    pub buy_signal: Vec<f64>,
    pub sell_signal: Vec<f64>,
    pub stepped_ma: Vec<f64>,
    pub state: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct ReversalSignalsParams {
    pub lookback_period: Option<usize>,
    pub confirmation_period: Option<usize>,
    pub use_volume_confirmation: Option<bool>,
    pub trend_ma_period: Option<usize>,
    pub trend_ma_type: Option<String>,
    pub ma_step_period: Option<usize>,
}

impl Default for ReversalSignalsParams {
    fn default() -> Self {
        Self {
            lookback_period: Some(DEFAULT_LOOKBACK_PERIOD),
            confirmation_period: Some(DEFAULT_CONFIRMATION_PERIOD),
            use_volume_confirmation: Some(DEFAULT_USE_VOLUME_CONFIRMATION),
            trend_ma_period: Some(DEFAULT_TREND_MA_PERIOD),
            trend_ma_type: Some(DEFAULT_TREND_MA_TYPE.to_string()),
            ma_step_period: Some(DEFAULT_MA_STEP_PERIOD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReversalSignalsInput<'a> {
    pub data: ReversalSignalsData<'a>,
    pub params: ReversalSignalsParams,
}

impl<'a> ReversalSignalsInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ReversalSignalsParams) -> Self {
        Self {
            data: ReversalSignalsData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: ReversalSignalsParams,
    ) -> Self {
        Self {
            data: ReversalSignalsData::Slices {
                open,
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
        Self::from_candles(candles, ReversalSignalsParams::default())
    }

    #[inline]
    pub fn get_lookback_period(&self) -> usize {
        self.params
            .lookback_period
            .unwrap_or(DEFAULT_LOOKBACK_PERIOD)
    }

    #[inline]
    pub fn get_confirmation_period(&self) -> usize {
        self.params
            .confirmation_period
            .unwrap_or(DEFAULT_CONFIRMATION_PERIOD)
    }

    #[inline]
    pub fn get_use_volume_confirmation(&self) -> bool {
        self.params
            .use_volume_confirmation
            .unwrap_or(DEFAULT_USE_VOLUME_CONFIRMATION)
    }

    #[inline]
    pub fn get_trend_ma_period(&self) -> usize {
        self.params
            .trend_ma_period
            .unwrap_or(DEFAULT_TREND_MA_PERIOD)
    }

    #[inline]
    pub fn get_trend_ma_type(&self) -> &str {
        self.params
            .trend_ma_type
            .as_deref()
            .unwrap_or(DEFAULT_TREND_MA_TYPE)
    }

    #[inline]
    pub fn get_ma_step_period(&self) -> usize {
        self.params.ma_step_period.unwrap_or(DEFAULT_MA_STEP_PERIOD)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReversalSignalsBuilder {
    lookback_period: Option<usize>,
    confirmation_period: Option<usize>,
    use_volume_confirmation: Option<bool>,
    trend_ma_period: Option<usize>,
    trend_ma_type: Option<TrendMaKind>,
    ma_step_period: Option<usize>,
    kernel: Kernel,
}

impl Default for ReversalSignalsBuilder {
    fn default() -> Self {
        Self {
            lookback_period: None,
            confirmation_period: None,
            use_volume_confirmation: None,
            trend_ma_period: None,
            trend_ma_type: None,
            ma_step_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ReversalSignalsBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn lookback_period(mut self, value: usize) -> Self {
        self.lookback_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn confirmation_period(mut self, value: usize) -> Self {
        self.confirmation_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn use_volume_confirmation(mut self, value: bool) -> Self {
        self.use_volume_confirmation = Some(value);
        self
    }

    #[inline(always)]
    pub fn trend_ma_period(mut self, value: usize) -> Self {
        self.trend_ma_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn trend_ma_type(mut self, value: &str) -> Result<Self, ReversalSignalsError> {
        self.trend_ma_type = Some(parse_trend_ma_kind(value)?);
        Ok(self)
    }

    #[inline(always)]
    pub fn ma_step_period(mut self, value: usize) -> Self {
        self.ma_step_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }
}

#[derive(Debug, Error)]
pub enum ReversalSignalsError {
    #[error("reversal_signals: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "reversal_signals: Input length mismatch: open = {open_len}, high = {high_len}, low = {low_len}, close = {close_len}, volume = {volume_len}"
    )]
    InputLengthMismatch {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error("reversal_signals: All values are NaN.")]
    AllValuesNaN,
    #[error("reversal_signals: Invalid lookback_period: {lookback_period}")]
    InvalidLookbackPeriod { lookback_period: usize },
    #[error("reversal_signals: Invalid trend_ma_period: {trend_ma_period}")]
    InvalidTrendMaPeriod { trend_ma_period: usize },
    #[error("reversal_signals: Invalid trend_ma_type: {value}")]
    InvalidTrendMaType { value: String },
    #[error("reversal_signals: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("reversal_signals: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "reversal_signals: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("reversal_signals: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("reversal_signals: Invalid kernel: {0:?}")]
    InvalidKernel(Kernel),
    #[error("reversal_signals: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("reversal_signals: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[inline(always)]
fn trend_ma_kind_name(kind: TrendMaKind) -> &'static str {
    match kind {
        TrendMaKind::Sma => "SMA",
        TrendMaKind::Ema => "EMA",
        TrendMaKind::Wma => "WMA",
        TrendMaKind::Vwma => "VWMA",
    }
}

#[inline(always)]
fn parse_trend_ma_kind(value: &str) -> Result<TrendMaKind, ReversalSignalsError> {
    if value.eq_ignore_ascii_case("SMA") {
        return Ok(TrendMaKind::Sma);
    }
    if value.eq_ignore_ascii_case("EMA") {
        return Ok(TrendMaKind::Ema);
    }
    if value.eq_ignore_ascii_case("WMA") {
        return Ok(TrendMaKind::Wma);
    }
    if value.eq_ignore_ascii_case("VWMA") {
        return Ok(TrendMaKind::Vwma);
    }
    Err(ReversalSignalsError::InvalidTrendMaType {
        value: value.to_string(),
    })
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a ReversalSignalsInput<'a>,
) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
    match &input.data {
        ReversalSignalsData::Candles { candles } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        ),
        ReversalSignalsData::Slices {
            open,
            high,
            low,
            close,
            volume,
        } => (open, high, low, close, volume),
    }
}

#[inline(always)]
fn is_valid_ohlcv(open: f64, high: f64, low: f64, close: f64, volume: f64) -> bool {
    open.is_finite()
        && high.is_finite()
        && low.is_finite()
        && close.is_finite()
        && volume.is_finite()
}

#[inline(always)]
fn longest_valid_run(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for i in 0..close.len() {
        if is_valid_ohlcv(open[i], high[i], low[i], close[i], volume[i]) {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

#[inline(always)]
fn required_run(
    lookback_period: usize,
    trend_ma_period: usize,
    trend_ma_kind: TrendMaKind,
    use_volume_confirmation: bool,
) -> usize {
    let ma_needed = match trend_ma_kind {
        TrendMaKind::Ema => 1,
        TrendMaKind::Sma | TrendMaKind::Wma | TrendMaKind::Vwma => trend_ma_period,
    };
    let volume_needed = if use_volume_confirmation {
        VOLUME_SMA_PERIOD
    } else {
        1
    };
    lookback_period.max(ma_needed).max(volume_needed)
}

#[inline(always)]
fn validate_params_only(
    lookback_period: usize,
    trend_ma_period: usize,
    trend_ma_type: &str,
) -> Result<TrendMaKind, ReversalSignalsError> {
    if lookback_period == 0 {
        return Err(ReversalSignalsError::InvalidLookbackPeriod { lookback_period });
    }
    if trend_ma_period == 0 {
        return Err(ReversalSignalsError::InvalidTrendMaPeriod { trend_ma_period });
    }
    parse_trend_ma_kind(trend_ma_type)
}

#[inline(always)]
fn validate_common(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    lookback_period: usize,
    trend_ma_period: usize,
    trend_ma_type: &str,
    use_volume_confirmation: bool,
) -> Result<TrendMaKind, ReversalSignalsError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty()
    {
        return Err(ReversalSignalsError::EmptyInputData);
    }
    if open.len() != high.len()
        || open.len() != low.len()
        || open.len() != close.len()
        || open.len() != volume.len()
    {
        return Err(ReversalSignalsError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let trend_ma_kind = validate_params_only(lookback_period, trend_ma_period, trend_ma_type)?;
    let longest = longest_valid_run(open, high, low, close, volume);
    if longest == 0 {
        return Err(ReversalSignalsError::AllValuesNaN);
    }
    let needed = required_run(
        lookback_period,
        trend_ma_period,
        trend_ma_kind,
        use_volume_confirmation,
    );
    if longest < needed {
        return Err(ReversalSignalsError::NotEnoughValidData {
            needed,
            valid: longest,
        });
    }
    Ok(trend_ma_kind)
}

#[inline(always)]
fn normalize_single_kernel(kernel: Kernel) -> Result<Kernel, ReversalSignalsError> {
    match kernel {
        Kernel::Auto => Ok(Kernel::Scalar),
        Kernel::Scalar | Kernel::ScalarBatch => Ok(Kernel::Scalar),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => Ok(Kernel::Avx2),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => Ok(Kernel::Avx512),
        other => Err(ReversalSignalsError::InvalidKernel(other)),
    }
}

#[inline(always)]
fn normalize_batch_kernel(kernel: Kernel) -> Result<Kernel, ReversalSignalsError> {
    match kernel {
        Kernel::Auto => Ok(detect_best_batch_kernel()),
        Kernel::Scalar | Kernel::ScalarBatch => Ok(Kernel::ScalarBatch),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => Ok(Kernel::Avx2Batch),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => Ok(Kernel::Avx512Batch),
        other => Err(ReversalSignalsError::InvalidKernelForBatch(other)),
    }
}

#[derive(Debug, Clone)]
struct RollingSmaState {
    period: usize,
    sum: f64,
    buf: VecDeque<f64>,
}

impl RollingSmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            sum: 0.0,
            buf: VecDeque::with_capacity(period),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.sum = 0.0;
        self.buf.clear();
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        self.buf.push_back(value);
        self.sum += value;
        if self.buf.len() > self.period {
            if let Some(old) = self.buf.pop_front() {
                self.sum -= old;
            }
        }
        if self.buf.len() == self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct EmaState {
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl EmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            value: f64::NAN,
            initialized: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.value = f64::NAN;
        self.initialized = false;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> f64 {
        if !self.initialized {
            self.initialized = true;
            self.value = value;
        } else {
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value;
        }
        self.value
    }
}

#[derive(Debug, Clone)]
struct RollingWmaState {
    period: usize,
    sum: f64,
    weighted_sum: f64,
    buf: VecDeque<f64>,
    norm: f64,
}

impl RollingWmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            sum: 0.0,
            weighted_sum: 0.0,
            buf: VecDeque::with_capacity(period),
            norm: (period * (period + 1) / 2) as f64,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.sum = 0.0;
        self.weighted_sum = 0.0;
        self.buf.clear();
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.buf.len() == self.period {
            let old_sum = self.sum;
            let oldest = self.buf.pop_front().unwrap_or(0.0);
            self.buf.push_back(value);
            self.sum = old_sum - oldest + value;
            self.weighted_sum = self.weighted_sum - old_sum + self.period as f64 * value;
        } else {
            self.buf.push_back(value);
            self.sum += value;
            self.weighted_sum += self.buf.len() as f64 * value;
        }
        if self.buf.len() == self.period {
            Some(self.weighted_sum / self.norm)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct RollingVwmaState {
    period: usize,
    pv_sum: f64,
    v_sum: f64,
    buf: VecDeque<(f64, f64)>,
}

impl RollingVwmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            pv_sum: 0.0,
            v_sum: 0.0,
            buf: VecDeque::with_capacity(period),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.pv_sum = 0.0;
        self.v_sum = 0.0;
        self.buf.clear();
    }

    #[inline(always)]
    fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.buf.push_back((price, volume));
        self.pv_sum += price * volume;
        self.v_sum += volume;
        if self.buf.len() > self.period {
            if let Some((old_price, old_volume)) = self.buf.pop_front() {
                self.pv_sum -= old_price * old_volume;
                self.v_sum -= old_volume;
            }
        }
        if self.buf.len() == self.period && self.v_sum != 0.0 {
            Some(self.pv_sum / self.v_sum)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
enum TrendMaState {
    Sma(RollingSmaState),
    Ema(EmaState),
    Wma(RollingWmaState),
    Vwma(RollingVwmaState),
}

impl TrendMaState {
    #[inline(always)]
    fn new(kind: TrendMaKind, period: usize) -> Self {
        match kind {
            TrendMaKind::Sma => Self::Sma(RollingSmaState::new(period)),
            TrendMaKind::Ema => Self::Ema(EmaState::new(period)),
            TrendMaKind::Wma => Self::Wma(RollingWmaState::new(period)),
            TrendMaKind::Vwma => Self::Vwma(RollingVwmaState::new(period)),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        match self {
            Self::Sma(state) => state.reset(),
            Self::Ema(state) => state.reset(),
            Self::Wma(state) => state.reset(),
            Self::Vwma(state) => state.reset(),
        }
    }

    #[inline(always)]
    fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        match self {
            Self::Sma(state) => state.update(close),
            Self::Ema(state) => Some(state.update(close)),
            Self::Wma(state) => state.update(close),
            Self::Vwma(state) => state.update(close, volume),
        }
    }
}

#[derive(Debug, Clone)]
struct ExtremumQueue {
    buf: VecDeque<(usize, f64)>,
    is_min: bool,
}

impl ExtremumQueue {
    #[inline(always)]
    fn new_min() -> Self {
        Self {
            buf: VecDeque::new(),
            is_min: true,
        }
    }

    #[inline(always)]
    fn new_max() -> Self {
        Self {
            buf: VecDeque::new(),
            is_min: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.buf.clear();
    }

    #[inline(always)]
    fn push(&mut self, index: usize, value: f64) {
        if self.is_min {
            while let Some((_, back)) = self.buf.back() {
                if *back > value {
                    self.buf.pop_back();
                } else {
                    break;
                }
            }
        } else {
            while let Some((_, back)) = self.buf.back() {
                if *back < value {
                    self.buf.pop_back();
                } else {
                    break;
                }
            }
        }
        self.buf.push_back((index, value));
    }

    #[inline(always)]
    fn prune(&mut self, min_index: usize) {
        while let Some((idx, _)) = self.buf.front() {
            if *idx < min_index {
                self.buf.pop_front();
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    fn current(&self) -> Option<f64> {
        self.buf.front().map(|(_, value)| *value)
    }
}

#[derive(Debug, Clone)]
struct ReversalCandidateState {
    bull_candidate: bool,
    bear_candidate: bool,
    bull_low: f64,
    bull_high: f64,
    bear_low: f64,
    bear_high: f64,
    bull_confirmed: bool,
    bear_confirmed: bool,
    bull_counter: usize,
    bear_counter: usize,
}

impl Default for ReversalCandidateState {
    fn default() -> Self {
        Self {
            bull_candidate: false,
            bear_candidate: false,
            bull_low: 0.0,
            bull_high: 0.0,
            bear_low: 0.0,
            bear_high: 0.0,
            bull_confirmed: false,
            bear_confirmed: false,
            bull_counter: 0,
            bear_counter: 0,
        }
    }
}

impl ReversalCandidateState {
    #[inline(always)]
    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[inline(always)]
fn reset_runtime(
    trend_ma_state: &mut TrendMaState,
    volume_sma: &mut RollingSmaState,
    prev_lows: &mut ExtremumQueue,
    prev_highs: &mut ExtremumQueue,
    reversal: &mut ReversalCandidateState,
    valid_run: &mut usize,
    stepped_ma: &mut f64,
    ma_last_update_bar: &mut usize,
    ma_direction: &mut i8,
) {
    trend_ma_state.reset();
    volume_sma.reset();
    prev_lows.reset();
    prev_highs.reset();
    reversal.reset();
    *valid_run = 0;
    *stepped_ma = f64::NAN;
    *ma_last_update_bar = 0;
    *ma_direction = 1;
}

#[inline(always)]
fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    lookback_period: usize,
    confirmation_period: usize,
    use_volume_confirmation: bool,
    trend_ma_period: usize,
    trend_ma_kind: TrendMaKind,
    ma_step_period: usize,
    out_buy_signal: &mut [f64],
    out_sell_signal: &mut [f64],
    out_stepped_ma: &mut [f64],
    out_state: &mut [f64],
) {
    let mut trend_ma_state = TrendMaState::new(trend_ma_kind, trend_ma_period);
    let mut volume_sma = RollingSmaState::new(VOLUME_SMA_PERIOD);
    let mut prev_lows = ExtremumQueue::new_min();
    let mut prev_highs = ExtremumQueue::new_max();
    let mut reversal = ReversalCandidateState::default();
    let mut valid_run = 0usize;
    let mut stepped_ma = f64::NAN;
    let mut ma_last_update_bar = 0usize;
    let mut ma_direction = 1i8;
    let prev_span = lookback_period.saturating_sub(1);

    for i in 0..close.len() {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let v = volume[i];

        if !is_valid_ohlcv(o, h, l, c, v) {
            out_buy_signal[i] = f64::NAN;
            out_sell_signal[i] = f64::NAN;
            out_stepped_ma[i] = f64::NAN;
            out_state[i] = f64::NAN;
            reset_runtime(
                &mut trend_ma_state,
                &mut volume_sma,
                &mut prev_lows,
                &mut prev_highs,
                &mut reversal,
                &mut valid_run,
                &mut stepped_ma,
                &mut ma_last_update_bar,
                &mut ma_direction,
            );
            continue;
        }

        valid_run += 1;
        out_buy_signal[i] = 0.0;
        out_sell_signal[i] = 0.0;

        let ma_current = trend_ma_state.update(c, v);
        let volume_avg = volume_sma.update(v);
        let volume_is_high = volume_avg.is_some_and(|avg| v > avg);

        let has_prev_window = prev_span == 0 || valid_run > prev_span;
        let bull_candidate_trigger = if prev_span == 0 {
            true
        } else {
            has_prev_window && prev_lows.current().is_some_and(|min_prev| c < min_prev)
        };
        let bear_candidate_trigger = if prev_span == 0 {
            true
        } else {
            has_prev_window && prev_highs.current().is_some_and(|max_prev| c > max_prev)
        };

        if bear_candidate_trigger {
            reversal.bear_candidate = true;
            reversal.bear_low = l;
            reversal.bear_high = h;
            reversal.bear_confirmed = false;
            reversal.bear_counter = 0;
        }

        if reversal.bear_candidate {
            reversal.bear_counter = reversal.bear_counter.saturating_add(1);
            if c > reversal.bear_high {
                reversal.bear_candidate = false;
            }
        }

        let mut bear_condition = false;
        if reversal.bear_candidate
            && c < reversal.bear_low
            && !reversal.bear_confirmed
            && reversal.bear_counter <= confirmation_period.saturating_add(1)
        {
            reversal.bear_confirmed = true;
            bear_condition = true;
        }

        if bear_condition && (!use_volume_confirmation || volume_is_high) {
            out_sell_signal[i] = 1.0;
        }

        if bull_candidate_trigger {
            reversal.bull_candidate = true;
            reversal.bull_low = l;
            reversal.bull_high = h;
            reversal.bull_confirmed = false;
            reversal.bull_counter = 0;
        }

        if reversal.bull_candidate {
            reversal.bull_counter = reversal.bull_counter.saturating_add(1);
            if c < reversal.bull_low {
                reversal.bull_candidate = false;
            }
        }

        let mut bull_condition = false;
        if reversal.bull_candidate
            && c > reversal.bull_high
            && !reversal.bull_confirmed
            && reversal.bull_counter <= confirmation_period.saturating_add(1)
        {
            reversal.bull_confirmed = true;
            bull_condition = true;
        }

        if bull_condition && (!use_volume_confirmation || volume_is_high) {
            out_buy_signal[i] = 1.0;
        }

        if let Some(ma_current) = ma_current {
            if stepped_ma.is_nan() {
                stepped_ma = ma_current;
                ma_last_update_bar = i;
            } else if ma_direction == 1 {
                if c < stepped_ma {
                    ma_direction = -1;
                    stepped_ma = ma_current;
                    ma_last_update_bar = i;
                } else if i.saturating_sub(ma_last_update_bar) >= ma_step_period {
                    stepped_ma = stepped_ma.max(ma_current);
                    ma_last_update_bar = i;
                }
            } else if c > stepped_ma {
                ma_direction = 1;
                stepped_ma = ma_current;
                ma_last_update_bar = i;
            } else if i.saturating_sub(ma_last_update_bar) >= ma_step_period {
                stepped_ma = stepped_ma.min(ma_current);
                ma_last_update_bar = i;
            }

            out_stepped_ma[i] = stepped_ma;
            out_state[i] = ma_direction as f64;
        } else {
            out_stepped_ma[i] = f64::NAN;
            out_state[i] = f64::NAN;
        }

        prev_lows.push(i, l);
        prev_highs.push(i, h);
        let min_index = i.saturating_add(1).saturating_sub(prev_span);
        prev_lows.prune(min_index);
        prev_highs.prune(min_index);
    }
}

#[inline(always)]
fn is_default_single_params(
    lookback_period: usize,
    confirmation_period: usize,
    use_volume_confirmation: bool,
    trend_ma_period: usize,
    trend_ma_kind: TrendMaKind,
    ma_step_period: usize,
) -> bool {
    lookback_period == DEFAULT_LOOKBACK_PERIOD
        && confirmation_period == DEFAULT_CONFIRMATION_PERIOD
        && use_volume_confirmation == DEFAULT_USE_VOLUME_CONFIRMATION
        && trend_ma_period == DEFAULT_TREND_MA_PERIOD
        && trend_ma_kind == TrendMaKind::Ema
        && ma_step_period == DEFAULT_MA_STEP_PERIOD
}

#[inline(always)]
fn compute_default_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out_buy_signal: &mut [f64],
    out_sell_signal: &mut [f64],
    out_stepped_ma: &mut [f64],
    out_state: &mut [f64],
) {
    const PREV_SPAN: usize = DEFAULT_LOOKBACK_PERIOD - 1;
    const CONFIRM_LIMIT: usize = DEFAULT_CONFIRMATION_PERIOD + 1;
    const VOLUME_DENOM: f64 = VOLUME_SMA_PERIOD as f64;
    const EMA_ALPHA: f64 = 2.0 / (DEFAULT_TREND_MA_PERIOD as f64 + 1.0);

    let mut volume_buf = [0.0; VOLUME_SMA_PERIOD];
    let mut volume_pos = 0usize;
    let mut volume_count = 0usize;
    let mut volume_sum = 0.0;

    let mut lows = [0.0; PREV_SPAN];
    let mut highs = [0.0; PREV_SPAN];
    let mut prev_pos = 0usize;
    let mut prev_count = 0usize;

    let mut reversal = ReversalCandidateState::default();
    let mut valid_run = 0usize;
    let mut ema_value = f64::NAN;
    let mut ema_initialized = false;
    let mut stepped_ma = f64::NAN;
    let mut ma_last_update_bar = 0usize;
    let mut ma_direction = 1i8;

    for i in 0..close.len() {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let v = volume[i];

        if !is_valid_ohlcv(o, h, l, c, v) {
            out_buy_signal[i] = f64::NAN;
            out_sell_signal[i] = f64::NAN;
            out_stepped_ma[i] = f64::NAN;
            out_state[i] = f64::NAN;
            volume_pos = 0;
            volume_count = 0;
            volume_sum = 0.0;
            prev_pos = 0;
            prev_count = 0;
            reversal.reset();
            valid_run = 0;
            ema_value = f64::NAN;
            ema_initialized = false;
            stepped_ma = f64::NAN;
            ma_last_update_bar = 0;
            ma_direction = 1;
            continue;
        }

        valid_run += 1;
        out_buy_signal[i] = 0.0;
        out_sell_signal[i] = 0.0;

        if !ema_initialized {
            ema_initialized = true;
            ema_value = c;
        } else {
            ema_value = EMA_ALPHA * c + (1.0 - EMA_ALPHA) * ema_value;
        }

        let volume_avg = if volume_count < VOLUME_SMA_PERIOD {
            volume_buf[volume_count] = v;
            volume_count += 1;
            volume_sum += v;
            if volume_count == VOLUME_SMA_PERIOD {
                Some(volume_sum / VOLUME_DENOM)
            } else {
                None
            }
        } else {
            let old = volume_buf[volume_pos];
            volume_buf[volume_pos] = v;
            volume_pos += 1;
            if volume_pos == VOLUME_SMA_PERIOD {
                volume_pos = 0;
            }
            volume_sum += v - old;
            Some(volume_sum / VOLUME_DENOM)
        };
        let volume_is_high = volume_avg.is_some_and(|avg| v > avg);

        let mut min_prev = f64::INFINITY;
        let mut max_prev = f64::NEG_INFINITY;
        if valid_run > PREV_SPAN {
            for j in 0..PREV_SPAN {
                let prev_low = lows[j];
                let prev_high = highs[j];
                if prev_low < min_prev {
                    min_prev = prev_low;
                }
                if prev_high > max_prev {
                    max_prev = prev_high;
                }
            }
        }
        let bull_candidate_trigger = valid_run > PREV_SPAN && c < min_prev;
        let bear_candidate_trigger = valid_run > PREV_SPAN && c > max_prev;

        if bear_candidate_trigger {
            reversal.bear_candidate = true;
            reversal.bear_low = l;
            reversal.bear_high = h;
            reversal.bear_confirmed = false;
            reversal.bear_counter = 0;
        }

        if reversal.bear_candidate {
            reversal.bear_counter = reversal.bear_counter.saturating_add(1);
            if c > reversal.bear_high {
                reversal.bear_candidate = false;
            }
        }

        let mut bear_condition = false;
        if reversal.bear_candidate
            && c < reversal.bear_low
            && !reversal.bear_confirmed
            && reversal.bear_counter <= CONFIRM_LIMIT
        {
            reversal.bear_confirmed = true;
            bear_condition = true;
        }

        if bear_condition && volume_is_high {
            out_sell_signal[i] = 1.0;
        }

        if bull_candidate_trigger {
            reversal.bull_candidate = true;
            reversal.bull_low = l;
            reversal.bull_high = h;
            reversal.bull_confirmed = false;
            reversal.bull_counter = 0;
        }

        if reversal.bull_candidate {
            reversal.bull_counter = reversal.bull_counter.saturating_add(1);
            if c < reversal.bull_low {
                reversal.bull_candidate = false;
            }
        }

        let mut bull_condition = false;
        if reversal.bull_candidate
            && c > reversal.bull_high
            && !reversal.bull_confirmed
            && reversal.bull_counter <= CONFIRM_LIMIT
        {
            reversal.bull_confirmed = true;
            bull_condition = true;
        }

        if bull_condition && volume_is_high {
            out_buy_signal[i] = 1.0;
        }

        if stepped_ma.is_nan() {
            stepped_ma = ema_value;
            ma_last_update_bar = i;
        } else if ma_direction == 1 {
            if c < stepped_ma {
                ma_direction = -1;
                stepped_ma = ema_value;
                ma_last_update_bar = i;
            } else if i.saturating_sub(ma_last_update_bar) >= DEFAULT_MA_STEP_PERIOD {
                stepped_ma = stepped_ma.max(ema_value);
                ma_last_update_bar = i;
            }
        } else if c > stepped_ma {
            ma_direction = 1;
            stepped_ma = ema_value;
            ma_last_update_bar = i;
        } else if i.saturating_sub(ma_last_update_bar) >= DEFAULT_MA_STEP_PERIOD {
            stepped_ma = stepped_ma.min(ema_value);
            ma_last_update_bar = i;
        }

        out_stepped_ma[i] = stepped_ma;
        out_state[i] = ma_direction as f64;

        if prev_count < PREV_SPAN {
            lows[prev_count] = l;
            highs[prev_count] = h;
            prev_count += 1;
        } else {
            lows[prev_pos] = l;
            highs[prev_pos] = h;
            prev_pos += 1;
            if prev_pos == PREV_SPAN {
                prev_pos = 0;
            }
        }
    }
}

#[inline]
pub fn reversal_signals(
    input: &ReversalSignalsInput,
) -> Result<ReversalSignalsOutput, ReversalSignalsError> {
    reversal_signals_with_kernel(input, Kernel::Auto)
}

pub fn reversal_signals_with_kernel(
    input: &ReversalSignalsInput,
    kernel: Kernel,
) -> Result<ReversalSignalsOutput, ReversalSignalsError> {
    let (open, high, low, close, volume) = input_slices(input);
    let lookback_period = input.get_lookback_period();
    let confirmation_period = input.get_confirmation_period();
    let use_volume_confirmation = input.get_use_volume_confirmation();
    let trend_ma_period = input.get_trend_ma_period();
    let trend_ma_type = input.get_trend_ma_type();
    let ma_step_period = input.get_ma_step_period();
    let trend_ma_kind = validate_common(
        open,
        high,
        low,
        close,
        volume,
        lookback_period,
        trend_ma_period,
        trend_ma_type,
        use_volume_confirmation,
    )?;

    let _chosen = normalize_single_kernel(kernel)?;
    let mut buy_signal = alloc_with_nan_prefix(close.len(), 0);
    let mut sell_signal = alloc_with_nan_prefix(close.len(), 0);
    let mut stepped_ma = alloc_with_nan_prefix(close.len(), 0);
    let mut state = alloc_with_nan_prefix(close.len(), 0);

    if is_default_single_params(
        lookback_period,
        confirmation_period,
        use_volume_confirmation,
        trend_ma_period,
        trend_ma_kind,
        ma_step_period,
    ) {
        compute_default_row(
            open,
            high,
            low,
            close,
            volume,
            &mut buy_signal,
            &mut sell_signal,
            &mut stepped_ma,
            &mut state,
        );
    } else {
        compute_row(
            open,
            high,
            low,
            close,
            volume,
            lookback_period,
            confirmation_period,
            use_volume_confirmation,
            trend_ma_period,
            trend_ma_kind,
            ma_step_period,
            &mut buy_signal,
            &mut sell_signal,
            &mut stepped_ma,
            &mut state,
        );
    }

    Ok(ReversalSignalsOutput {
        buy_signal,
        sell_signal,
        stepped_ma,
        state,
    })
}

pub fn reversal_signals_into_slice(
    out_buy_signal: &mut [f64],
    out_sell_signal: &mut [f64],
    out_stepped_ma: &mut [f64],
    out_state: &mut [f64],
    input: &ReversalSignalsInput,
    kernel: Kernel,
) -> Result<(), ReversalSignalsError> {
    let (open, high, low, close, volume) = input_slices(input);
    let lookback_period = input.get_lookback_period();
    let confirmation_period = input.get_confirmation_period();
    let use_volume_confirmation = input.get_use_volume_confirmation();
    let trend_ma_period = input.get_trend_ma_period();
    let trend_ma_type = input.get_trend_ma_type();
    let ma_step_period = input.get_ma_step_period();
    let trend_ma_kind = validate_common(
        open,
        high,
        low,
        close,
        volume,
        lookback_period,
        trend_ma_period,
        trend_ma_type,
        use_volume_confirmation,
    )?;

    for out in [
        &mut *out_buy_signal,
        &mut *out_sell_signal,
        &mut *out_stepped_ma,
        &mut *out_state,
    ] {
        if out.len() != close.len() {
            return Err(ReversalSignalsError::OutputLengthMismatch {
                expected: close.len(),
                got: out.len(),
            });
        }
    }

    let _chosen = normalize_single_kernel(kernel)?;
    if is_default_single_params(
        lookback_period,
        confirmation_period,
        use_volume_confirmation,
        trend_ma_period,
        trend_ma_kind,
        ma_step_period,
    ) {
        compute_default_row(
            open,
            high,
            low,
            close,
            volume,
            out_buy_signal,
            out_sell_signal,
            out_stepped_ma,
            out_state,
        );
    } else {
        compute_row(
            open,
            high,
            low,
            close,
            volume,
            lookback_period,
            confirmation_period,
            use_volume_confirmation,
            trend_ma_period,
            trend_ma_kind,
            ma_step_period,
            out_buy_signal,
            out_sell_signal,
            out_stepped_ma,
            out_state,
        );
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn reversal_signals_into(
    input: &ReversalSignalsInput,
    out_buy_signal: &mut [f64],
    out_sell_signal: &mut [f64],
    out_stepped_ma: &mut [f64],
    out_state: &mut [f64],
) -> Result<(), ReversalSignalsError> {
    reversal_signals_into_slice(
        out_buy_signal,
        out_sell_signal,
        out_stepped_ma,
        out_state,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone)]
pub struct ReversalSignalsBatchRange {
    pub lookback_period: (usize, usize, usize),
    pub confirmation_period: (usize, usize, usize),
    pub trend_ma_period: (usize, usize, usize),
    pub ma_step_period: (usize, usize, usize),
    pub use_volume_confirmation: bool,
    pub trend_ma_type: String,
}

impl Default for ReversalSignalsBatchRange {
    fn default() -> Self {
        Self {
            lookback_period: (DEFAULT_LOOKBACK_PERIOD, DEFAULT_LOOKBACK_PERIOD, 0),
            confirmation_period: (DEFAULT_CONFIRMATION_PERIOD, DEFAULT_CONFIRMATION_PERIOD, 0),
            trend_ma_period: (DEFAULT_TREND_MA_PERIOD, DEFAULT_TREND_MA_PERIOD, 0),
            ma_step_period: (DEFAULT_MA_STEP_PERIOD, DEFAULT_MA_STEP_PERIOD, 0),
            use_volume_confirmation: DEFAULT_USE_VOLUME_CONFIRMATION,
            trend_ma_type: DEFAULT_TREND_MA_TYPE.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReversalSignalsBatchOutput {
    pub buy_signal: Vec<f64>,
    pub sell_signal: Vec<f64>,
    pub stepped_ma: Vec<f64>,
    pub state: Vec<f64>,
    pub combos: Vec<ReversalSignalsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone)]
pub struct ReversalSignalsBatchBuilder {
    range: ReversalSignalsBatchRange,
    kernel: Kernel,
}

impl Default for ReversalSignalsBatchBuilder {
    fn default() -> Self {
        Self {
            range: ReversalSignalsBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl ReversalSignalsBatchBuilder {
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
    pub fn lookback_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn confirmation_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.confirmation_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn trend_ma_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.trend_ma_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn ma_step_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ma_step_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn use_volume_confirmation(mut self, value: bool) -> Self {
        self.range.use_volume_confirmation = value;
        self
    }

    #[inline(always)]
    pub fn trend_ma_type(mut self, value: &str) -> Result<Self, ReversalSignalsError> {
        let kind = parse_trend_ma_kind(value)?;
        self.range.trend_ma_type = trend_ma_kind_name(kind).to_string();
        Ok(self)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<ReversalSignalsBatchOutput, ReversalSignalsError> {
        reversal_signals_batch_with_kernel(open, high, low, close, volume, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<ReversalSignalsBatchOutput, ReversalSignalsError> {
        reversal_signals_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
            &self.range,
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
) -> Result<Vec<usize>, ReversalSignalsError> {
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(ReversalSignalsError::InvalidRange {
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
            return Err(ReversalSignalsError::InvalidRange {
                start: field.to_string(),
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
fn expand_grid_checked(
    range: &ReversalSignalsBatchRange,
) -> Result<Vec<ReversalSignalsParams>, ReversalSignalsError> {
    let _ = parse_trend_ma_kind(range.trend_ma_type.as_str())?;
    let lookback_periods = expand_usize_range(
        "lookback_period",
        range.lookback_period.0,
        range.lookback_period.1,
        range.lookback_period.2,
    )?;
    let confirmation_periods = expand_usize_range(
        "confirmation_period",
        range.confirmation_period.0,
        range.confirmation_period.1,
        range.confirmation_period.2,
    )?;
    let trend_ma_periods = expand_usize_range(
        "trend_ma_period",
        range.trend_ma_period.0,
        range.trend_ma_period.1,
        range.trend_ma_period.2,
    )?;
    let ma_step_periods = expand_usize_range(
        "ma_step_period",
        range.ma_step_period.0,
        range.ma_step_period.1,
        range.ma_step_period.2,
    )?;

    let mut combos = Vec::new();
    for &lookback_period in &lookback_periods {
        for &confirmation_period in &confirmation_periods {
            for &trend_ma_period in &trend_ma_periods {
                for &ma_step_period in &ma_step_periods {
                    validate_params_only(
                        lookback_period,
                        trend_ma_period,
                        range.trend_ma_type.as_str(),
                    )?;
                    combos.push(ReversalSignalsParams {
                        lookback_period: Some(lookback_period),
                        confirmation_period: Some(confirmation_period),
                        use_volume_confirmation: Some(range.use_volume_confirmation),
                        trend_ma_period: Some(trend_ma_period),
                        trend_ma_type: Some(range.trend_ma_type.clone()),
                        ma_step_period: Some(ma_step_period),
                    });
                }
            }
        }
    }
    Ok(combos)
}

pub fn expand_grid_reversal_signals(
    range: &ReversalSignalsBatchRange,
) -> Vec<ReversalSignalsParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn reversal_signals_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &ReversalSignalsBatchRange,
    kernel: Kernel,
) -> Result<ReversalSignalsBatchOutput, ReversalSignalsError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| ReversalSignalsError::InvalidInput {
            msg: "reversal_signals: rows*cols overflow in batch".to_string(),
        })?;

    let buy_mu = make_uninit_matrix(rows, cols);
    let sell_mu = make_uninit_matrix(rows, cols);
    let stepped_mu = make_uninit_matrix(rows, cols);
    let state_mu = make_uninit_matrix(rows, cols);

    let mut buy_guard = core::mem::ManuallyDrop::new(buy_mu);
    let mut sell_guard = core::mem::ManuallyDrop::new(sell_mu);
    let mut stepped_guard = core::mem::ManuallyDrop::new(stepped_mu);
    let mut state_guard = core::mem::ManuallyDrop::new(state_mu);

    let buy_signal =
        unsafe { std::slice::from_raw_parts_mut(buy_guard.as_mut_ptr() as *mut f64, total) };
    let sell_signal =
        unsafe { std::slice::from_raw_parts_mut(sell_guard.as_mut_ptr() as *mut f64, total) };
    let stepped_ma =
        unsafe { std::slice::from_raw_parts_mut(stepped_guard.as_mut_ptr() as *mut f64, total) };
    let state =
        unsafe { std::slice::from_raw_parts_mut(state_guard.as_mut_ptr() as *mut f64, total) };

    reversal_signals_batch_inner_into(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        kernel,
        true,
        buy_signal,
        sell_signal,
        stepped_ma,
        state,
    )?;

    let buy_signal = unsafe {
        Vec::from_raw_parts(
            buy_guard.as_mut_ptr() as *mut f64,
            buy_guard.len(),
            buy_guard.capacity(),
        )
    };
    let sell_signal = unsafe {
        Vec::from_raw_parts(
            sell_guard.as_mut_ptr() as *mut f64,
            sell_guard.len(),
            sell_guard.capacity(),
        )
    };
    let stepped_ma = unsafe {
        Vec::from_raw_parts(
            stepped_guard.as_mut_ptr() as *mut f64,
            stepped_guard.len(),
            stepped_guard.capacity(),
        )
    };
    let state = unsafe {
        Vec::from_raw_parts(
            state_guard.as_mut_ptr() as *mut f64,
            state_guard.len(),
            state_guard.capacity(),
        )
    };

    Ok(ReversalSignalsBatchOutput {
        buy_signal,
        sell_signal,
        stepped_ma,
        state,
        combos,
        rows,
        cols,
    })
}

pub fn reversal_signals_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &ReversalSignalsBatchRange,
    kernel: Kernel,
) -> Result<ReversalSignalsBatchOutput, ReversalSignalsError> {
    reversal_signals_batch_with_kernel(open, high, low, close, volume, sweep, kernel)
}

pub fn reversal_signals_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &ReversalSignalsBatchRange,
    kernel: Kernel,
) -> Result<ReversalSignalsBatchOutput, ReversalSignalsError> {
    reversal_signals_batch_with_kernel(open, high, low, close, volume, sweep, kernel)
}

fn reversal_signals_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &ReversalSignalsBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_buy_signal: &mut [f64],
    out_sell_signal: &mut [f64],
    out_stepped_ma: &mut [f64],
    out_state: &mut [f64],
) -> Result<Vec<ReversalSignalsParams>, ReversalSignalsError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| ReversalSignalsError::InvalidInput {
            msg: "reversal_signals: rows*cols overflow in batch_into".to_string(),
        })?;

    for out in [
        &mut *out_buy_signal,
        &mut *out_sell_signal,
        &mut *out_stepped_ma,
        &mut *out_state,
    ] {
        if out.len() != total {
            return Err(ReversalSignalsError::MismatchedOutputLen {
                dst_len: out.len(),
                expected_len: total,
            });
        }
    }

    let max_lookback = combos
        .iter()
        .map(|params| params.lookback_period.unwrap_or(DEFAULT_LOOKBACK_PERIOD))
        .max()
        .unwrap_or(DEFAULT_LOOKBACK_PERIOD);
    let max_trend_ma_period = combos
        .iter()
        .map(|params| params.trend_ma_period.unwrap_or(DEFAULT_TREND_MA_PERIOD))
        .max()
        .unwrap_or(DEFAULT_TREND_MA_PERIOD);
    let trend_ma_type = combos
        .first()
        .and_then(|params| params.trend_ma_type.as_deref())
        .unwrap_or(DEFAULT_TREND_MA_TYPE);
    let use_volume_confirmation = combos
        .first()
        .and_then(|params| params.use_volume_confirmation)
        .unwrap_or(DEFAULT_USE_VOLUME_CONFIRMATION);

    let _ = validate_common(
        open,
        high,
        low,
        close,
        volume,
        max_lookback,
        max_trend_ma_period,
        trend_ma_type,
        use_volume_confirmation,
    )?;
    let _chosen = normalize_batch_kernel(kernel)?;

    let worker = |row: usize,
                  buy_row: &mut [f64],
                  sell_row: &mut [f64],
                  stepped_row: &mut [f64],
                  state_row: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            open,
            high,
            low,
            close,
            volume,
            params.lookback_period.unwrap_or(DEFAULT_LOOKBACK_PERIOD),
            params
                .confirmation_period
                .unwrap_or(DEFAULT_CONFIRMATION_PERIOD),
            params
                .use_volume_confirmation
                .unwrap_or(DEFAULT_USE_VOLUME_CONFIRMATION),
            params.trend_ma_period.unwrap_or(DEFAULT_TREND_MA_PERIOD),
            parse_trend_ma_kind(
                params
                    .trend_ma_type
                    .as_deref()
                    .unwrap_or(DEFAULT_TREND_MA_TYPE),
            )
            .unwrap_or(TrendMaKind::Ema),
            params.ma_step_period.unwrap_or(DEFAULT_MA_STEP_PERIOD),
            buy_row,
            sell_row,
            stepped_row,
            state_row,
        );
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel && rows > 1 {
        out_buy_signal
            .par_chunks_mut(cols)
            .zip(out_sell_signal.par_chunks_mut(cols))
            .zip(out_stepped_ma.par_chunks_mut(cols))
            .zip(out_state.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (((buy_row, sell_row), stepped_row), state_row))| {
                worker(row, buy_row, sell_row, stepped_row, state_row);
            });
    } else {
        for (row, (((buy_row, sell_row), stepped_row), state_row)) in out_buy_signal
            .chunks_mut(cols)
            .zip(out_sell_signal.chunks_mut(cols))
            .zip(out_stepped_ma.chunks_mut(cols))
            .zip(out_state.chunks_mut(cols))
            .enumerate()
        {
            worker(row, buy_row, sell_row, stepped_row, state_row);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, (((buy_row, sell_row), stepped_row), state_row)) in out_buy_signal
            .chunks_mut(cols)
            .zip(out_sell_signal.chunks_mut(cols))
            .zip(out_stepped_ma.chunks_mut(cols))
            .zip(out_state.chunks_mut(cols))
            .enumerate()
        {
            worker(row, buy_row, sell_row, stepped_row, state_row);
        }
    }

    Ok(combos)
}

#[derive(Debug, Clone)]
pub struct ReversalSignalsStream {
    lookback_period: usize,
    confirmation_period: usize,
    use_volume_confirmation: bool,
    ma_step_period: usize,
    index: usize,
    valid_run: usize,
    trend_ma_state: TrendMaState,
    volume_sma: RollingSmaState,
    prev_lows: ExtremumQueue,
    prev_highs: ExtremumQueue,
    reversal: ReversalCandidateState,
    stepped_ma: f64,
    ma_last_update_bar: usize,
    ma_direction: i8,
}

impl ReversalSignalsStream {
    pub fn try_new(params: ReversalSignalsParams) -> Result<Self, ReversalSignalsError> {
        let lookback_period = params.lookback_period.unwrap_or(DEFAULT_LOOKBACK_PERIOD);
        let confirmation_period = params
            .confirmation_period
            .unwrap_or(DEFAULT_CONFIRMATION_PERIOD);
        let use_volume_confirmation = params
            .use_volume_confirmation
            .unwrap_or(DEFAULT_USE_VOLUME_CONFIRMATION);
        let trend_ma_period = params.trend_ma_period.unwrap_or(DEFAULT_TREND_MA_PERIOD);
        let trend_ma_type = params
            .trend_ma_type
            .as_deref()
            .unwrap_or(DEFAULT_TREND_MA_TYPE);
        let ma_step_period = params.ma_step_period.unwrap_or(DEFAULT_MA_STEP_PERIOD);
        let trend_ma_kind = validate_params_only(lookback_period, trend_ma_period, trend_ma_type)?;

        Ok(Self {
            lookback_period,
            confirmation_period,
            use_volume_confirmation,
            ma_step_period,
            index: 0,
            valid_run: 0,
            trend_ma_state: TrendMaState::new(trend_ma_kind, trend_ma_period),
            volume_sma: RollingSmaState::new(VOLUME_SMA_PERIOD),
            prev_lows: ExtremumQueue::new_min(),
            prev_highs: ExtremumQueue::new_max(),
            reversal: ReversalCandidateState::default(),
            stepped_ma: f64::NAN,
            ma_last_update_bar: 0,
            ma_direction: 1,
        })
    }

    #[inline(always)]
    fn reset(&mut self) {
        reset_runtime(
            &mut self.trend_ma_state,
            &mut self.volume_sma,
            &mut self.prev_lows,
            &mut self.prev_highs,
            &mut self.reversal,
            &mut self.valid_run,
            &mut self.stepped_ma,
            &mut self.ma_last_update_bar,
            &mut self.ma_direction,
        );
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        let i = self.index;
        self.index = self.index.saturating_add(1);

        if !is_valid_ohlcv(open, high, low, close, volume) {
            self.reset();
            return None;
        }

        self.valid_run += 1;
        let mut buy_signal = 0.0;
        let mut sell_signal = 0.0;
        let mut stepped_ma_out = f64::NAN;
        let mut state_out = f64::NAN;

        let ma_current = self.trend_ma_state.update(close, volume);
        let volume_avg = self.volume_sma.update(volume);
        let volume_is_high = volume_avg.is_some_and(|avg| volume > avg);
        let prev_span = self.lookback_period.saturating_sub(1);
        let has_prev_window = prev_span == 0 || self.valid_run > prev_span;

        let bull_candidate_trigger = if prev_span == 0 {
            true
        } else {
            has_prev_window
                && self
                    .prev_lows
                    .current()
                    .is_some_and(|min_prev| close < min_prev)
        };
        let bear_candidate_trigger = if prev_span == 0 {
            true
        } else {
            has_prev_window
                && self
                    .prev_highs
                    .current()
                    .is_some_and(|max_prev| close > max_prev)
        };

        if bear_candidate_trigger {
            self.reversal.bear_candidate = true;
            self.reversal.bear_low = low;
            self.reversal.bear_high = high;
            self.reversal.bear_confirmed = false;
            self.reversal.bear_counter = 0;
        }

        if self.reversal.bear_candidate {
            self.reversal.bear_counter = self.reversal.bear_counter.saturating_add(1);
            if close > self.reversal.bear_high {
                self.reversal.bear_candidate = false;
            }
        }

        if self.reversal.bear_candidate
            && close < self.reversal.bear_low
            && !self.reversal.bear_confirmed
            && self.reversal.bear_counter <= self.confirmation_period.saturating_add(1)
            && (!self.use_volume_confirmation || volume_is_high)
        {
            self.reversal.bear_confirmed = true;
            sell_signal = 1.0;
        }

        if bull_candidate_trigger {
            self.reversal.bull_candidate = true;
            self.reversal.bull_low = low;
            self.reversal.bull_high = high;
            self.reversal.bull_confirmed = false;
            self.reversal.bull_counter = 0;
        }

        if self.reversal.bull_candidate {
            self.reversal.bull_counter = self.reversal.bull_counter.saturating_add(1);
            if close < self.reversal.bull_low {
                self.reversal.bull_candidate = false;
            }
        }

        if self.reversal.bull_candidate
            && close > self.reversal.bull_high
            && !self.reversal.bull_confirmed
            && self.reversal.bull_counter <= self.confirmation_period.saturating_add(1)
            && (!self.use_volume_confirmation || volume_is_high)
        {
            self.reversal.bull_confirmed = true;
            buy_signal = 1.0;
        }

        if let Some(ma_current) = ma_current {
            if self.stepped_ma.is_nan() {
                self.stepped_ma = ma_current;
                self.ma_last_update_bar = i;
            } else if self.ma_direction == 1 {
                if close < self.stepped_ma {
                    self.ma_direction = -1;
                    self.stepped_ma = ma_current;
                    self.ma_last_update_bar = i;
                } else if i.saturating_sub(self.ma_last_update_bar) >= self.ma_step_period {
                    self.stepped_ma = self.stepped_ma.max(ma_current);
                    self.ma_last_update_bar = i;
                }
            } else if close > self.stepped_ma {
                self.ma_direction = 1;
                self.stepped_ma = ma_current;
                self.ma_last_update_bar = i;
            } else if i.saturating_sub(self.ma_last_update_bar) >= self.ma_step_period {
                self.stepped_ma = self.stepped_ma.min(ma_current);
                self.ma_last_update_bar = i;
            }

            stepped_ma_out = self.stepped_ma;
            state_out = self.ma_direction as f64;
        }

        self.prev_lows.push(i, low);
        self.prev_highs.push(i, high);
        let min_index = i.saturating_add(1).saturating_sub(prev_span);
        self.prev_lows.prune(min_index);
        self.prev_highs.prune(min_index);

        Some((buy_signal, sell_signal, stepped_ma_out, state_out))
    }
}

impl ReversalSignalsBuilder {
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<ReversalSignalsOutput, ReversalSignalsError> {
        reversal_signals_with_kernel(
            &ReversalSignalsInput::from_candles(
                candles,
                ReversalSignalsParams {
                    lookback_period: self.lookback_period,
                    confirmation_period: self.confirmation_period,
                    use_volume_confirmation: self.use_volume_confirmation,
                    trend_ma_period: self.trend_ma_period,
                    trend_ma_type: self
                        .trend_ma_type
                        .map(|kind| trend_ma_kind_name(kind).to_string()),
                    ma_step_period: self.ma_step_period,
                },
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<ReversalSignalsOutput, ReversalSignalsError> {
        reversal_signals_with_kernel(
            &ReversalSignalsInput::from_slices(
                open,
                high,
                low,
                close,
                volume,
                ReversalSignalsParams {
                    lookback_period: self.lookback_period,
                    confirmation_period: self.confirmation_period,
                    use_volume_confirmation: self.use_volume_confirmation,
                    trend_ma_period: self.trend_ma_period,
                    trend_ma_type: self
                        .trend_ma_type
                        .map(|kind| trend_ma_kind_name(kind).to_string()),
                    ma_step_period: self.ma_step_period,
                },
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<ReversalSignalsStream, ReversalSignalsError> {
        ReversalSignalsStream::try_new(ReversalSignalsParams {
            lookback_period: self.lookback_period,
            confirmation_period: self.confirmation_period,
            use_volume_confirmation: self.use_volume_confirmation,
            trend_ma_period: self.trend_ma_period,
            trend_ma_type: self
                .trend_ma_type
                .map(|kind| trend_ma_kind_name(kind).to_string()),
            ma_step_period: self.ma_step_period,
        })
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "reversal_signals",
    signature = (open, high, low, close, volume, lookback_period=DEFAULT_LOOKBACK_PERIOD, confirmation_period=DEFAULT_CONFIRMATION_PERIOD, use_volume_confirmation=DEFAULT_USE_VOLUME_CONFIRMATION, trend_ma_period=DEFAULT_TREND_MA_PERIOD, trend_ma_type=DEFAULT_TREND_MA_TYPE, ma_step_period=DEFAULT_MA_STEP_PERIOD, kernel=None)
)]
pub fn reversal_signals_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    lookback_period: usize,
    confirmation_period: usize,
    use_volume_confirmation: bool,
    trend_ma_period: usize,
    trend_ma_type: &str,
    ma_step_period: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| {
            reversal_signals_with_kernel(
                &ReversalSignalsInput::from_slices(
                    open,
                    high,
                    low,
                    close,
                    volume,
                    ReversalSignalsParams {
                        lookback_period: Some(lookback_period),
                        confirmation_period: Some(confirmation_period),
                        use_volume_confirmation: Some(use_volume_confirmation),
                        trend_ma_period: Some(trend_ma_period),
                        trend_ma_type: Some(trend_ma_type.to_string()),
                        ma_step_period: Some(ma_step_period),
                    },
                ),
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.buy_signal.into_pyarray(py),
        out.sell_signal.into_pyarray(py),
        out.stepped_ma.into_pyarray(py),
        out.state.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "reversal_signals_batch",
    signature = (open, high, low, close, volume, lookback_period_range=(DEFAULT_LOOKBACK_PERIOD, DEFAULT_LOOKBACK_PERIOD, 0), confirmation_period_range=(DEFAULT_CONFIRMATION_PERIOD, DEFAULT_CONFIRMATION_PERIOD, 0), trend_ma_period_range=(DEFAULT_TREND_MA_PERIOD, DEFAULT_TREND_MA_PERIOD, 0), ma_step_period_range=(DEFAULT_MA_STEP_PERIOD, DEFAULT_MA_STEP_PERIOD, 0), use_volume_confirmation=DEFAULT_USE_VOLUME_CONFIRMATION, trend_ma_type=DEFAULT_TREND_MA_TYPE, kernel=None)
)]
pub fn reversal_signals_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    lookback_period_range: (usize, usize, usize),
    confirmation_period_range: (usize, usize, usize),
    trend_ma_period_range: (usize, usize, usize),
    ma_step_period_range: (usize, usize, usize),
    use_volume_confirmation: bool,
    trend_ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let out = py
        .allow_threads(|| {
            reversal_signals_batch_with_kernel(
                open,
                high,
                low,
                close,
                volume,
                &ReversalSignalsBatchRange {
                    lookback_period: lookback_period_range,
                    confirmation_period: confirmation_period_range,
                    trend_ma_period: trend_ma_period_range,
                    ma_step_period: ma_step_period_range,
                    use_volume_confirmation,
                    trend_ma_type: trend_ma_type.to_string(),
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "buy_signal",
        out.buy_signal
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "sell_signal",
        out.sell_signal
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "stepped_ma",
        out.stepped_ma
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "state",
        out.state.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "lookback_periods",
        out.combos
            .iter()
            .map(|params| params.lookback_period.unwrap_or(DEFAULT_LOOKBACK_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "confirmation_periods",
        out.combos
            .iter()
            .map(|params| {
                params
                    .confirmation_period
                    .unwrap_or(DEFAULT_CONFIRMATION_PERIOD) as u64
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "trend_ma_periods",
        out.combos
            .iter()
            .map(|params| params.trend_ma_period.unwrap_or(DEFAULT_TREND_MA_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_step_periods",
        out.combos
            .iter()
            .map(|params| params.ma_step_period.unwrap_or(DEFAULT_MA_STEP_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "ReversalSignalsStream")]
pub struct ReversalSignalsStreamPy {
    stream: ReversalSignalsStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ReversalSignalsStreamPy {
    #[new]
    #[pyo3(signature = (lookback_period=DEFAULT_LOOKBACK_PERIOD, confirmation_period=DEFAULT_CONFIRMATION_PERIOD, use_volume_confirmation=DEFAULT_USE_VOLUME_CONFIRMATION, trend_ma_period=DEFAULT_TREND_MA_PERIOD, trend_ma_type=DEFAULT_TREND_MA_TYPE, ma_step_period=DEFAULT_MA_STEP_PERIOD))]
    fn new(
        lookback_period: usize,
        confirmation_period: usize,
        use_volume_confirmation: bool,
        trend_ma_period: usize,
        trend_ma_type: &str,
        ma_step_period: usize,
    ) -> PyResult<Self> {
        let stream = ReversalSignalsStream::try_new(ReversalSignalsParams {
            lookback_period: Some(lookback_period),
            confirmation_period: Some(confirmation_period),
            use_volume_confirmation: Some(use_volume_confirmation),
            trend_ma_period: Some(trend_ma_period),
            trend_ma_type: Some(trend_ma_type.to_string()),
            ma_step_period: Some(ma_step_period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(open, high, low, close, volume)
    }
}

#[cfg(feature = "python")]
pub fn register_reversal_signals_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reversal_signals_py, m)?)?;
    m.add_function(wrap_pyfunction!(reversal_signals_batch_py, m)?)?;
    m.add_class::<ReversalSignalsStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReversalSignalsBatchConfig {
    pub lookback_period_range: Vec<usize>,
    pub confirmation_period_range: Vec<usize>,
    pub trend_ma_period_range: Vec<usize>,
    pub ma_step_period_range: Vec<usize>,
    pub use_volume_confirmation: bool,
    pub trend_ma_type: String,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = reversal_signals_js)]
pub fn reversal_signals_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    lookback_period: usize,
    confirmation_period: usize,
    use_volume_confirmation: bool,
    trend_ma_period: usize,
    trend_ma_type: String,
    ma_step_period: usize,
) -> Result<JsValue, JsValue> {
    let out = reversal_signals_with_kernel(
        &ReversalSignalsInput::from_slices(
            open,
            high,
            low,
            close,
            volume,
            ReversalSignalsParams {
                lookback_period: Some(lookback_period),
                confirmation_period: Some(confirmation_period),
                use_volume_confirmation: Some(use_volume_confirmation),
                trend_ma_period: Some(trend_ma_period),
                trend_ma_type: Some(trend_ma_type),
                ma_step_period: Some(ma_step_period),
            },
        ),
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("buy_signal"),
        &serde_wasm_bindgen::to_value(&out.buy_signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("sell_signal"),
        &serde_wasm_bindgen::to_value(&out.sell_signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("stepped_ma"),
        &serde_wasm_bindgen::to_value(&out.stepped_ma).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("state"),
        &serde_wasm_bindgen::to_value(&out.state).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = reversal_signals_batch_js)]
pub fn reversal_signals_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: ReversalSignalsBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if cfg.lookback_period_range.len() != 3
        || cfg.confirmation_period_range.len() != 3
        || cfg.trend_ma_period_range.len() != 3
        || cfg.ma_step_period_range.len() != 3
    {
        return Err(JsValue::from_str(
            "reversal_signals_batch_js: range vectors must have length 3",
        ));
    }

    let out = reversal_signals_batch_with_kernel(
        open,
        high,
        low,
        close,
        volume,
        &ReversalSignalsBatchRange {
            lookback_period: (
                cfg.lookback_period_range[0],
                cfg.lookback_period_range[1],
                cfg.lookback_period_range[2],
            ),
            confirmation_period: (
                cfg.confirmation_period_range[0],
                cfg.confirmation_period_range[1],
                cfg.confirmation_period_range[2],
            ),
            trend_ma_period: (
                cfg.trend_ma_period_range[0],
                cfg.trend_ma_period_range[1],
                cfg.trend_ma_period_range[2],
            ),
            ma_step_period: (
                cfg.ma_step_period_range[0],
                cfg.ma_step_period_range[1],
                cfg.ma_step_period_range[2],
            ),
            use_volume_confirmation: cfg.use_volume_confirmation,
            trend_ma_type: cfg.trend_ma_type,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("buy_signal"),
        &serde_wasm_bindgen::to_value(&out.buy_signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("sell_signal"),
        &serde_wasm_bindgen::to_value(&out.sell_signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("stepped_ma"),
        &serde_wasm_bindgen::to_value(&out.stepped_ma).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("state"),
        &serde_wasm_bindgen::to_value(&out.state).unwrap(),
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
pub fn reversal_signals_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(OUTPUTS * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reversal_signals_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, OUTPUTS * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reversal_signals_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback_period: usize,
    confirmation_period: usize,
    use_volume_confirmation: bool,
    trend_ma_period: usize,
    trend_ma_type: String,
    ma_step_period: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to reversal_signals_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, OUTPUTS * len);
        let (buy_signal, tail) = out.split_at_mut(len);
        let (sell_signal, tail) = tail.split_at_mut(len);
        let (stepped_ma, state) = tail.split_at_mut(len);
        reversal_signals_into_slice(
            buy_signal,
            sell_signal,
            stepped_ma,
            state,
            &ReversalSignalsInput::from_slices(
                open,
                high,
                low,
                close,
                volume,
                ReversalSignalsParams {
                    lookback_period: Some(lookback_period),
                    confirmation_period: Some(confirmation_period),
                    use_volume_confirmation: Some(use_volume_confirmation),
                    trend_ma_period: Some(trend_ma_period),
                    trend_ma_type: Some(trend_ma_type),
                    ma_step_period: Some(ma_step_period),
                },
            ),
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reversal_signals_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
    confirmation_start: usize,
    confirmation_end: usize,
    confirmation_step: usize,
    trend_ma_period_start: usize,
    trend_ma_period_end: usize,
    trend_ma_period_step: usize,
    ma_step_start: usize,
    ma_step_end: usize,
    ma_step_step: usize,
    use_volume_confirmation: bool,
    trend_ma_type: String,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to reversal_signals_batch_into",
        ));
    }

    let sweep = ReversalSignalsBatchRange {
        lookback_period: (lookback_start, lookback_end, lookback_step),
        confirmation_period: (confirmation_start, confirmation_end, confirmation_step),
        trend_ma_period: (
            trend_ma_period_start,
            trend_ma_period_end,
            trend_ma_period_step,
        ),
        ma_step_period: (ma_step_start, ma_step_end, ma_step_step),
        use_volume_confirmation,
        trend_ma_type,
    };

    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let split = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in reversal_signals_batch_into"))?;
    let total = split.checked_mul(OUTPUTS).ok_or_else(|| {
        JsValue::from_str("outputs*rows*cols overflow in reversal_signals_batch_into")
    })?;

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (buy_signal, tail) = out.split_at_mut(split);
        let (sell_signal, tail) = tail.split_at_mut(split);
        let (stepped_ma, state) = tail.split_at_mut(split);
        reversal_signals_batch_inner_into(
            open,
            high,
            low,
            close,
            volume,
            &sweep,
            Kernel::Auto,
            false,
            buy_signal,
            sell_signal,
            stepped_ma,
            state,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reversal_signals_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    lookback_period: usize,
    confirmation_period: usize,
    use_volume_confirmation: bool,
    trend_ma_period: usize,
    trend_ma_type: String,
    ma_step_period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = reversal_signals_js(
        open,
        high,
        low,
        close,
        volume,
        lookback_period,
        confirmation_period,
        use_volume_confirmation,
        trend_ma_period,
        trend_ma_type,
        ma_step_period,
    )?;
    crate::write_wasm_object_f64_outputs("reversal_signals_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reversal_signals_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = reversal_signals_batch_js(open, high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "reversal_signals_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, ParamKV,
        ParamValue,
    };

    fn assert_vec_eq_nan(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            assert!(
                (a - e).abs() <= 1e-12,
                "mismatch at {idx}: actual={a}, expected={e}"
            );
        }
    }

    fn sample_ohlcv(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);
        for i in 0..len {
            let t = i as f64;
            let base = 100.0 + (t * 0.14).sin() * 6.0 + (t * 0.03).cos() * 2.0;
            let drift = (t * 0.021).sin() * 1.8;
            let o = base + drift;
            let c = base - drift * 0.7 + (t * 0.17).sin() * 0.9;
            let h = o.max(c) + 1.2 + (t * 0.07).sin().abs();
            let l = o.min(c) - 1.1 - (t * 0.11).cos().abs() * 0.7;
            let v = 1000.0 + (t * 0.19).sin().abs() * 700.0 + (i % 11) as f64 * 25.0;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(v);
        }
        (open, high, low, close, volume)
    }

    #[test]
    fn reversal_signals_output_contract() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, volume) = sample_ohlcv(256);
        let out = reversal_signals_with_kernel(
            &ReversalSignalsInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                &volume,
                ReversalSignalsParams::default(),
            ),
            Kernel::Scalar,
        )?;
        assert_eq!(out.buy_signal.len(), close.len());
        assert_eq!(out.sell_signal.len(), close.len());
        assert_eq!(out.stepped_ma.len(), close.len());
        assert_eq!(out.state.len(), close.len());
        assert!(out.stepped_ma.iter().any(|v| v.is_finite()));
        assert!(out.state.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn reversal_signals_rejects_invalid_params() {
        let (open, high, low, close, volume) = sample_ohlcv(64);
        let err = reversal_signals_with_kernel(
            &ReversalSignalsInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                &volume,
                ReversalSignalsParams {
                    lookback_period: Some(0),
                    ..ReversalSignalsParams::default()
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReversalSignalsError::InvalidLookbackPeriod { .. }
        ));

        let err = reversal_signals_with_kernel(
            &ReversalSignalsInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                &volume,
                ReversalSignalsParams {
                    trend_ma_period: Some(0),
                    ..ReversalSignalsParams::default()
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReversalSignalsError::InvalidTrendMaPeriod { .. }
        ));

        let err = reversal_signals_with_kernel(
            &ReversalSignalsInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                &volume,
                ReversalSignalsParams {
                    trend_ma_type: Some("bad".to_string()),
                    ..ReversalSignalsParams::default()
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ReversalSignalsError::InvalidTrendMaType { .. }
        ));
    }

    #[test]
    fn reversal_signals_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, volume) = sample_ohlcv(180);
        let input = ReversalSignalsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            ReversalSignalsParams::default(),
        );
        let baseline = reversal_signals_with_kernel(&input, Kernel::Scalar)?;
        let mut buy_signal = vec![f64::NAN; close.len()];
        let mut sell_signal = vec![f64::NAN; close.len()];
        let mut stepped_ma = vec![f64::NAN; close.len()];
        let mut state = vec![f64::NAN; close.len()];
        reversal_signals_into_slice(
            &mut buy_signal,
            &mut sell_signal,
            &mut stepped_ma,
            &mut state,
            &input,
            Kernel::Scalar,
        )?;
        assert_vec_eq_nan(&baseline.buy_signal, &buy_signal);
        assert_vec_eq_nan(&baseline.sell_signal, &sell_signal);
        assert_vec_eq_nan(&baseline.stepped_ma, &stepped_ma);
        assert_vec_eq_nan(&baseline.state, &state);
        Ok(())
    }

    #[test]
    fn reversal_signals_into_overwrites_stale_buffers() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, volume) = sample_ohlcv(180);
        let input = ReversalSignalsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            ReversalSignalsParams {
                trend_ma_type: Some("SMA".to_string()),
                ..ReversalSignalsParams::default()
            },
        );
        let baseline = reversal_signals_with_kernel(&input, Kernel::Scalar)?;
        let mut buy_signal = vec![12345.0; close.len()];
        let mut sell_signal = vec![12345.0; close.len()];
        let mut stepped_ma = vec![12345.0; close.len()];
        let mut state = vec![12345.0; close.len()];
        reversal_signals_into_slice(
            &mut buy_signal,
            &mut sell_signal,
            &mut stepped_ma,
            &mut state,
            &input,
            Kernel::Scalar,
        )?;
        assert_vec_eq_nan(&baseline.buy_signal, &buy_signal);
        assert_vec_eq_nan(&baseline.sell_signal, &sell_signal);
        assert_vec_eq_nan(&baseline.stepped_ma, &stepped_ma);
        assert_vec_eq_nan(&baseline.state, &state);
        Ok(())
    }

    #[test]
    fn reversal_signals_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, volume) = sample_ohlcv(200);
        let single = reversal_signals(&ReversalSignalsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            ReversalSignalsParams::default(),
        ))?;
        let batch = reversal_signals_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &volume,
            &ReversalSignalsBatchRange::default(),
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_vec_eq_nan(&batch.buy_signal, &single.buy_signal);
        assert_vec_eq_nan(&batch.sell_signal, &single.sell_signal);
        assert_vec_eq_nan(&batch.stepped_ma, &single.stepped_ma);
        assert_vec_eq_nan(&batch.state, &single.state);
        Ok(())
    }

    #[test]
    fn reversal_signals_stream_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, volume) = sample_ohlcv(220);
        let safe = reversal_signals(&ReversalSignalsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            ReversalSignalsParams {
                trend_ma_type: Some("VWMA".to_string()),
                ..ReversalSignalsParams::default()
            },
        ))?;
        let mut stream = ReversalSignalsStream::try_new(ReversalSignalsParams {
            trend_ma_type: Some("VWMA".to_string()),
            ..ReversalSignalsParams::default()
        })?;
        let mut buy_signal = Vec::with_capacity(close.len());
        let mut sell_signal = Vec::with_capacity(close.len());
        let mut stepped_ma = Vec::with_capacity(close.len());
        let mut state = Vec::with_capacity(close.len());

        for ((((o, h), l), c), v) in open
            .iter()
            .zip(high.iter())
            .zip(low.iter())
            .zip(close.iter())
            .zip(volume.iter())
        {
            let point = stream.update(*o, *h, *l, *c, *v).unwrap();
            buy_signal.push(point.0);
            sell_signal.push(point.1);
            stepped_ma.push(point.2);
            state.push(point.3);
        }

        assert_vec_eq_nan(&safe.buy_signal, &buy_signal);
        assert_vec_eq_nan(&safe.sell_signal, &sell_signal);
        assert_vec_eq_nan(&safe.stepped_ma, &stepped_ma);
        assert_vec_eq_nan(&safe.state, &state);
        Ok(())
    }

    #[test]
    fn reversal_signals_dispatch_compute_returns_expected_outputs() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, volume) = sample_ohlcv(192);
        let params = [
            ParamKV {
                key: "lookback_period",
                value: ParamValue::Int(DEFAULT_LOOKBACK_PERIOD as i64),
            },
            ParamKV {
                key: "confirmation_period",
                value: ParamValue::Int(DEFAULT_CONFIRMATION_PERIOD as i64),
            },
            ParamKV {
                key: "use_volume_confirmation",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "trend_ma_period",
                value: ParamValue::Int(DEFAULT_TREND_MA_PERIOD as i64),
            },
            ParamKV {
                key: "trend_ma_type",
                value: ParamValue::EnumString("EMA"),
            },
            ParamKV {
                key: "ma_step_period",
                value: ParamValue::Int(DEFAULT_MA_STEP_PERIOD as i64),
            },
        ];

        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "reversal_signals",
            output_id: Some("stepped_ma"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            params: &params,
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(out.output_id, "stepped_ma");
        match out.series {
            IndicatorSeries::F64(values) => assert_eq!(values.len(), close.len()),
            other => panic!("expected f64 series, got {:?}", other),
        }

        let buy_out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "reversal_signals",
            output_id: Some("buy_signal"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            params: &params,
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(buy_out.output_id, "buy_signal");
        Ok(())
    }
}
