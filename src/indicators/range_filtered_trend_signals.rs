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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_KALMAN_ALPHA: f64 = 0.01;
const DEFAULT_KALMAN_BETA: f64 = 0.1;
const DEFAULT_KALMAN_PERIOD: usize = 77;
const DEFAULT_DEV: f64 = 1.2;
const DEFAULT_SUPERTREND_FACTOR: f64 = 0.7;
const DEFAULT_SUPERTREND_ATR_PERIOD: usize = 7;
const WMA_PERIOD: usize = 200;

#[derive(Debug, Clone)]
pub enum RangeFilteredTrendSignalsData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct RangeFilteredTrendSignalsOutput {
    pub kalman: Vec<f64>,
    pub supertrend: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub trend: Vec<f64>,
    pub kalman_trend: Vec<f64>,
    pub state: Vec<f64>,
    pub market_trending: Vec<f64>,
    pub market_ranging: Vec<f64>,
    pub short_term_bullish: Vec<f64>,
    pub short_term_bearish: Vec<f64>,
    pub long_term_bullish: Vec<f64>,
    pub long_term_bearish: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RangeFilteredTrendSignalsParams {
    pub kalman_alpha: Option<f64>,
    pub kalman_beta: Option<f64>,
    pub kalman_period: Option<usize>,
    pub dev: Option<f64>,
    pub supertrend_factor: Option<f64>,
    pub supertrend_atr_period: Option<usize>,
}

impl Default for RangeFilteredTrendSignalsParams {
    fn default() -> Self {
        Self {
            kalman_alpha: Some(DEFAULT_KALMAN_ALPHA),
            kalman_beta: Some(DEFAULT_KALMAN_BETA),
            kalman_period: Some(DEFAULT_KALMAN_PERIOD),
            dev: Some(DEFAULT_DEV),
            supertrend_factor: Some(DEFAULT_SUPERTREND_FACTOR),
            supertrend_atr_period: Some(DEFAULT_SUPERTREND_ATR_PERIOD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangeFilteredTrendSignalsInput<'a> {
    pub data: RangeFilteredTrendSignalsData<'a>,
    pub params: RangeFilteredTrendSignalsParams,
}

impl<'a> RangeFilteredTrendSignalsInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: RangeFilteredTrendSignalsParams) -> Self {
        Self {
            data: RangeFilteredTrendSignalsData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: RangeFilteredTrendSignalsParams,
    ) -> Self {
        Self {
            data: RangeFilteredTrendSignalsData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, RangeFilteredTrendSignalsParams::default())
    }
}

#[derive(Clone, Debug)]
pub struct RangeFilteredTrendSignalsBuilder {
    kalman_alpha: Option<f64>,
    kalman_beta: Option<f64>,
    kalman_period: Option<usize>,
    dev: Option<f64>,
    supertrend_factor: Option<f64>,
    supertrend_atr_period: Option<usize>,
    kernel: Kernel,
}

impl Default for RangeFilteredTrendSignalsBuilder {
    fn default() -> Self {
        Self {
            kalman_alpha: None,
            kalman_beta: None,
            kalman_period: None,
            dev: None,
            supertrend_factor: None,
            supertrend_atr_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RangeFilteredTrendSignalsBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kalman_alpha(mut self, value: f64) -> Self {
        self.kalman_alpha = Some(value);
        self
    }

    #[inline(always)]
    pub fn kalman_beta(mut self, value: f64) -> Self {
        self.kalman_beta = Some(value);
        self
    }

    #[inline(always)]
    pub fn kalman_period(mut self, value: usize) -> Self {
        self.kalman_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn dev(mut self, value: f64) -> Self {
        self.dev = Some(value);
        self
    }

    #[inline(always)]
    pub fn supertrend_factor(mut self, value: f64) -> Self {
        self.supertrend_factor = Some(value);
        self
    }

    #[inline(always)]
    pub fn supertrend_atr_period(mut self, value: usize) -> Self {
        self.supertrend_atr_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }
}

#[derive(Debug, Error)]
pub enum RangeFilteredTrendSignalsError {
    #[error("range_filtered_trend_signals: input data slice is empty")]
    EmptyInputData,
    #[error(
        "range_filtered_trend_signals: data length mismatch: high={high}, low={low}, close={close}"
    )]
    DataLengthMismatch {
        high: usize,
        low: usize,
        close: usize,
    },
    #[error("range_filtered_trend_signals: all values are NaN")]
    AllValuesNaN,
    #[error("range_filtered_trend_signals: invalid kalman_alpha: {kalman_alpha}")]
    InvalidKalmanAlpha { kalman_alpha: f64 },
    #[error("range_filtered_trend_signals: invalid kalman_beta: {kalman_beta}")]
    InvalidKalmanBeta { kalman_beta: f64 },
    #[error("range_filtered_trend_signals: invalid kalman_period: kalman_period = {kalman_period}, data length = {data_len}")]
    InvalidKalmanPeriod {
        kalman_period: usize,
        data_len: usize,
    },
    #[error("range_filtered_trend_signals: invalid dev: {dev}")]
    InvalidDev { dev: f64 },
    #[error("range_filtered_trend_signals: invalid supertrend_factor: {supertrend_factor}")]
    InvalidSupertrendFactor { supertrend_factor: f64 },
    #[error("range_filtered_trend_signals: invalid supertrend_atr_period: supertrend_atr_period = {supertrend_atr_period}, data length = {data_len}")]
    InvalidSupertrendAtrPeriod {
        supertrend_atr_period: usize,
        data_len: usize,
    },
    #[error(
        "range_filtered_trend_signals: not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "range_filtered_trend_signals: output length mismatch: expected {expected}, got {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("range_filtered_trend_signals: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("range_filtered_trend_signals: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    kalman_alpha: f64,
    kalman_beta: f64,
    kalman_period: usize,
    dev: f64,
    supertrend_factor: f64,
    supertrend_atr_period: usize,
}

#[derive(Clone, Debug)]
struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    len: usize,
    params: ResolvedParams,
    warmup: usize,
    all_valid_from_first: bool,
}

#[derive(Clone, Copy, Debug)]
struct RangeFilteredTrendSignalsPoint {
    kalman: f64,
    supertrend: f64,
    upper_band: f64,
    lower_band: f64,
    trend: f64,
    kalman_trend: f64,
    state: f64,
    market_trending: f64,
    market_ranging: f64,
    short_term_bullish: f64,
    short_term_bearish: f64,
    long_term_bullish: f64,
    long_term_bearish: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RangeFilteredTrendSignalsStreamOutput {
    pub kalman: f64,
    pub supertrend: f64,
    pub upper_band: f64,
    pub lower_band: f64,
    pub trend: f64,
    pub kalman_trend: f64,
    pub state: f64,
    pub market_trending: f64,
    pub market_ranging: f64,
    pub short_term_bullish: f64,
    pub short_term_bearish: f64,
    pub long_term_bullish: f64,
    pub long_term_bearish: f64,
}

#[derive(Clone, Debug)]
struct KalmanState {
    alpha_mul_period: f64,
    beta_div_period: f64,
    value: Option<f64>,
    covariance: f64,
}

impl KalmanState {
    #[inline(always)]
    fn new(alpha: f64, period: usize, beta: f64) -> Self {
        Self {
            alpha_mul_period: alpha * period as f64,
            beta_div_period: beta / period as f64,
            value: None,
            covariance: 1.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.value = None;
        self.covariance = 1.0;
    }

    #[inline(always)]
    fn update(&mut self, input: f64, prev_input: Option<f64>) -> Option<f64> {
        let gain = self.covariance / (self.covariance + self.alpha_mul_period);
        if self.value.is_none() {
            self.value = prev_input;
        }
        let out = self.value.map(|prior| {
            let next = prior + gain * (input - prior);
            self.value = Some(next);
            next
        });
        self.covariance = (1.0 - gain) * self.covariance + self.beta_div_period;
        out
    }
}

#[derive(Clone, Debug)]
struct AtrState {
    period: usize,
    count: usize,
    sum: f64,
    value: Option<f64>,
    prev_close: Option<f64>,
}

impl AtrState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            sum: 0.0,
            value: None,
            prev_close: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = None;
        self.prev_close = None;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = if let Some(prev_close) = self.prev_close {
            let hl = high - low;
            let hc = (high - prev_close).abs();
            let lc = (low - prev_close).abs();
            hl.max(hc).max(lc)
        } else {
            high - low
        };
        self.prev_close = Some(close);
        if let Some(prev) = self.value {
            let next = ((prev * (self.period as f64 - 1.0)) + tr) / self.period as f64;
            self.value = Some(next);
            Some(next)
        } else {
            self.count += 1;
            self.sum += tr;
            if self.count == self.period {
                let seeded = self.sum / self.period as f64;
                self.value = Some(seeded);
                Some(seeded)
            } else {
                None
            }
        }
    }
}

#[derive(Clone, Debug)]
struct WmaState {
    period: usize,
    buffer: Vec<f64>,
    pos: usize,
    len: usize,
    sum: f64,
    weighted_sum: f64,
    divisor: f64,
}

impl WmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buffer: vec![0.0; period],
            pos: 0,
            len: 0,
            sum: 0.0,
            weighted_sum: 0.0,
            divisor: (period * (period + 1) / 2) as f64,
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
        if self.len < self.period {
            self.buffer[self.pos] = value;
            self.pos = (self.pos + 1) % self.period;
            self.len += 1;
            self.sum += value;
            self.weighted_sum += self.len as f64 * value;
            if self.len == self.period {
                Some(self.weighted_sum / self.divisor)
            } else {
                None
            }
        } else {
            let oldest = self.buffer[self.pos];
            let old_sum = self.sum;
            self.buffer[self.pos] = value;
            self.pos = (self.pos + 1) % self.period;
            self.weighted_sum = self.weighted_sum - old_sum + self.period as f64 * value;
            self.sum = old_sum - oldest + value;
            Some(self.weighted_sum / self.divisor)
        }
    }
}

#[derive(Clone, Debug)]
struct SuperTrendState {
    factor: f64,
    prev_lower_band: Option<f64>,
    prev_upper_band: Option<f64>,
    prev_supertrend: Option<f64>,
    prev_k: Option<f64>,
    prev_atr_ready: bool,
}

impl SuperTrendState {
    #[inline(always)]
    fn new(factor: f64) -> Self {
        Self {
            factor,
            prev_lower_band: None,
            prev_upper_band: None,
            prev_supertrend: None,
            prev_k: None,
            prev_atr_ready: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev_lower_band = None;
        self.prev_upper_band = None;
        self.prev_supertrend = None;
        self.prev_k = None;
        self.prev_atr_ready = false;
    }

    #[inline(always)]
    fn update(&mut self, k: f64, atr: f64) -> (f64, i32) {
        let mut upper_band = k + self.factor * atr;
        let mut lower_band = k - self.factor * atr;
        let prev_lower_band = self.prev_lower_band.unwrap_or(lower_band);
        let prev_upper_band = self.prev_upper_band.unwrap_or(upper_band);
        let prev_k = self.prev_k.unwrap_or(k);

        if !(lower_band > prev_lower_band || prev_k < prev_lower_band) {
            lower_band = prev_lower_band;
        }
        if !(upper_band < prev_upper_band || prev_k > prev_upper_band) {
            upper_band = prev_upper_band;
        }

        let direction = if !self.prev_atr_ready {
            1
        } else if self.prev_supertrend == Some(prev_upper_band) {
            if k > upper_band {
                -1
            } else {
                1
            }
        } else if k < lower_band {
            1
        } else {
            -1
        };

        let supertrend = if direction == -1 {
            lower_band
        } else {
            upper_band
        };
        self.prev_lower_band = Some(lower_band);
        self.prev_upper_band = Some(upper_band);
        self.prev_supertrend = Some(supertrend);
        self.prev_k = Some(k);
        self.prev_atr_ready = true;
        (supertrend, direction)
    }
}

#[derive(Clone, Debug)]
struct RangeFilteredTrendSignalsCore {
    dev: f64,
    kalman: KalmanState,
    atr: AtrState,
    wma: WmaState,
    supertrend: SuperTrendState,
    prev_close: Option<f64>,
    trend_state: f64,
    prev_trend: Option<f64>,
    prev_kalman_trend: Option<f64>,
    prev_state: Option<f64>,
}

impl RangeFilteredTrendSignalsCore {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        Self {
            dev: params.dev,
            kalman: KalmanState::new(
                params.kalman_alpha,
                params.kalman_period,
                params.kalman_beta,
            ),
            atr: AtrState::new(params.supertrend_atr_period),
            wma: WmaState::new(WMA_PERIOD),
            supertrend: SuperTrendState::new(params.supertrend_factor),
            prev_close: None,
            trend_state: 0.0,
            prev_trend: None,
            prev_kalman_trend: None,
            prev_state: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.kalman.reset();
        self.atr.reset();
        self.wma.reset();
        self.supertrend.reset();
        self.prev_close = None;
        self.trend_state = 0.0;
        self.prev_trend = None;
        self.prev_kalman_trend = None;
        self.prev_state = None;
    }

    #[inline(always)]
    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<RangeFilteredTrendSignalsPoint> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.reset();
            return None;
        }

        let prev_close = self.prev_close;
        let kalman = self.kalman.update(close, prev_close);
        self.prev_close = Some(close);
        let atr = self.atr.update(high, low, close);
        let vola = self.wma.update(high - low);
        let supertrend_out = match (kalman, atr) {
            (Some(k), Some(a)) => Some(self.supertrend.update(k, a)),
            _ => None,
        };
        let (kalman, vola, supertrend, direction) = match (kalman, vola, supertrend_out) {
            (Some(k), Some(v), Some((s, d))) => (k, v, s, d),
            _ => return None,
        };

        let upper_band = kalman + vola * self.dev;
        let lower_band = kalman - vola * self.dev;
        if close > upper_band {
            self.trend_state = 1.0;
        } else if close < lower_band {
            self.trend_state = -1.0;
        }

        let kalman_trend = if direction < 0 { 1.0 } else { -1.0 };
        let state = kalman_trend * self.trend_state;
        let market_trending = if self
            .prev_state
            .map(|prev| state > 0.0 && prev <= 0.0)
            .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };
        let market_ranging = if self
            .prev_state
            .map(|prev| state < 0.0 && prev >= 0.0)
            .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };
        let short_term_bullish = if self
            .prev_trend
            .map(|prev| self.trend_state > 0.0 && prev <= 0.0)
            .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };
        let short_term_bearish = if self
            .prev_trend
            .map(|prev| self.trend_state < 0.0 && prev >= 0.0)
            .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };
        let long_term_bullish = if self
            .prev_kalman_trend
            .map(|prev| kalman_trend > 0.0 && prev <= 0.0)
            .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };
        let long_term_bearish = if self
            .prev_kalman_trend
            .map(|prev| kalman_trend < 0.0 && prev >= 0.0)
            .unwrap_or(false)
        {
            1.0
        } else {
            0.0
        };

        self.prev_trend = Some(self.trend_state);
        self.prev_kalman_trend = Some(kalman_trend);
        self.prev_state = Some(state);

        Some(RangeFilteredTrendSignalsPoint {
            kalman,
            supertrend,
            upper_band,
            lower_band,
            trend: self.trend_state,
            kalman_trend,
            state,
            market_trending,
            market_ranging,
            short_term_bullish,
            short_term_bearish,
            long_term_bullish,
            long_term_bearish,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RangeFilteredTrendSignalsStream {
    core: RangeFilteredTrendSignalsCore,
}

impl RangeFilteredTrendSignalsStream {
    #[inline]
    pub fn try_new(
        params: RangeFilteredTrendSignalsParams,
    ) -> Result<Self, RangeFilteredTrendSignalsError> {
        let resolved = resolve_params(params, usize::MAX)?;
        Ok(Self {
            core: RangeFilteredTrendSignalsCore::new(resolved),
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<RangeFilteredTrendSignalsStreamOutput> {
        self.core
            .update(high, low, close)
            .map(|point| RangeFilteredTrendSignalsStreamOutput {
                kalman: point.kalman,
                supertrend: point.supertrend,
                upper_band: point.upper_band,
                lower_band: point.lower_band,
                trend: point.trend,
                kalman_trend: point.kalman_trend,
                state: point.state,
                market_trending: point.market_trending,
                market_ranging: point.market_ranging,
                short_term_bullish: point.short_term_bullish,
                short_term_bearish: point.short_term_bearish,
                long_term_bullish: point.long_term_bullish,
                long_term_bearish: point.long_term_bearish,
            })
    }
}

#[inline]
pub fn range_filtered_trend_signals(
    input: &RangeFilteredTrendSignalsInput<'_>,
) -> Result<RangeFilteredTrendSignalsOutput, RangeFilteredTrendSignalsError> {
    range_filtered_trend_signals_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn range_filtered_trend_signals_with_kernel(
    input: &RangeFilteredTrendSignalsInput<'_>,
    kernel: Kernel,
) -> Result<RangeFilteredTrendSignalsOutput, RangeFilteredTrendSignalsError> {
    let prepared = prepare_input(input, kernel)?;
    let mut kalman = alloc_with_nan_prefix(prepared.len, 0);
    let mut supertrend = alloc_with_nan_prefix(prepared.len, 0);
    let mut upper_band = alloc_with_nan_prefix(prepared.len, 0);
    let mut lower_band = alloc_with_nan_prefix(prepared.len, 0);
    let mut trend = alloc_with_nan_prefix(prepared.len, 0);
    let mut kalman_trend = alloc_with_nan_prefix(prepared.len, 0);
    let mut state = alloc_with_nan_prefix(prepared.len, 0);
    let mut market_trending = alloc_with_nan_prefix(prepared.len, 0);
    let mut market_ranging = alloc_with_nan_prefix(prepared.len, 0);
    let mut short_term_bullish = alloc_with_nan_prefix(prepared.len, 0);
    let mut short_term_bearish = alloc_with_nan_prefix(prepared.len, 0);
    let mut long_term_bullish = alloc_with_nan_prefix(prepared.len, 0);
    let mut long_term_bearish = alloc_with_nan_prefix(prepared.len, 0);

    range_filtered_trend_signals_into_slices(
        input,
        kernel,
        &mut kalman,
        &mut supertrend,
        &mut upper_band,
        &mut lower_band,
        &mut trend,
        &mut kalman_trend,
        &mut state,
        &mut market_trending,
        &mut market_ranging,
        &mut short_term_bullish,
        &mut short_term_bearish,
        &mut long_term_bullish,
        &mut long_term_bearish,
    )?;

    Ok(RangeFilteredTrendSignalsOutput {
        kalman,
        supertrend,
        upper_band,
        lower_band,
        trend,
        kalman_trend,
        state,
        market_trending,
        market_ranging,
        short_term_bullish,
        short_term_bearish,
        long_term_bullish,
        long_term_bearish,
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn range_filtered_trend_signals_into(
    input: &RangeFilteredTrendSignalsInput<'_>,
    kalman: &mut [f64],
    supertrend: &mut [f64],
    upper_band: &mut [f64],
    lower_band: &mut [f64],
    trend: &mut [f64],
    kalman_trend: &mut [f64],
    state: &mut [f64],
    market_trending: &mut [f64],
    market_ranging: &mut [f64],
    short_term_bullish: &mut [f64],
    short_term_bearish: &mut [f64],
    long_term_bullish: &mut [f64],
    long_term_bearish: &mut [f64],
) -> Result<(), RangeFilteredTrendSignalsError> {
    range_filtered_trend_signals_into_slices(
        input,
        Kernel::Auto,
        kalman,
        supertrend,
        upper_band,
        lower_band,
        trend,
        kalman_trend,
        state,
        market_trending,
        market_ranging,
        short_term_bullish,
        short_term_bearish,
        long_term_bullish,
        long_term_bearish,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn range_filtered_trend_signals_into_slices(
    input: &RangeFilteredTrendSignalsInput<'_>,
    kernel: Kernel,
    kalman: &mut [f64],
    supertrend: &mut [f64],
    upper_band: &mut [f64],
    lower_band: &mut [f64],
    trend: &mut [f64],
    kalman_trend: &mut [f64],
    state: &mut [f64],
    market_trending: &mut [f64],
    market_ranging: &mut [f64],
    short_term_bullish: &mut [f64],
    short_term_bearish: &mut [f64],
    long_term_bullish: &mut [f64],
    long_term_bearish: &mut [f64],
) -> Result<(), RangeFilteredTrendSignalsError> {
    let prepared = prepare_input(input, kernel)?;
    let got = *[
        kalman.len(),
        supertrend.len(),
        upper_band.len(),
        lower_band.len(),
        trend.len(),
        kalman_trend.len(),
        state.len(),
        market_trending.len(),
        market_ranging.len(),
        short_term_bullish.len(),
        short_term_bearish.len(),
        long_term_bullish.len(),
        long_term_bearish.len(),
    ]
    .iter()
    .min()
    .unwrap_or(&0);
    if kalman.len() != prepared.len
        || supertrend.len() != prepared.len
        || upper_band.len() != prepared.len
        || lower_band.len() != prepared.len
        || trend.len() != prepared.len
        || kalman_trend.len() != prepared.len
        || state.len() != prepared.len
        || market_trending.len() != prepared.len
        || market_ranging.len() != prepared.len
        || short_term_bullish.len() != prepared.len
        || short_term_bearish.len() != prepared.len
        || long_term_bullish.len() != prepared.len
        || long_term_bearish.len() != prepared.len
    {
        return Err(RangeFilteredTrendSignalsError::OutputLengthMismatch {
            expected: prepared.len,
            got,
        });
    }

    compute_into_slices(
        &prepared,
        kalman,
        supertrend,
        upper_band,
        lower_band,
        trend,
        kalman_trend,
        state,
        market_trending,
        market_ranging,
        short_term_bullish,
        short_term_bearish,
        long_term_bullish,
        long_term_bearish,
    )
}

#[inline]
fn resolve_data<'a>(
    input: &'a RangeFilteredTrendSignalsInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), RangeFilteredTrendSignalsError> {
    match &input.data {
        RangeFilteredTrendSignalsData::Candles { candles } => Ok((
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        RangeFilteredTrendSignalsData::Slices { high, low, close } => {
            if high.len() != low.len() || high.len() != close.len() {
                return Err(RangeFilteredTrendSignalsError::DataLengthMismatch {
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                });
            }
            Ok((high, low, close))
        }
    }
}

#[inline]
fn resolve_params(
    params: RangeFilteredTrendSignalsParams,
    data_len: usize,
) -> Result<ResolvedParams, RangeFilteredTrendSignalsError> {
    let kalman_alpha = params.kalman_alpha.unwrap_or(DEFAULT_KALMAN_ALPHA);
    let kalman_beta = params.kalman_beta.unwrap_or(DEFAULT_KALMAN_BETA);
    let kalman_period = params.kalman_period.unwrap_or(DEFAULT_KALMAN_PERIOD);
    let dev = params.dev.unwrap_or(DEFAULT_DEV);
    let supertrend_factor = params
        .supertrend_factor
        .unwrap_or(DEFAULT_SUPERTREND_FACTOR);
    let supertrend_atr_period = params
        .supertrend_atr_period
        .unwrap_or(DEFAULT_SUPERTREND_ATR_PERIOD);

    if !kalman_alpha.is_finite() || kalman_alpha <= 0.0 {
        return Err(RangeFilteredTrendSignalsError::InvalidKalmanAlpha { kalman_alpha });
    }
    if !kalman_beta.is_finite() || kalman_beta < 0.0 {
        return Err(RangeFilteredTrendSignalsError::InvalidKalmanBeta { kalman_beta });
    }
    if kalman_period == 0 || (data_len != usize::MAX && kalman_period > data_len) {
        return Err(RangeFilteredTrendSignalsError::InvalidKalmanPeriod {
            kalman_period,
            data_len,
        });
    }
    if !dev.is_finite() || dev < 0.0 {
        return Err(RangeFilteredTrendSignalsError::InvalidDev { dev });
    }
    if !supertrend_factor.is_finite() || supertrend_factor < 0.0 {
        return Err(RangeFilteredTrendSignalsError::InvalidSupertrendFactor { supertrend_factor });
    }
    if supertrend_atr_period == 0 || (data_len != usize::MAX && supertrend_atr_period > data_len) {
        return Err(RangeFilteredTrendSignalsError::InvalidSupertrendAtrPeriod {
            supertrend_atr_period,
            data_len,
        });
    }

    Ok(ResolvedParams {
        kalman_alpha,
        kalman_beta,
        kalman_period,
        dev,
        supertrend_factor,
        supertrend_atr_period,
    })
}

#[inline]
fn prepare_input<'a>(
    input: &'a RangeFilteredTrendSignalsInput<'a>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, RangeFilteredTrendSignalsError> {
    let (high, low, close) = resolve_data(input)?;
    let len = close.len();
    if len == 0 {
        return Err(RangeFilteredTrendSignalsError::EmptyInputData);
    }
    let first = (0..len)
        .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        .ok_or(RangeFilteredTrendSignalsError::AllValuesNaN)?;
    let params = resolve_params(input.params.clone(), len)?;
    let valid = (first..len)
        .filter(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        .count();
    let all_valid_from_first = valid == len - first;
    let needed = WMA_PERIOD.max(params.supertrend_atr_period).max(2);
    if valid < needed {
        return Err(RangeFilteredTrendSignalsError::NotEnoughValidData { needed, valid });
    }
    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        value => value,
    };
    Ok(PreparedInput {
        high,
        low,
        close,
        len,
        params,
        warmup: first + (WMA_PERIOD - 1).max(params.supertrend_atr_period.saturating_sub(1)),
        all_valid_from_first,
    })
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn compute_into_slices(
    prepared: &PreparedInput<'_>,
    dst_kalman: &mut [f64],
    dst_supertrend: &mut [f64],
    dst_upper_band: &mut [f64],
    dst_lower_band: &mut [f64],
    dst_trend: &mut [f64],
    dst_kalman_trend: &mut [f64],
    dst_state: &mut [f64],
    dst_market_trending: &mut [f64],
    dst_market_ranging: &mut [f64],
    dst_short_term_bullish: &mut [f64],
    dst_short_term_bearish: &mut [f64],
    dst_long_term_bullish: &mut [f64],
    dst_long_term_bearish: &mut [f64],
) -> Result<(), RangeFilteredTrendSignalsError> {
    let init_len = if prepared.all_valid_from_first {
        prepared.warmup.min(prepared.len)
    } else {
        prepared.len
    };
    dst_kalman[..init_len].fill(f64::NAN);
    dst_supertrend[..init_len].fill(f64::NAN);
    dst_upper_band[..init_len].fill(f64::NAN);
    dst_lower_band[..init_len].fill(f64::NAN);
    dst_trend[..init_len].fill(f64::NAN);
    dst_kalman_trend[..init_len].fill(f64::NAN);
    dst_state[..init_len].fill(f64::NAN);
    dst_market_trending[..init_len].fill(f64::NAN);
    dst_market_ranging[..init_len].fill(f64::NAN);
    dst_short_term_bullish[..init_len].fill(f64::NAN);
    dst_short_term_bearish[..init_len].fill(f64::NAN);
    dst_long_term_bullish[..init_len].fill(f64::NAN);
    dst_long_term_bearish[..init_len].fill(f64::NAN);

    let mut core = RangeFilteredTrendSignalsCore::new(prepared.params);
    for i in 0..prepared.len {
        let Some(point) = core.update(prepared.high[i], prepared.low[i], prepared.close[i]) else {
            continue;
        };
        dst_kalman[i] = point.kalman;
        dst_supertrend[i] = point.supertrend;
        dst_upper_band[i] = point.upper_band;
        dst_lower_band[i] = point.lower_band;
        dst_trend[i] = point.trend;
        dst_kalman_trend[i] = point.kalman_trend;
        dst_state[i] = point.state;
        dst_market_trending[i] = point.market_trending;
        dst_market_ranging[i] = point.market_ranging;
        dst_short_term_bullish[i] = point.short_term_bullish;
        dst_short_term_bearish[i] = point.short_term_bearish;
        dst_long_term_bullish[i] = point.long_term_bullish;
        dst_long_term_bearish[i] = point.long_term_bearish;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct RangeFilteredTrendSignalsBatchRange {
    pub kalman_alpha: (f64, f64, f64),
    pub kalman_beta: (f64, f64, f64),
    pub kalman_period: (usize, usize, usize),
    pub dev: (f64, f64, f64),
    pub supertrend_factor: (f64, f64, f64),
    pub supertrend_atr_period: (usize, usize, usize),
}

impl Default for RangeFilteredTrendSignalsBatchRange {
    fn default() -> Self {
        Self {
            kalman_alpha: (DEFAULT_KALMAN_ALPHA, DEFAULT_KALMAN_ALPHA, 0.0),
            kalman_beta: (DEFAULT_KALMAN_BETA, DEFAULT_KALMAN_BETA, 0.0),
            kalman_period: (DEFAULT_KALMAN_PERIOD, DEFAULT_KALMAN_PERIOD, 0),
            dev: (DEFAULT_DEV, DEFAULT_DEV, 0.0),
            supertrend_factor: (DEFAULT_SUPERTREND_FACTOR, DEFAULT_SUPERTREND_FACTOR, 0.0),
            supertrend_atr_period: (
                DEFAULT_SUPERTREND_ATR_PERIOD,
                DEFAULT_SUPERTREND_ATR_PERIOD,
                0,
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RangeFilteredTrendSignalsBatchOutput {
    pub kalman: Vec<f64>,
    pub supertrend: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub trend: Vec<f64>,
    pub kalman_trend: Vec<f64>,
    pub state: Vec<f64>,
    pub market_trending: Vec<f64>,
    pub market_ranging: Vec<f64>,
    pub short_term_bullish: Vec<f64>,
    pub short_term_bearish: Vec<f64>,
    pub long_term_bullish: Vec<f64>,
    pub long_term_bearish: Vec<f64>,
    pub combos: Vec<RangeFilteredTrendSignalsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct RangeFilteredTrendSignalsBatchBuilder {
    range: RangeFilteredTrendSignalsBatchRange,
    kernel: Kernel,
}

impl Default for RangeFilteredTrendSignalsBatchBuilder {
    fn default() -> Self {
        Self {
            range: RangeFilteredTrendSignalsBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl RangeFilteredTrendSignalsBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range(mut self, value: RangeFilteredTrendSignalsBatchRange) -> Self {
        self.range = value;
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
    ) -> Result<RangeFilteredTrendSignalsBatchOutput, RangeFilteredTrendSignalsError> {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<RangeFilteredTrendSignalsBatchOutput, RangeFilteredTrendSignalsError> {
        range_filtered_trend_signals_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, RangeFilteredTrendSignalsError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start <= end {
        let mut current = start;
        while current <= end {
            out.push(current);
            match current.checked_add(step) {
                Some(next) => current = next,
                None => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            out.push(current);
            match current.checked_sub(step) {
                Some(next) => current = next,
                None => break,
            }
            if current < end {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(RangeFilteredTrendSignalsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, RangeFilteredTrendSignalsError> {
    let eps = 1e-12;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(RangeFilteredTrendSignalsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step.abs() < eps || (start - end).abs() < eps {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    let dir = if end >= start { 1.0 } else { -1.0 };
    let step_eff = dir * step.abs();
    let mut current = start;
    if dir > 0.0 {
        while current <= end + eps {
            out.push(current);
            current += step_eff;
        }
    } else {
        while current >= end - eps {
            out.push(current);
            current += step_eff;
        }
    }
    if out.is_empty() {
        return Err(RangeFilteredTrendSignalsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid(
    range: &RangeFilteredTrendSignalsBatchRange,
) -> Result<Vec<RangeFilteredTrendSignalsParams>, RangeFilteredTrendSignalsError> {
    let kalman_alphas = axis_f64(range.kalman_alpha)?;
    let kalman_betas = axis_f64(range.kalman_beta)?;
    let kalman_periods = axis_usize(range.kalman_period)?;
    let devs = axis_f64(range.dev)?;
    let supertrend_factors = axis_f64(range.supertrend_factor)?;
    let supertrend_atr_periods = axis_usize(range.supertrend_atr_period)?;

    let total = kalman_alphas
        .len()
        .checked_mul(kalman_betas.len())
        .and_then(|n| n.checked_mul(kalman_periods.len()))
        .and_then(|n| n.checked_mul(devs.len()))
        .and_then(|n| n.checked_mul(supertrend_factors.len()))
        .and_then(|n| n.checked_mul(supertrend_atr_periods.len()))
        .ok_or_else(|| RangeFilteredTrendSignalsError::InvalidRange {
            start: range.kalman_period.0.to_string(),
            end: range.kalman_period.1.to_string(),
            step: range.kalman_period.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(total);
    for &kalman_alpha in &kalman_alphas {
        for &kalman_beta in &kalman_betas {
            for &kalman_period in &kalman_periods {
                for &dev in &devs {
                    for &supertrend_factor in &supertrend_factors {
                        for &supertrend_atr_period in &supertrend_atr_periods {
                            out.push(RangeFilteredTrendSignalsParams {
                                kalman_alpha: Some(kalman_alpha),
                                kalman_beta: Some(kalman_beta),
                                kalman_period: Some(kalman_period),
                                dev: Some(dev),
                                supertrend_factor: Some(supertrend_factor),
                                supertrend_atr_period: Some(supertrend_atr_period),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn range_filtered_trend_signals_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    range: &RangeFilteredTrendSignalsBatchRange,
    kernel: Kernel,
) -> Result<RangeFilteredTrendSignalsBatchOutput, RangeFilteredTrendSignalsError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(RangeFilteredTrendSignalsError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(RangeFilteredTrendSignalsError::DataLengthMismatch {
            high: high.len(),
            low: low.len(),
            close: close.len(),
        });
    }

    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        value if value.is_batch() => value,
        _ => {
            return Err(RangeFilteredTrendSignalsError::InvalidKernelForBatch(
                kernel,
            ))
        }
    };
    let single_kernel = batch_kernel.to_non_batch();
    let combos = expand_grid(range)?;
    let rows = combos.len();
    let cols = close.len();

    let first = (0..cols)
        .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        .ok_or(RangeFilteredTrendSignalsError::AllValuesNaN)?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            first
                + (WMA_PERIOD - 1).max(
                    combo
                        .supertrend_atr_period
                        .unwrap_or(DEFAULT_SUPERTREND_ATR_PERIOD)
                        .saturating_sub(1),
                )
        })
        .collect();

    let mut kalman_mu = make_uninit_matrix(rows, cols);
    let mut supertrend_mu = make_uninit_matrix(rows, cols);
    let mut upper_band_mu = make_uninit_matrix(rows, cols);
    let mut lower_band_mu = make_uninit_matrix(rows, cols);
    let mut trend_mu = make_uninit_matrix(rows, cols);
    let mut kalman_trend_mu = make_uninit_matrix(rows, cols);
    let mut state_mu = make_uninit_matrix(rows, cols);
    let mut market_trending_mu = make_uninit_matrix(rows, cols);
    let mut market_ranging_mu = make_uninit_matrix(rows, cols);
    let mut short_term_bullish_mu = make_uninit_matrix(rows, cols);
    let mut short_term_bearish_mu = make_uninit_matrix(rows, cols);
    let mut long_term_bullish_mu = make_uninit_matrix(rows, cols);
    let mut long_term_bearish_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut kalman_mu, cols, &warmups);
    init_matrix_prefixes(&mut supertrend_mu, cols, &warmups);
    init_matrix_prefixes(&mut upper_band_mu, cols, &warmups);
    init_matrix_prefixes(&mut lower_band_mu, cols, &warmups);
    init_matrix_prefixes(&mut trend_mu, cols, &warmups);
    init_matrix_prefixes(&mut kalman_trend_mu, cols, &warmups);
    init_matrix_prefixes(&mut state_mu, cols, &warmups);
    init_matrix_prefixes(&mut market_trending_mu, cols, &warmups);
    init_matrix_prefixes(&mut market_ranging_mu, cols, &warmups);
    init_matrix_prefixes(&mut short_term_bullish_mu, cols, &warmups);
    init_matrix_prefixes(&mut short_term_bearish_mu, cols, &warmups);
    init_matrix_prefixes(&mut long_term_bullish_mu, cols, &warmups);
    init_matrix_prefixes(&mut long_term_bearish_mu, cols, &warmups);

    let mut kalman_guard = ManuallyDrop::new(kalman_mu);
    let mut supertrend_guard = ManuallyDrop::new(supertrend_mu);
    let mut upper_band_guard = ManuallyDrop::new(upper_band_mu);
    let mut lower_band_guard = ManuallyDrop::new(lower_band_mu);
    let mut trend_guard = ManuallyDrop::new(trend_mu);
    let mut kalman_trend_guard = ManuallyDrop::new(kalman_trend_mu);
    let mut state_guard = ManuallyDrop::new(state_mu);
    let mut market_trending_guard = ManuallyDrop::new(market_trending_mu);
    let mut market_ranging_guard = ManuallyDrop::new(market_ranging_mu);
    let mut short_term_bullish_guard = ManuallyDrop::new(short_term_bullish_mu);
    let mut short_term_bearish_guard = ManuallyDrop::new(short_term_bearish_mu);
    let mut long_term_bullish_guard = ManuallyDrop::new(long_term_bullish_mu);
    let mut long_term_bearish_guard = ManuallyDrop::new(long_term_bearish_mu);

    let kalman_all = unsafe { mu_slice_as_f64_slice_mut(&mut kalman_guard) };
    let supertrend_all = unsafe { mu_slice_as_f64_slice_mut(&mut supertrend_guard) };
    let upper_band_all = unsafe { mu_slice_as_f64_slice_mut(&mut upper_band_guard) };
    let lower_band_all = unsafe { mu_slice_as_f64_slice_mut(&mut lower_band_guard) };
    let trend_all = unsafe { mu_slice_as_f64_slice_mut(&mut trend_guard) };
    let kalman_trend_all = unsafe { mu_slice_as_f64_slice_mut(&mut kalman_trend_guard) };
    let state_all = unsafe { mu_slice_as_f64_slice_mut(&mut state_guard) };
    let market_trending_all = unsafe { mu_slice_as_f64_slice_mut(&mut market_trending_guard) };
    let market_ranging_all = unsafe { mu_slice_as_f64_slice_mut(&mut market_ranging_guard) };
    let short_term_bullish_all =
        unsafe { mu_slice_as_f64_slice_mut(&mut short_term_bullish_guard) };
    let short_term_bearish_all =
        unsafe { mu_slice_as_f64_slice_mut(&mut short_term_bearish_guard) };
    let long_term_bullish_all = unsafe { mu_slice_as_f64_slice_mut(&mut long_term_bullish_guard) };
    let long_term_bearish_all = unsafe { mu_slice_as_f64_slice_mut(&mut long_term_bearish_guard) };

    let run_row = |row: usize,
                   kalman_row: &mut [f64],
                   supertrend_row: &mut [f64],
                   upper_band_row: &mut [f64],
                   lower_band_row: &mut [f64],
                   trend_row: &mut [f64],
                   kalman_trend_row: &mut [f64],
                   state_row: &mut [f64],
                   market_trending_row: &mut [f64],
                   market_ranging_row: &mut [f64],
                   short_term_bullish_row: &mut [f64],
                   short_term_bearish_row: &mut [f64],
                   long_term_bullish_row: &mut [f64],
                   long_term_bearish_row: &mut [f64]|
     -> Result<(), RangeFilteredTrendSignalsError> {
        let input =
            RangeFilteredTrendSignalsInput::from_slices(high, low, close, combos[row].clone());
        range_filtered_trend_signals_into_slices(
            &input,
            single_kernel,
            kalman_row,
            supertrend_row,
            upper_band_row,
            lower_band_row,
            trend_row,
            kalman_trend_row,
            state_row,
            market_trending_row,
            market_ranging_row,
            short_term_bullish_row,
            short_term_bearish_row,
            long_term_bullish_row,
            long_term_bearish_row,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        kalman_all
            .par_chunks_mut(cols)
            .zip(supertrend_all.par_chunks_mut(cols))
            .zip(upper_band_all.par_chunks_mut(cols))
            .zip(lower_band_all.par_chunks_mut(cols))
            .zip(trend_all.par_chunks_mut(cols))
            .zip(kalman_trend_all.par_chunks_mut(cols))
            .zip(state_all.par_chunks_mut(cols))
            .zip(market_trending_all.par_chunks_mut(cols))
            .zip(market_ranging_all.par_chunks_mut(cols))
            .zip(short_term_bullish_all.par_chunks_mut(cols))
            .zip(short_term_bearish_all.par_chunks_mut(cols))
            .zip(long_term_bullish_all.par_chunks_mut(cols))
            .zip(long_term_bearish_all.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(
                |(
                    row,
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (kalman_row, supertrend_row),
                                                                upper_band_row,
                                                            ),
                                                            lower_band_row,
                                                        ),
                                                        trend_row,
                                                    ),
                                                    kalman_trend_row,
                                                ),
                                                state_row,
                                            ),
                                            market_trending_row,
                                        ),
                                        market_ranging_row,
                                    ),
                                    short_term_bullish_row,
                                ),
                                short_term_bearish_row,
                            ),
                            long_term_bullish_row,
                        ),
                        long_term_bearish_row,
                    ),
                )| {
                    run_row(
                        row,
                        kalman_row,
                        supertrend_row,
                        upper_band_row,
                        lower_band_row,
                        trend_row,
                        kalman_trend_row,
                        state_row,
                        market_trending_row,
                        market_ranging_row,
                        short_term_bullish_row,
                        short_term_bearish_row,
                        long_term_bullish_row,
                        long_term_bearish_row,
                    )
                },
            )?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            run_row(
                row,
                &mut kalman_all[start..end],
                &mut supertrend_all[start..end],
                &mut upper_band_all[start..end],
                &mut lower_band_all[start..end],
                &mut trend_all[start..end],
                &mut kalman_trend_all[start..end],
                &mut state_all[start..end],
                &mut market_trending_all[start..end],
                &mut market_ranging_all[start..end],
                &mut short_term_bullish_all[start..end],
                &mut short_term_bearish_all[start..end],
                &mut long_term_bullish_all[start..end],
                &mut long_term_bearish_all[start..end],
            )?;
        }
    }

    let kalman = unsafe { assume_init_vec(kalman_guard) };
    let supertrend = unsafe { assume_init_vec(supertrend_guard) };
    let upper_band = unsafe { assume_init_vec(upper_band_guard) };
    let lower_band = unsafe { assume_init_vec(lower_band_guard) };
    let trend = unsafe { assume_init_vec(trend_guard) };
    let kalman_trend = unsafe { assume_init_vec(kalman_trend_guard) };
    let state = unsafe { assume_init_vec(state_guard) };
    let market_trending = unsafe { assume_init_vec(market_trending_guard) };
    let market_ranging = unsafe { assume_init_vec(market_ranging_guard) };
    let short_term_bullish = unsafe { assume_init_vec(short_term_bullish_guard) };
    let short_term_bearish = unsafe { assume_init_vec(short_term_bearish_guard) };
    let long_term_bullish = unsafe { assume_init_vec(long_term_bullish_guard) };
    let long_term_bearish = unsafe { assume_init_vec(long_term_bearish_guard) };

    Ok(RangeFilteredTrendSignalsBatchOutput {
        kalman,
        supertrend,
        upper_band,
        lower_band,
        trend,
        kalman_trend,
        state,
        market_trending,
        market_ranging,
        short_term_bullish,
        short_term_bearish,
        long_term_bullish,
        long_term_bearish,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn mu_slice_as_f64_slice_mut(buf: &mut ManuallyDrop<Vec<MaybeUninit<f64>>>) -> &mut [f64] {
    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f64, buf.len())
}

#[inline(always)]
unsafe fn assume_init_vec(buf: ManuallyDrop<Vec<MaybeUninit<f64>>>) -> Vec<f64> {
    let mut buf = buf;
    Vec::from_raw_parts(buf.as_mut_ptr() as *mut f64, buf.len(), buf.capacity())
}

#[cfg(feature = "python")]
#[pyfunction(name = "range_filtered_trend_signals")]
#[pyo3(signature = (high, low, close, kalman_alpha=DEFAULT_KALMAN_ALPHA, kalman_beta=DEFAULT_KALMAN_BETA, kalman_period=DEFAULT_KALMAN_PERIOD, dev=DEFAULT_DEV, supertrend_factor=DEFAULT_SUPERTREND_FACTOR, supertrend_atr_period=DEFAULT_SUPERTREND_ATR_PERIOD, kernel=None))]
pub fn range_filtered_trend_signals_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    kalman_alpha: f64,
    kalman_beta: f64,
    kalman_period: usize,
    dev: f64,
    supertrend_factor: f64,
    supertrend_atr_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = RangeFilteredTrendSignalsInput::from_slices(
        high,
        low,
        close,
        RangeFilteredTrendSignalsParams {
            kalman_alpha: Some(kalman_alpha),
            kalman_beta: Some(kalman_beta),
            kalman_period: Some(kalman_period),
            dev: Some(dev),
            supertrend_factor: Some(supertrend_factor),
            supertrend_atr_period: Some(supertrend_atr_period),
        },
    );
    let output = py
        .allow_threads(|| range_filtered_trend_signals_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("kalman", output.kalman.into_pyarray(py))?;
    dict.set_item("supertrend", output.supertrend.into_pyarray(py))?;
    dict.set_item("upper_band", output.upper_band.into_pyarray(py))?;
    dict.set_item("lower_band", output.lower_band.into_pyarray(py))?;
    dict.set_item("trend", output.trend.into_pyarray(py))?;
    dict.set_item("kalman_trend", output.kalman_trend.into_pyarray(py))?;
    dict.set_item("state", output.state.into_pyarray(py))?;
    dict.set_item("market_trending", output.market_trending.into_pyarray(py))?;
    dict.set_item("market_ranging", output.market_ranging.into_pyarray(py))?;
    dict.set_item(
        "short_term_bullish",
        output.short_term_bullish.into_pyarray(py),
    )?;
    dict.set_item(
        "short_term_bearish",
        output.short_term_bearish.into_pyarray(py),
    )?;
    dict.set_item(
        "long_term_bullish",
        output.long_term_bullish.into_pyarray(py),
    )?;
    dict.set_item(
        "long_term_bearish",
        output.long_term_bearish.into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "range_filtered_trend_signals_batch")]
#[pyo3(signature = (high, low, close, kalman_alpha_range, kalman_beta_range, kalman_period_range, dev_range, supertrend_factor_range, supertrend_atr_period_range, kernel=None))]
pub fn range_filtered_trend_signals_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    kalman_alpha_range: (f64, f64, f64),
    kalman_beta_range: (f64, f64, f64),
    kalman_period_range: (usize, usize, usize),
    dev_range: (f64, f64, f64),
    supertrend_factor_range: (f64, f64, f64),
    supertrend_atr_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            range_filtered_trend_signals_batch_with_kernel(
                high,
                low,
                close,
                &RangeFilteredTrendSignalsBatchRange {
                    kalman_alpha: kalman_alpha_range,
                    kalman_beta: kalman_beta_range,
                    kalman_period: kalman_period_range,
                    dev: dev_range,
                    supertrend_factor: supertrend_factor_range,
                    supertrend_atr_period: supertrend_atr_period_range,
                },
                kernel,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let total = output.rows * output.cols;
    let arrays = [
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
    ];
    unsafe { arrays[0].as_slice_mut()? }.copy_from_slice(&output.kalman);
    unsafe { arrays[1].as_slice_mut()? }.copy_from_slice(&output.supertrend);
    unsafe { arrays[2].as_slice_mut()? }.copy_from_slice(&output.upper_band);
    unsafe { arrays[3].as_slice_mut()? }.copy_from_slice(&output.lower_band);
    unsafe { arrays[4].as_slice_mut()? }.copy_from_slice(&output.trend);
    unsafe { arrays[5].as_slice_mut()? }.copy_from_slice(&output.kalman_trend);
    unsafe { arrays[6].as_slice_mut()? }.copy_from_slice(&output.state);
    unsafe { arrays[7].as_slice_mut()? }.copy_from_slice(&output.market_trending);
    unsafe { arrays[8].as_slice_mut()? }.copy_from_slice(&output.market_ranging);
    unsafe { arrays[9].as_slice_mut()? }.copy_from_slice(&output.short_term_bullish);
    unsafe { arrays[10].as_slice_mut()? }.copy_from_slice(&output.short_term_bearish);
    unsafe { arrays[11].as_slice_mut()? }.copy_from_slice(&output.long_term_bullish);
    unsafe { arrays[12].as_slice_mut()? }.copy_from_slice(&output.long_term_bearish);

    let dict = PyDict::new(py);
    dict.set_item("kalman", arrays[0].reshape((output.rows, output.cols))?)?;
    dict.set_item("supertrend", arrays[1].reshape((output.rows, output.cols))?)?;
    dict.set_item("upper_band", arrays[2].reshape((output.rows, output.cols))?)?;
    dict.set_item("lower_band", arrays[3].reshape((output.rows, output.cols))?)?;
    dict.set_item("trend", arrays[4].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "kalman_trend",
        arrays[5].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item("state", arrays[6].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "market_trending",
        arrays[7].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "market_ranging",
        arrays[8].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "short_term_bullish",
        arrays[9].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "short_term_bearish",
        arrays[10].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "long_term_bullish",
        arrays[11].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "long_term_bearish",
        arrays[12].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "kalman_alphas",
        output
            .combos
            .iter()
            .map(|combo| combo.kalman_alpha.unwrap_or(DEFAULT_KALMAN_ALPHA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "kalman_betas",
        output
            .combos
            .iter()
            .map(|combo| combo.kalman_beta.unwrap_or(DEFAULT_KALMAN_BETA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "kalman_periods",
        output
            .combos
            .iter()
            .map(|combo| combo.kalman_period.unwrap_or(DEFAULT_KALMAN_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "devs",
        output
            .combos
            .iter()
            .map(|combo| combo.dev.unwrap_or(DEFAULT_DEV))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "supertrend_factors",
        output
            .combos
            .iter()
            .map(|combo| combo.supertrend_factor.unwrap_or(DEFAULT_SUPERTREND_FACTOR))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "supertrend_atr_periods",
        output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .supertrend_atr_period
                    .unwrap_or(DEFAULT_SUPERTREND_ATR_PERIOD) as u64
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "RangeFilteredTrendSignalsStream")]
pub struct RangeFilteredTrendSignalsStreamPy {
    stream: RangeFilteredTrendSignalsStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RangeFilteredTrendSignalsStreamPy {
    #[new]
    #[pyo3(signature = (kalman_alpha=DEFAULT_KALMAN_ALPHA, kalman_beta=DEFAULT_KALMAN_BETA, kalman_period=DEFAULT_KALMAN_PERIOD, dev=DEFAULT_DEV, supertrend_factor=DEFAULT_SUPERTREND_FACTOR, supertrend_atr_period=DEFAULT_SUPERTREND_ATR_PERIOD))]
    fn new(
        kalman_alpha: f64,
        kalman_beta: f64,
        kalman_period: usize,
        dev: f64,
        supertrend_factor: f64,
        supertrend_atr_period: usize,
    ) -> PyResult<Self> {
        let stream = RangeFilteredTrendSignalsStream::try_new(RangeFilteredTrendSignalsParams {
            kalman_alpha: Some(kalman_alpha),
            kalman_beta: Some(kalman_beta),
            kalman_period: Some(kalman_period),
            dev: Some(dev),
            supertrend_factor: Some(supertrend_factor),
            supertrend_atr_period: Some(supertrend_atr_period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<Vec<f64>> {
        self.stream.update(high, low, close).map(|output| {
            vec![
                output.kalman,
                output.supertrend,
                output.upper_band,
                output.lower_band,
                output.trend,
                output.kalman_trend,
                output.state,
                output.market_trending,
                output.market_ranging,
                output.short_term_bullish,
                output.short_term_bearish,
                output.long_term_bullish,
                output.long_term_bearish,
            ]
        })
    }
}

#[cfg(feature = "python")]
pub fn register_range_filtered_trend_signals_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(range_filtered_trend_signals_py, m)?)?;
    m.add_function(wrap_pyfunction!(range_filtered_trend_signals_batch_py, m)?)?;
    m.add_class::<RangeFilteredTrendSignalsStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RangeFilteredTrendSignalsJsOutput {
    pub kalman: Vec<f64>,
    pub supertrend: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub trend: Vec<f64>,
    pub kalman_trend: Vec<f64>,
    pub state: Vec<f64>,
    pub market_trending: Vec<f64>,
    pub market_ranging: Vec<f64>,
    pub short_term_bullish: Vec<f64>,
    pub short_term_bearish: Vec<f64>,
    pub long_term_bullish: Vec<f64>,
    pub long_term_bearish: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = range_filtered_trend_signals_js)]
pub fn range_filtered_trend_signals_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kalman_alpha: f64,
    kalman_beta: f64,
    kalman_period: usize,
    dev: f64,
    supertrend_factor: f64,
    supertrend_atr_period: usize,
) -> Result<JsValue, JsValue> {
    let input = RangeFilteredTrendSignalsInput::from_slices(
        high,
        low,
        close,
        RangeFilteredTrendSignalsParams {
            kalman_alpha: Some(kalman_alpha),
            kalman_beta: Some(kalman_beta),
            kalman_period: Some(kalman_period),
            dev: Some(dev),
            supertrend_factor: Some(supertrend_factor),
            supertrend_atr_period: Some(supertrend_atr_period),
        },
    );
    let output = range_filtered_trend_signals_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&RangeFilteredTrendSignalsJsOutput {
        kalman: output.kalman,
        supertrend: output.supertrend,
        upper_band: output.upper_band,
        lower_band: output.lower_band,
        trend: output.trend,
        kalman_trend: output.kalman_trend,
        state: output.state,
        market_trending: output.market_trending,
        market_ranging: output.market_ranging,
        short_term_bullish: output.short_term_bullish,
        short_term_bearish: output.short_term_bearish,
        long_term_bullish: output.long_term_bullish,
        long_term_bearish: output.long_term_bearish,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RangeFilteredTrendSignalsBatchConfig {
    pub kalman_alpha_range: (f64, f64, f64),
    pub kalman_beta_range: (f64, f64, f64),
    pub kalman_period_range: (usize, usize, usize),
    pub dev_range: (f64, f64, f64),
    pub supertrend_factor_range: (f64, f64, f64),
    pub supertrend_atr_period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RangeFilteredTrendSignalsBatchJsOutput {
    pub kalman: Vec<f64>,
    pub supertrend: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub trend: Vec<f64>,
    pub kalman_trend: Vec<f64>,
    pub state: Vec<f64>,
    pub market_trending: Vec<f64>,
    pub market_ranging: Vec<f64>,
    pub short_term_bullish: Vec<f64>,
    pub short_term_bearish: Vec<f64>,
    pub long_term_bullish: Vec<f64>,
    pub long_term_bearish: Vec<f64>,
    pub kalman_alphas: Vec<f64>,
    pub kalman_betas: Vec<f64>,
    pub kalman_periods: Vec<usize>,
    pub devs: Vec<f64>,
    pub supertrend_factors: Vec<f64>,
    pub supertrend_atr_periods: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = range_filtered_trend_signals_batch)]
pub fn range_filtered_trend_signals_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: RangeFilteredTrendSignalsBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = range_filtered_trend_signals_batch_with_kernel(
        high,
        low,
        close,
        &RangeFilteredTrendSignalsBatchRange {
            kalman_alpha: cfg.kalman_alpha_range,
            kalman_beta: cfg.kalman_beta_range,
            kalman_period: cfg.kalman_period_range,
            dev: cfg.dev_range,
            supertrend_factor: cfg.supertrend_factor_range,
            supertrend_atr_period: cfg.supertrend_atr_period_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&RangeFilteredTrendSignalsBatchJsOutput {
        kalman: output.kalman,
        supertrend: output.supertrend,
        upper_band: output.upper_band,
        lower_band: output.lower_band,
        trend: output.trend,
        kalman_trend: output.kalman_trend,
        state: output.state,
        market_trending: output.market_trending,
        market_ranging: output.market_ranging,
        short_term_bullish: output.short_term_bullish,
        short_term_bearish: output.short_term_bearish,
        long_term_bullish: output.long_term_bullish,
        long_term_bearish: output.long_term_bearish,
        kalman_alphas: output
            .combos
            .iter()
            .map(|combo| combo.kalman_alpha.unwrap_or(DEFAULT_KALMAN_ALPHA))
            .collect(),
        kalman_betas: output
            .combos
            .iter()
            .map(|combo| combo.kalman_beta.unwrap_or(DEFAULT_KALMAN_BETA))
            .collect(),
        kalman_periods: output
            .combos
            .iter()
            .map(|combo| combo.kalman_period.unwrap_or(DEFAULT_KALMAN_PERIOD))
            .collect(),
        devs: output
            .combos
            .iter()
            .map(|combo| combo.dev.unwrap_or(DEFAULT_DEV))
            .collect(),
        supertrend_factors: output
            .combos
            .iter()
            .map(|combo| combo.supertrend_factor.unwrap_or(DEFAULT_SUPERTREND_FACTOR))
            .collect(),
        supertrend_atr_periods: output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .supertrend_atr_period
                    .unwrap_or(DEFAULT_SUPERTREND_ATR_PERIOD)
            })
            .collect(),
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_filtered_trend_signals_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kalman_alpha: f64,
    kalman_beta: f64,
    kalman_period: usize,
    dev: f64,
    supertrend_factor: f64,
    supertrend_atr_period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = range_filtered_trend_signals_js(
        high,
        low,
        close,
        kalman_alpha,
        kalman_beta,
        kalman_period,
        dev,
        supertrend_factor,
        supertrend_atr_period,
    )?;
    crate::write_wasm_object_f64_outputs("range_filtered_trend_signals_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_filtered_trend_signals_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = range_filtered_trend_signals_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "range_filtered_trend_signals_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(360);
        let mut low = Vec::with_capacity(360);
        let mut close = Vec::with_capacity(360);
        for i in 0..360 {
            let base = 100.0 + i as f64 * 0.11 + (i as f64 * 0.19).sin() * 1.8;
            let c = base + (i as f64 * 0.07).cos() * 0.55;
            let h = c + 0.9 + (i as f64 * 0.13).sin().abs() * 0.25;
            let l = c - 0.9 - (i as f64 * 0.09).cos().abs() * 0.35;
            high.push(h);
            low.push(l);
            close.push(c);
        }
        (high, low, close)
    }

    #[test]
    fn range_filtered_trend_signals_into_matches_single() {
        let (high, low, close) = sample_ohlc();
        let input = RangeFilteredTrendSignalsInput::from_slices(
            &high,
            &low,
            &close,
            RangeFilteredTrendSignalsParams::default(),
        );
        let out = range_filtered_trend_signals_with_kernel(&input, Kernel::Scalar).expect("single");
        let mut kalman = vec![0.0; close.len()];
        let mut supertrend = vec![0.0; close.len()];
        let mut upper_band = vec![0.0; close.len()];
        let mut lower_band = vec![0.0; close.len()];
        let mut trend = vec![0.0; close.len()];
        let mut kalman_trend = vec![0.0; close.len()];
        let mut state = vec![0.0; close.len()];
        let mut market_trending = vec![0.0; close.len()];
        let mut market_ranging = vec![0.0; close.len()];
        let mut short_term_bullish = vec![0.0; close.len()];
        let mut short_term_bearish = vec![0.0; close.len()];
        let mut long_term_bullish = vec![0.0; close.len()];
        let mut long_term_bearish = vec![0.0; close.len()];

        range_filtered_trend_signals_into_slices(
            &input,
            Kernel::Scalar,
            &mut kalman,
            &mut supertrend,
            &mut upper_band,
            &mut lower_band,
            &mut trend,
            &mut kalman_trend,
            &mut state,
            &mut market_trending,
            &mut market_ranging,
            &mut short_term_bullish,
            &mut short_term_bearish,
            &mut long_term_bullish,
            &mut long_term_bearish,
        )
        .expect("into");

        for i in 0..close.len() {
            for (lhs, rhs) in [
                (out.kalman[i], kalman[i]),
                (out.supertrend[i], supertrend[i]),
                (out.upper_band[i], upper_band[i]),
                (out.lower_band[i], lower_band[i]),
                (out.trend[i], trend[i]),
                (out.kalman_trend[i], kalman_trend[i]),
                (out.state[i], state[i]),
                (out.market_trending[i], market_trending[i]),
                (out.market_ranging[i], market_ranging[i]),
                (out.short_term_bullish[i], short_term_bullish[i]),
                (out.short_term_bearish[i], short_term_bearish[i]),
                (out.long_term_bullish[i], long_term_bullish[i]),
                (out.long_term_bearish[i], long_term_bearish[i]),
            ] {
                if lhs.is_nan() {
                    assert!(rhs.is_nan());
                } else {
                    assert!((lhs - rhs).abs() <= 1e-12);
                }
            }
        }
    }

    #[test]
    fn range_filtered_trend_signals_stream_matches_batch() {
        let (high, low, close) = sample_ohlc();
        let input = RangeFilteredTrendSignalsInput::from_slices(
            &high,
            &low,
            &close,
            RangeFilteredTrendSignalsParams::default(),
        );
        let out = range_filtered_trend_signals(&input).expect("batch");
        let mut stream =
            RangeFilteredTrendSignalsStream::try_new(RangeFilteredTrendSignalsParams::default())
                .expect("stream");
        let mut collected = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            collected.push(stream.update(high[i], low[i], close[i]));
        }
        for i in 0..close.len() {
            let Some(point) = collected[i] else {
                assert!(out.kalman[i].is_nan());
                continue;
            };
            for (lhs, rhs) in [
                (point.kalman, out.kalman[i]),
                (point.supertrend, out.supertrend[i]),
                (point.upper_band, out.upper_band[i]),
                (point.lower_band, out.lower_band[i]),
                (point.trend, out.trend[i]),
                (point.kalman_trend, out.kalman_trend[i]),
                (point.state, out.state[i]),
                (point.market_trending, out.market_trending[i]),
                (point.market_ranging, out.market_ranging[i]),
                (point.short_term_bullish, out.short_term_bullish[i]),
                (point.short_term_bearish, out.short_term_bearish[i]),
                (point.long_term_bullish, out.long_term_bullish[i]),
                (point.long_term_bearish, out.long_term_bearish[i]),
            ] {
                if rhs.is_nan() {
                    assert!(lhs.is_nan());
                } else {
                    assert!((lhs - rhs).abs() <= 1e-12);
                }
            }
        }
    }

    #[test]
    fn range_filtered_trend_signals_batch_first_row_matches_single() {
        let (high, low, close) = sample_ohlc();
        let single = range_filtered_trend_signals(&RangeFilteredTrendSignalsInput::from_slices(
            &high,
            &low,
            &close,
            RangeFilteredTrendSignalsParams::default(),
        ))
        .expect("single");
        let batch = range_filtered_trend_signals_batch_with_kernel(
            &high,
            &low,
            &close,
            &RangeFilteredTrendSignalsBatchRange {
                kalman_alpha: (0.01, 0.02, 0.01),
                kalman_beta: (0.1, 0.1, 0.0),
                kalman_period: (77, 77, 0),
                dev: (1.2, 1.2, 0.0),
                supertrend_factor: (0.7, 0.7, 0.0),
                supertrend_atr_period: (7, 7, 0),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, close.len());
        for i in 0..close.len() {
            let idx = i;
            for (lhs, rhs) in [
                (single.kalman[i], batch.kalman[idx]),
                (single.supertrend[i], batch.supertrend[idx]),
                (single.upper_band[i], batch.upper_band[idx]),
                (single.lower_band[i], batch.lower_band[idx]),
                (single.trend[i], batch.trend[idx]),
                (single.kalman_trend[i], batch.kalman_trend[idx]),
                (single.state[i], batch.state[idx]),
                (single.market_trending[i], batch.market_trending[idx]),
                (single.market_ranging[i], batch.market_ranging[idx]),
                (single.short_term_bullish[i], batch.short_term_bullish[idx]),
                (single.short_term_bearish[i], batch.short_term_bearish[idx]),
                (single.long_term_bullish[i], batch.long_term_bullish[idx]),
                (single.long_term_bearish[i], batch.long_term_bearish[idx]),
            ] {
                if lhs.is_nan() {
                    assert!(rhs.is_nan());
                } else {
                    assert!((lhs - rhs).abs() <= 1e-12);
                }
            }
        }
    }

    #[test]
    fn range_filtered_trend_signals_rejects_invalid_params() {
        let (high, low, close) = sample_ohlc();
        let err = range_filtered_trend_signals(&RangeFilteredTrendSignalsInput::from_slices(
            &high,
            &low,
            &close,
            RangeFilteredTrendSignalsParams {
                kalman_alpha: Some(0.0),
                ..RangeFilteredTrendSignalsParams::default()
            },
        ))
        .expect_err("invalid alpha");
        assert!(err.to_string().contains("invalid kalman_alpha"));

        let err = RangeFilteredTrendSignalsStream::try_new(RangeFilteredTrendSignalsParams {
            dev: Some(-1.0),
            ..RangeFilteredTrendSignalsParams::default()
        })
        .expect_err("invalid dev");
        assert!(err.to_string().contains("invalid dev"));
    }
}
