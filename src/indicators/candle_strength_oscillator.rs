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
use std::error::Error;
use thiserror::Error;

const DEFAULT_PERIOD: usize = 50;
const DEFAULT_ATR_LENGTH: usize = 50;
const DEFAULT_MODE: &str = "bollinger";
const BOLLINGER_STD_MULTIPLIER: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandleStrengthOscillatorMode {
    Bollinger,
    Donchian,
}

impl CandleStrengthOscillatorMode {
    #[inline(always)]
    fn from_str(value: &str) -> Result<Self, CandleStrengthOscillatorError> {
        if value.eq_ignore_ascii_case("bollinger") || value.eq_ignore_ascii_case("bb") {
            return Ok(Self::Bollinger);
        }
        if value.eq_ignore_ascii_case("donchian") || value.eq_ignore_ascii_case("dc") {
            return Ok(Self::Donchian);
        }
        Err(CandleStrengthOscillatorError::InvalidMode {
            mode: value.to_string(),
        })
    }
}

#[derive(Debug, Clone)]
pub enum CandleStrengthOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct CandleStrengthOscillatorOutput {
    pub strength: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub mid: Vec<f64>,
    pub long_signal: Vec<f64>,
    pub short_signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandleStrengthOscillatorOutputField {
    Strength,
    Highs,
    Lows,
    Mid,
    LongSignal,
    ShortSignal,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct CandleStrengthOscillatorParams {
    pub period: Option<usize>,
    pub atr_enabled: Option<bool>,
    pub atr_length: Option<usize>,
    pub mode: Option<String>,
}

impl Default for CandleStrengthOscillatorParams {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
            atr_enabled: Some(false),
            atr_length: Some(DEFAULT_ATR_LENGTH),
            mode: Some(DEFAULT_MODE.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CandleStrengthOscillatorInput<'a> {
    pub data: CandleStrengthOscillatorData<'a>,
    pub params: CandleStrengthOscillatorParams,
}

impl<'a> CandleStrengthOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: CandleStrengthOscillatorParams) -> Self {
        Self {
            data: CandleStrengthOscillatorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: CandleStrengthOscillatorParams,
    ) -> Self {
        Self {
            data: CandleStrengthOscillatorData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, CandleStrengthOscillatorParams::default())
    }

    #[inline(always)]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(DEFAULT_PERIOD)
    }

    #[inline(always)]
    pub fn get_atr_enabled(&self) -> bool {
        self.params.atr_enabled.unwrap_or(false)
    }

    #[inline(always)]
    pub fn get_atr_length(&self) -> usize {
        self.params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH)
    }

    #[inline(always)]
    pub fn get_mode(&self) -> &str {
        self.params.mode.as_deref().unwrap_or(DEFAULT_MODE)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CandleStrengthOscillatorBuilder {
    period: Option<usize>,
    atr_enabled: Option<bool>,
    atr_length: Option<usize>,
    mode: Option<&'static str>,
    kernel: Kernel,
}

impl Default for CandleStrengthOscillatorBuilder {
    fn default() -> Self {
        Self {
            period: None,
            atr_enabled: None,
            atr_length: None,
            mode: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CandleStrengthOscillatorBuilder {
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
    pub fn atr_enabled(mut self, value: bool) -> Self {
        self.atr_enabled = Some(value);
        self
    }

    #[inline(always)]
    pub fn atr_length(mut self, value: usize) -> Self {
        self.atr_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn mode(mut self, value: &'static str) -> Self {
        self.mode = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    fn build_params(self) -> CandleStrengthOscillatorParams {
        CandleStrengthOscillatorParams {
            period: self.period,
            atr_enabled: self.atr_enabled,
            atr_length: self.atr_length,
            mode: self.mode.map(str::to_string),
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<CandleStrengthOscillatorOutput, CandleStrengthOscillatorError> {
        candle_strength_oscillator_with_kernel(
            &CandleStrengthOscillatorInput::from_candles(candles, self.build_params()),
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
    ) -> Result<CandleStrengthOscillatorOutput, CandleStrengthOscillatorError> {
        candle_strength_oscillator_with_kernel(
            &CandleStrengthOscillatorInput::from_slices(
                open,
                high,
                low,
                close,
                self.build_params(),
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<CandleStrengthOscillatorStream, CandleStrengthOscillatorError> {
        CandleStrengthOscillatorStream::try_new(self.build_params())
    }
}

#[derive(Debug, Error)]
pub enum CandleStrengthOscillatorError {
    #[error("candle_strength_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "candle_strength_oscillator: Input length mismatch: open = {open_len}, high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    InputLengthMismatch {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("candle_strength_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("candle_strength_oscillator: Invalid period: {period}")]
    InvalidPeriod { period: usize },
    #[error("candle_strength_oscillator: Invalid atr_length: {atr_length}")]
    InvalidAtrLength { atr_length: usize },
    #[error("candle_strength_oscillator: Invalid mode: {mode}")]
    InvalidMode { mode: String },
    #[error(
        "candle_strength_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "candle_strength_oscillator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "candle_strength_oscillator: Invalid range: {field} start={start} end={end} step={step}"
    )]
    InvalidRange {
        field: &'static str,
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("candle_strength_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "candle_strength_oscillator: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("candle_strength_oscillator: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone, Copy)]
struct CandleStrengthOscillatorResolved {
    period: usize,
    atr_enabled: bool,
    atr_length: usize,
    mode: CandleStrengthOscillatorMode,
}

#[derive(Debug, Clone, Copy)]
pub struct CandleStrengthOscillatorPoint {
    pub strength: f64,
    pub highs: f64,
    pub lows: f64,
    pub mid: f64,
    pub long_signal: f64,
    pub short_signal: f64,
}

#[derive(Debug, Clone)]
struct WilderAtrState {
    length: usize,
    alpha: f64,
    prev_close: f64,
    rma: f64,
    warm_sum: f64,
    warm_count: usize,
    seeded: bool,
}

impl WilderAtrState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            length,
            alpha: 1.0 / length as f64,
            prev_close: f64::NAN,
            rma: f64::NAN,
            warm_sum: 0.0,
            warm_count: 0,
            seeded: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.rma = f64::NAN;
        self.warm_sum = 0.0;
        self.warm_count = 0;
        self.seeded = false;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.reset();
            return None;
        }
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let up = if high > self.prev_close {
                high
            } else {
                self.prev_close
            };
            let dn = if low < self.prev_close {
                low
            } else {
                self.prev_close
            };
            up - dn
        };
        self.prev_close = close;
        if !self.seeded {
            self.warm_sum += tr;
            self.warm_count += 1;
            if self.warm_count == self.length {
                self.rma = self.warm_sum * self.alpha;
                self.seeded = true;
                return Some(self.rma);
            }
            return None;
        }
        self.rma = self.alpha.mul_add(tr - self.rma, self.rma);
        Some(self.rma)
    }
}

#[derive(Debug, Clone)]
struct LinearWmaState {
    period: usize,
    denom: f64,
    window: VecDeque<f64>,
    sum: f64,
    weighted_sum: f64,
}

impl LinearWmaState {
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
        if self.period <= 1 {
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
struct HmaState {
    half: LinearWmaState,
    full: LinearWmaState,
    sqrt: LinearWmaState,
}

impl HmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let half = (period / 2).max(1);
        let sqrt = sqrt_period(period);
        Self {
            half: LinearWmaState::new(half),
            full: LinearWmaState::new(period.max(1)),
            sqrt: LinearWmaState::new(sqrt),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.half.reset();
        self.full.reset();
        self.sqrt.reset();
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        let full = self.full.update(value);
        let half = self.half.update(value);
        if let (Some(full), Some(half)) = (full, half) {
            self.sqrt.update(2.0f64.mul_add(half, -full))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct SmaStdDevState {
    period: usize,
    inv_n: f64,
    buf: Vec<f64>,
    idx: usize,
    len: usize,
    sum: f64,
    sum_sq: f64,
}

impl SmaStdDevState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            inv_n: 1.0 / period as f64,
            buf: vec![0.0; period.max(1)],
            idx: 0,
            len: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.idx = 0;
        self.len = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if self.period <= 1 {
            return Some((value, 0.0));
        }
        if self.len < self.period {
            self.buf[self.idx] = value;
            self.idx = (self.idx + 1) % self.period;
            self.len += 1;
            self.sum += value;
            self.sum_sq = value.mul_add(value, self.sum_sq);
            if self.len < self.period {
                return None;
            }
        } else {
            let old = self.buf[self.idx];
            self.buf[self.idx] = value;
            self.idx = (self.idx + 1) % self.period;
            self.sum += value - old;
            self.sum_sq = value.mul_add(value, self.sum_sq - old * old);
        }
        let mean = self.sum * self.inv_n;
        let var = (self.sum_sq * self.inv_n) - mean * mean;
        Some((mean, if var > 0.0 { var.sqrt() } else { 0.0 }))
    }
}

#[derive(Debug, Clone)]
struct DonchianState {
    period: usize,
    window: VecDeque<f64>,
}

impl DonchianState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            window: VecDeque::with_capacity(period.max(1)),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.window.clear();
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if self.period <= 1 {
            return Some((value, value, value));
        }
        if self.window.len() == self.period {
            self.window.pop_front();
        }
        self.window.push_back(value);
        if self.window.len() < self.period {
            return None;
        }
        let mut high = f64::NEG_INFINITY;
        let mut low = f64::INFINITY;
        for &v in &self.window {
            if v > high {
                high = v;
            }
            if v < low {
                low = v;
            }
        }
        Some((high, low, 0.5 * (high + low)))
    }
}

#[derive(Debug, Clone)]
enum LevelState {
    Bollinger(SmaStdDevState),
    Donchian(DonchianState),
}

impl LevelState {
    #[inline(always)]
    fn reset(&mut self) {
        match self {
            Self::Bollinger(state) => state.reset(),
            Self::Donchian(state) => state.reset(),
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        match self {
            Self::Bollinger(state) => state.update(value).map(|(mean, stddev)| {
                (
                    mean + BOLLINGER_STD_MULTIPLIER * stddev,
                    mean - BOLLINGER_STD_MULTIPLIER * stddev,
                    mean,
                )
            }),
            Self::Donchian(state) => state.update(value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CandleStrengthOscillatorStream {
    period: usize,
    atr_enabled: bool,
    atr_state: WilderAtrState,
    hma_state: HmaState,
    level_state: LevelState,
    prev_strength: f64,
    prev_mid: f64,
    has_prev_levels: bool,
}

impl CandleStrengthOscillatorStream {
    #[inline(always)]
    pub fn try_new(
        params: CandleStrengthOscillatorParams,
    ) -> Result<Self, CandleStrengthOscillatorError> {
        let resolved = resolve_params(&params)?;
        Ok(Self {
            period: resolved.period,
            atr_enabled: resolved.atr_enabled,
            atr_state: WilderAtrState::new(resolved.atr_length),
            hma_state: HmaState::new(resolved.period),
            level_state: match resolved.mode {
                CandleStrengthOscillatorMode::Bollinger => {
                    LevelState::Bollinger(SmaStdDevState::new(resolved.period))
                }
                CandleStrengthOscillatorMode::Donchian => {
                    LevelState::Donchian(DonchianState::new(resolved.period))
                }
            },
            prev_strength: f64::NAN,
            prev_mid: f64::NAN,
            has_prev_levels: false,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.atr_state.reset();
        self.hma_state.reset();
        self.level_state.reset();
        self.prev_strength = f64::NAN;
        self.prev_mid = f64::NAN;
        self.has_prev_levels = false;
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<CandleStrengthOscillatorPoint> {
        if !is_valid_quad(open, high, low, close) {
            self.reset();
            return None;
        }
        let atr_factor = if self.atr_enabled {
            self.atr_state.update(high, low, close)?
        } else {
            1.0
        };
        let range_high_low = high - low;
        if !range_high_low.is_finite() || range_high_low.abs() <= f64::EPSILON {
            self.reset();
            return None;
        }
        let body = (close - open).abs();
        let signed_score = if close > open {
            body / range_high_low
        } else {
            -(body / range_high_low)
        } * atr_factor
            * 100.0;
        let strength = self.hma_state.update(signed_score)?;
        let mut point = CandleStrengthOscillatorPoint {
            strength,
            highs: f64::NAN,
            lows: f64::NAN,
            mid: f64::NAN,
            long_signal: 0.0,
            short_signal: 0.0,
        };
        if let Some((highs, lows, mid)) = self.level_state.update(strength) {
            point.highs = highs;
            point.lows = lows;
            point.mid = mid;
            if self.has_prev_levels {
                if self.prev_strength <= self.prev_mid && strength > mid {
                    point.long_signal = 1.0;
                }
                if self.prev_strength >= self.prev_mid && strength < mid {
                    point.short_signal = 1.0;
                }
            }
            self.prev_strength = strength;
            self.prev_mid = mid;
            self.has_prev_levels = true;
        }
        Some(point)
    }

    #[inline(always)]
    pub fn get_strength_warmup_period(&self) -> usize {
        strength_warmup_prefix(self.period, self.atr_enabled, self.atr_state.length)
    }

    #[inline(always)]
    pub fn get_levels_warmup_period(&self) -> usize {
        levels_warmup_prefix(self.period, self.atr_enabled, self.atr_state.length)
    }
}

#[inline(always)]
fn sqrt_period(period: usize) -> usize {
    (period as f64).sqrt().floor() as usize
}

#[inline(always)]
fn hma_warmup_prefix(period: usize) -> usize {
    period.saturating_add(sqrt_period(period)).saturating_sub(2)
}

#[inline(always)]
fn atr_warmup_prefix(atr_enabled: bool, atr_length: usize) -> usize {
    if atr_enabled {
        atr_length.saturating_sub(1)
    } else {
        0
    }
}

#[inline(always)]
fn strength_warmup_prefix(period: usize, atr_enabled: bool, atr_length: usize) -> usize {
    atr_warmup_prefix(atr_enabled, atr_length).saturating_add(hma_warmup_prefix(period))
}

#[inline(always)]
fn levels_warmup_prefix(period: usize, atr_enabled: bool, atr_length: usize) -> usize {
    strength_warmup_prefix(period, atr_enabled, atr_length).saturating_add(period.saturating_sub(1))
}

#[inline(always)]
fn levels_needed_bars(period: usize, atr_enabled: bool, atr_length: usize) -> usize {
    levels_warmup_prefix(period, atr_enabled, atr_length).saturating_add(1)
}

#[inline(always)]
fn resolve_params(
    params: &CandleStrengthOscillatorParams,
) -> Result<CandleStrengthOscillatorResolved, CandleStrengthOscillatorError> {
    let period = params.period.unwrap_or(DEFAULT_PERIOD);
    if period == 0 {
        return Err(CandleStrengthOscillatorError::InvalidPeriod { period });
    }
    let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
    if atr_length == 0 {
        return Err(CandleStrengthOscillatorError::InvalidAtrLength { atr_length });
    }
    let atr_enabled = params.atr_enabled.unwrap_or(false);
    let mode =
        CandleStrengthOscillatorMode::from_str(params.mode.as_deref().unwrap_or(DEFAULT_MODE))?;
    Ok(CandleStrengthOscillatorResolved {
        period,
        atr_enabled,
        atr_length,
        mode,
    })
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a CandleStrengthOscillatorInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), CandleStrengthOscillatorError> {
    match &input.data {
        CandleStrengthOscillatorData::Candles { candles } => Ok((
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        CandleStrengthOscillatorData::Slices {
            open,
            high,
            low,
            close,
        } => Ok((open, high, low, close)),
    }
}

#[inline(always)]
fn is_valid_quad(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline(always)]
fn longest_valid_run(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for (((&o, &h), &l), &c) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
    {
        if is_valid_quad(o, h, l, c) {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

fn validate_common(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    resolved: CandleStrengthOscillatorResolved,
) -> Result<(), CandleStrengthOscillatorError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(CandleStrengthOscillatorError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(CandleStrengthOscillatorError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let valid = longest_valid_run(open, high, low, close);
    if valid == 0 {
        return Err(CandleStrengthOscillatorError::AllValuesNaN);
    }
    let needed = levels_needed_bars(resolved.period, resolved.atr_enabled, resolved.atr_length);
    if valid < needed {
        return Err(CandleStrengthOscillatorError::NotEnoughValidData { needed, valid });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    resolved: CandleStrengthOscillatorResolved,
    strength: &mut [f64],
    highs: &mut [f64],
    lows: &mut [f64],
    mid: &mut [f64],
    long_signal: &mut [f64],
    short_signal: &mut [f64],
) {
    strength.fill(f64::NAN);
    highs.fill(f64::NAN);
    lows.fill(f64::NAN);
    mid.fill(f64::NAN);
    long_signal.fill(0.0);
    short_signal.fill(0.0);

    let mut stream = CandleStrengthOscillatorStream::try_new(CandleStrengthOscillatorParams {
        period: Some(resolved.period),
        atr_enabled: Some(resolved.atr_enabled),
        atr_length: Some(resolved.atr_length),
        mode: Some(match resolved.mode {
            CandleStrengthOscillatorMode::Bollinger => "bollinger".to_string(),
            CandleStrengthOscillatorMode::Donchian => "donchian".to_string(),
        }),
    })
    .expect("validated params");

    for i in 0..open.len() {
        if let Some(point) = stream.update(open[i], high[i], low[i], close[i]) {
            strength[i] = point.strength;
            if point.highs.is_finite() {
                highs[i] = point.highs;
            }
            if point.lows.is_finite() {
                lows[i] = point.lows;
            }
            if point.mid.is_finite() {
                mid[i] = point.mid;
            }
            long_signal[i] = point.long_signal;
            short_signal[i] = point.short_signal;
        }
    }
}

pub fn candle_strength_oscillator(
    input: &CandleStrengthOscillatorInput,
) -> Result<CandleStrengthOscillatorOutput, CandleStrengthOscillatorError> {
    candle_strength_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn candle_strength_oscillator_with_kernel(
    input: &CandleStrengthOscillatorInput,
    kernel: Kernel,
) -> Result<CandleStrengthOscillatorOutput, CandleStrengthOscillatorError> {
    let (open, high, low, close) = input_slices(input)?;
    let resolved = resolve_params(&input.params)?;
    validate_common(open, high, low, close, resolved)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let len = close.len();
    let strength_prefix =
        strength_warmup_prefix(resolved.period, resolved.atr_enabled, resolved.atr_length);
    let levels_prefix =
        levels_warmup_prefix(resolved.period, resolved.atr_enabled, resolved.atr_length);
    let mut strength = alloc_with_nan_prefix(len, strength_prefix);
    let mut highs = alloc_with_nan_prefix(len, levels_prefix);
    let mut lows = alloc_with_nan_prefix(len, levels_prefix);
    let mut mid = alloc_with_nan_prefix(len, levels_prefix);
    let mut long_signal = vec![0.0; len];
    let mut short_signal = vec![0.0; len];
    compute_row(
        open,
        high,
        low,
        close,
        resolved,
        &mut strength,
        &mut highs,
        &mut lows,
        &mut mid,
        &mut long_signal,
        &mut short_signal,
    );
    Ok(CandleStrengthOscillatorOutput {
        strength,
        highs,
        lows,
        mid,
        long_signal,
        short_signal,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn candle_strength_oscillator_into_slice(
    dst_strength: &mut [f64],
    dst_highs: &mut [f64],
    dst_lows: &mut [f64],
    dst_mid: &mut [f64],
    dst_long_signal: &mut [f64],
    dst_short_signal: &mut [f64],
    input: &CandleStrengthOscillatorInput,
    kernel: Kernel,
) -> Result<(), CandleStrengthOscillatorError> {
    let (open, high, low, close) = input_slices(input)?;
    let resolved = resolve_params(&input.params)?;
    validate_common(open, high, low, close, resolved)?;
    let expected = close.len();
    for dst in [
        &*dst_strength,
        &*dst_highs,
        &*dst_lows,
        &*dst_mid,
        &*dst_long_signal,
        &*dst_short_signal,
    ] {
        if dst.len() != expected {
            return Err(CandleStrengthOscillatorError::OutputLengthMismatch {
                expected,
                got: dst.len(),
            });
        }
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    compute_row(
        open,
        high,
        low,
        close,
        resolved,
        dst_strength,
        dst_highs,
        dst_lows,
        dst_mid,
        dst_long_signal,
        dst_short_signal,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[allow(clippy::too_many_arguments)]
pub fn candle_strength_oscillator_into(
    input: &CandleStrengthOscillatorInput,
    dst_strength: &mut [f64],
    dst_highs: &mut [f64],
    dst_lows: &mut [f64],
    dst_mid: &mut [f64],
    dst_long_signal: &mut [f64],
    dst_short_signal: &mut [f64],
) -> Result<(), CandleStrengthOscillatorError> {
    candle_strength_oscillator_into_slice(
        dst_strength,
        dst_highs,
        dst_lows,
        dst_mid,
        dst_long_signal,
        dst_short_signal,
        input,
        Kernel::Auto,
    )
}

pub fn candle_strength_oscillator_output_into_slice(
    dst: &mut [f64],
    input: &CandleStrengthOscillatorInput,
    kernel: Kernel,
    field: CandleStrengthOscillatorOutputField,
) -> Result<(), CandleStrengthOscillatorError> {
    let (open, high, low, close) = input_slices(input)?;
    let resolved = resolve_params(&input.params)?;
    validate_common(open, high, low, close, resolved)?;
    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    if dst.len() != close.len() {
        return Err(CandleStrengthOscillatorError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }

    match field {
        CandleStrengthOscillatorOutputField::LongSignal
        | CandleStrengthOscillatorOutputField::ShortSignal => dst.fill(0.0),
        _ => dst.fill(f64::NAN),
    }

    let mut stream = CandleStrengthOscillatorStream::try_new(input.params.clone())?;
    for i in 0..close.len() {
        if let Some(point) = stream.update(open[i], high[i], low[i], close[i]) {
            dst[i] = match field {
                CandleStrengthOscillatorOutputField::Strength => point.strength,
                CandleStrengthOscillatorOutputField::Highs => point.highs,
                CandleStrengthOscillatorOutputField::Lows => point.lows,
                CandleStrengthOscillatorOutputField::Mid => point.mid,
                CandleStrengthOscillatorOutputField::LongSignal => point.long_signal,
                CandleStrengthOscillatorOutputField::ShortSignal => point.short_signal,
            };
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct CandleStrengthOscillatorBatchRange {
    pub period: (usize, usize, usize),
    pub atr_length: (usize, usize, usize),
}

impl Default for CandleStrengthOscillatorBatchRange {
    fn default() -> Self {
        Self {
            period: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
            atr_length: (DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CandleStrengthOscillatorBatchOutput {
    pub strength: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub mid: Vec<f64>,
    pub long_signal: Vec<f64>,
    pub short_signal: Vec<f64>,
    pub combos: Vec<CandleStrengthOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct CandleStrengthOscillatorBatchBuilder {
    range: CandleStrengthOscillatorBatchRange,
    atr_enabled: bool,
    mode: &'static str,
    kernel: Kernel,
}

impl Default for CandleStrengthOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: CandleStrengthOscillatorBatchRange::default(),
            atr_enabled: false,
            mode: DEFAULT_MODE,
            kernel: Kernel::Auto,
        }
    }
}

impl CandleStrengthOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.period = value;
        self
    }

    #[inline(always)]
    pub fn atr_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.atr_length = value;
        self
    }

    #[inline(always)]
    pub fn atr_enabled(mut self, value: bool) -> Self {
        self.atr_enabled = value;
        self
    }

    #[inline(always)]
    pub fn mode(mut self, value: &'static str) -> Self {
        self.mode = value;
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
    ) -> Result<CandleStrengthOscillatorBatchOutput, CandleStrengthOscillatorError> {
        candle_strength_oscillator_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
            &CandleStrengthOscillatorParams {
                period: None,
                atr_enabled: Some(self.atr_enabled),
                atr_length: None,
                mode: Some(self.mode.to_string()),
            },
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
    ) -> Result<CandleStrengthOscillatorBatchOutput, CandleStrengthOscillatorError> {
        candle_strength_oscillator_batch_with_kernel(
            open,
            high,
            low,
            close,
            &self.range,
            &CandleStrengthOscillatorParams {
                period: None,
                atr_enabled: Some(self.atr_enabled),
                atr_length: None,
                mode: Some(self.mode.to_string()),
            },
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_axis(
    field: &'static str,
    range: (usize, usize, usize),
) -> Result<Vec<usize>, CandleStrengthOscillatorError> {
    let (start, end, step) = range;
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(CandleStrengthOscillatorError::InvalidRange {
            field,
            start,
            end,
            step,
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end {
            break;
        }
        let next = current.checked_add(step).ok_or_else(|| {
            CandleStrengthOscillatorError::InvalidInput {
                msg: "candle_strength_oscillator: range step overflow".to_string(),
            }
        })?;
        if next <= current {
            return Err(CandleStrengthOscillatorError::InvalidRange {
                field,
                start,
                end,
                step,
            });
        }
        current = next.min(end);
    }
    Ok(out)
}

fn expand_grid_checked(
    sweep: &CandleStrengthOscillatorBatchRange,
    fixed: &CandleStrengthOscillatorParams,
) -> Result<Vec<CandleStrengthOscillatorParams>, CandleStrengthOscillatorError> {
    let periods = expand_axis("period", sweep.period)?;
    let atr_lengths = expand_axis("atr_length", sweep.atr_length)?;
    let total = periods
        .len()
        .checked_mul(atr_lengths.len())
        .ok_or_else(|| CandleStrengthOscillatorError::InvalidInput {
            msg: "candle_strength_oscillator: parameter grid size overflow".to_string(),
        })?;
    let mut out = Vec::with_capacity(total);
    for &period in &periods {
        for &atr_length in &atr_lengths {
            out.push(CandleStrengthOscillatorParams {
                period: Some(period),
                atr_enabled: Some(fixed.atr_enabled.unwrap_or(false)),
                atr_length: Some(atr_length),
                mode: Some(
                    fixed
                        .mode
                        .clone()
                        .unwrap_or_else(|| DEFAULT_MODE.to_string()),
                ),
            });
        }
    }
    Ok(out)
}

pub fn expand_grid_candle_strength_oscillator(
    sweep: &CandleStrengthOscillatorBatchRange,
    fixed: &CandleStrengthOscillatorParams,
) -> Result<Vec<CandleStrengthOscillatorParams>, CandleStrengthOscillatorError> {
    expand_grid_checked(sweep, fixed)
}

pub fn candle_strength_oscillator_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CandleStrengthOscillatorBatchRange,
    fixed: &CandleStrengthOscillatorParams,
    kernel: Kernel,
) -> Result<CandleStrengthOscillatorBatchOutput, CandleStrengthOscillatorError> {
    candle_strength_oscillator_batch_inner(open, high, low, close, sweep, fixed, kernel, true)
}

pub fn candle_strength_oscillator_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CandleStrengthOscillatorBatchRange,
    fixed: &CandleStrengthOscillatorParams,
    kernel: Kernel,
) -> Result<CandleStrengthOscillatorBatchOutput, CandleStrengthOscillatorError> {
    candle_strength_oscillator_batch_inner(open, high, low, close, sweep, fixed, kernel, false)
}

pub fn candle_strength_oscillator_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CandleStrengthOscillatorBatchRange,
    fixed: &CandleStrengthOscillatorParams,
    kernel: Kernel,
) -> Result<CandleStrengthOscillatorBatchOutput, CandleStrengthOscillatorError> {
    candle_strength_oscillator_batch_inner(open, high, low, close, sweep, fixed, kernel, true)
}

fn candle_strength_oscillator_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CandleStrengthOscillatorBatchRange,
    fixed: &CandleStrengthOscillatorParams,
    kernel: Kernel,
    parallel: bool,
) -> Result<CandleStrengthOscillatorBatchOutput, CandleStrengthOscillatorError> {
    let combos = expand_grid_checked(sweep, fixed)?;
    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| CandleStrengthOscillatorError::InvalidInput {
                msg: "candle_strength_oscillator: rows*cols overflow in batch".to_string(),
            })?;

    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(CandleStrengthOscillatorError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(CandleStrengthOscillatorError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let valid = longest_valid_run(open, high, low, close);
    if valid == 0 {
        return Err(CandleStrengthOscillatorError::AllValuesNaN);
    }

    let mut strength_warmups = Vec::with_capacity(rows);
    let mut levels_warmups = Vec::with_capacity(rows);
    let mut max_needed = 0usize;
    for combo in &combos {
        let resolved = resolve_params(combo)?;
        strength_warmups.push(strength_warmup_prefix(
            resolved.period,
            resolved.atr_enabled,
            resolved.atr_length,
        ));
        levels_warmups.push(levels_warmup_prefix(
            resolved.period,
            resolved.atr_enabled,
            resolved.atr_length,
        ));
        max_needed = max_needed.max(levels_needed_bars(
            resolved.period,
            resolved.atr_enabled,
            resolved.atr_length,
        ));
    }
    if valid < max_needed {
        return Err(CandleStrengthOscillatorError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let mut strength_mu = make_uninit_matrix(rows, cols);
    let mut highs_mu = make_uninit_matrix(rows, cols);
    let mut lows_mu = make_uninit_matrix(rows, cols);
    let mut mid_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut strength_mu, cols, &strength_warmups);
    init_matrix_prefixes(&mut highs_mu, cols, &levels_warmups);
    init_matrix_prefixes(&mut lows_mu, cols, &levels_warmups);
    init_matrix_prefixes(&mut mid_mu, cols, &levels_warmups);

    let mut strength = unsafe {
        Vec::from_raw_parts(
            strength_mu.as_mut_ptr() as *mut f64,
            strength_mu.len(),
            strength_mu.capacity(),
        )
    };
    let mut highs = unsafe {
        Vec::from_raw_parts(
            highs_mu.as_mut_ptr() as *mut f64,
            highs_mu.len(),
            highs_mu.capacity(),
        )
    };
    let mut lows = unsafe {
        Vec::from_raw_parts(
            lows_mu.as_mut_ptr() as *mut f64,
            lows_mu.len(),
            lows_mu.capacity(),
        )
    };
    let mut mid = unsafe {
        Vec::from_raw_parts(
            mid_mu.as_mut_ptr() as *mut f64,
            mid_mu.len(),
            mid_mu.capacity(),
        )
    };
    std::mem::forget(strength_mu);
    std::mem::forget(highs_mu);
    std::mem::forget(lows_mu);
    std::mem::forget(mid_mu);

    let mut long_signal = vec![0.0; total];
    let mut short_signal = vec![0.0; total];

    candle_strength_oscillator_batch_inner_into(
        open,
        high,
        low,
        close,
        sweep,
        fixed,
        kernel,
        parallel,
        &mut strength,
        &mut highs,
        &mut lows,
        &mut mid,
        &mut long_signal,
        &mut short_signal,
    )?;

    Ok(CandleStrengthOscillatorBatchOutput {
        strength,
        highs,
        lows,
        mid,
        long_signal,
        short_signal,
        combos,
        rows,
        cols,
    })
}

#[allow(clippy::too_many_arguments)]
fn candle_strength_oscillator_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CandleStrengthOscillatorBatchRange,
    fixed: &CandleStrengthOscillatorParams,
    kernel: Kernel,
    parallel: bool,
    dst_strength: &mut [f64],
    dst_highs: &mut [f64],
    dst_lows: &mut [f64],
    dst_mid: &mut [f64],
    dst_long_signal: &mut [f64],
    dst_short_signal: &mut [f64],
) -> Result<Vec<CandleStrengthOscillatorParams>, CandleStrengthOscillatorError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(CandleStrengthOscillatorError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep, fixed)?;
    let len = close.len();
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(CandleStrengthOscillatorError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(CandleStrengthOscillatorError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let total = combos.len().checked_mul(len).ok_or_else(|| {
        CandleStrengthOscillatorError::InvalidInput {
            msg: "candle_strength_oscillator: rows*cols overflow in batch_into".to_string(),
        }
    })?;
    for dst in [
        &*dst_strength,
        &*dst_highs,
        &*dst_lows,
        &*dst_mid,
        &*dst_long_signal,
        &*dst_short_signal,
    ] {
        if dst.len() != total {
            return Err(CandleStrengthOscillatorError::MismatchedOutputLen {
                dst_len: dst.len(),
                expected_len: total,
            });
        }
    }

    let valid = longest_valid_run(open, high, low, close);
    if valid == 0 {
        return Err(CandleStrengthOscillatorError::AllValuesNaN);
    }
    let mut max_needed = 0usize;
    for combo in &combos {
        let resolved = resolve_params(combo)?;
        max_needed = max_needed.max(levels_needed_bars(
            resolved.period,
            resolved.atr_enabled,
            resolved.atr_length,
        ));
    }
    if valid < max_needed {
        return Err(CandleStrengthOscillatorError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize,
                  out_strength: &mut [f64],
                  out_highs: &mut [f64],
                  out_lows: &mut [f64],
                  out_mid: &mut [f64],
                  out_long_signal: &mut [f64],
                  out_short_signal: &mut [f64]| {
        let resolved = resolve_params(&combos[row]).expect("validated combos");
        compute_row(
            open,
            high,
            low,
            close,
            resolved,
            out_strength,
            out_highs,
            out_lows,
            out_mid,
            out_long_signal,
            out_short_signal,
        );
    };

    macro_rules! run_rows {
        ($iter:expr) => {
            for (
                row,
                (
                    ((((out_strength, out_highs), out_lows), out_mid), out_long_signal),
                    out_short_signal,
                ),
            ) in $iter.enumerate()
            {
                worker(
                    row,
                    out_strength,
                    out_highs,
                    out_lows,
                    out_mid,
                    out_long_signal,
                    out_short_signal,
                );
            }
        };
    }

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            dst_strength
                .par_chunks_mut(len)
                .zip(dst_highs.par_chunks_mut(len))
                .zip(dst_lows.par_chunks_mut(len))
                .zip(dst_mid.par_chunks_mut(len))
                .zip(dst_long_signal.par_chunks_mut(len))
                .zip(dst_short_signal.par_chunks_mut(len))
                .enumerate()
                .for_each(
                    |(
                        row,
                        (
                            ((((out_strength, out_highs), out_lows), out_mid), out_long_signal),
                            out_short_signal,
                        ),
                    )| {
                        worker(
                            row,
                            out_strength,
                            out_highs,
                            out_lows,
                            out_mid,
                            out_long_signal,
                            out_short_signal,
                        );
                    },
                );
        }
        #[cfg(target_arch = "wasm32")]
        {
            run_rows!(dst_strength
                .chunks_mut(len)
                .zip(dst_highs.chunks_mut(len))
                .zip(dst_lows.chunks_mut(len))
                .zip(dst_mid.chunks_mut(len))
                .zip(dst_long_signal.chunks_mut(len))
                .zip(dst_short_signal.chunks_mut(len)));
        }
    } else {
        run_rows!(dst_strength
            .chunks_mut(len)
            .zip(dst_highs.chunks_mut(len))
            .zip(dst_lows.chunks_mut(len))
            .zip(dst_mid.chunks_mut(len))
            .zip(dst_long_signal.chunks_mut(len))
            .zip(dst_short_signal.chunks_mut(len)));
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "candle_strength_oscillator",
    signature = (open, high, low, close, period=DEFAULT_PERIOD, atr_enabled=false, atr_length=DEFAULT_ATR_LENGTH, mode=DEFAULT_MODE, kernel=None)
)]
pub fn candle_strength_oscillator_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
    atr_enabled: bool,
    atr_length: usize,
    mode: &str,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = CandleStrengthOscillatorInput::from_slices(
        open,
        high,
        low,
        close,
        CandleStrengthOscillatorParams {
            period: Some(period),
            atr_enabled: Some(atr_enabled),
            atr_length: Some(atr_length),
            mode: Some(mode.to_string()),
        },
    );
    let out = py
        .allow_threads(|| candle_strength_oscillator_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.strength.into_pyarray(py),
        out.highs.into_pyarray(py),
        out.lows.into_pyarray(py),
        out.mid.into_pyarray(py),
        out.long_signal.into_pyarray(py),
        out.short_signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "CandleStrengthOscillatorStream")]
pub struct CandleStrengthOscillatorStreamPy {
    inner: CandleStrengthOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CandleStrengthOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (period=DEFAULT_PERIOD, atr_enabled=false, atr_length=DEFAULT_ATR_LENGTH, mode=DEFAULT_MODE))]
    fn new(period: usize, atr_enabled: bool, atr_length: usize, mode: &str) -> PyResult<Self> {
        let inner = CandleStrengthOscillatorStream::try_new(CandleStrengthOscillatorParams {
            period: Some(period),
            atr_enabled: Some(atr_enabled),
            atr_length: Some(atr_length),
            mode: Some(mode.to_string()),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.inner.update(open, high, low, close).map(|point| {
            (
                point.strength,
                point.highs,
                point.lows,
                point.mid,
                point.long_signal,
                point.short_signal,
            )
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "candle_strength_oscillator_batch",
    signature = (open, high, low, close, period_range=(DEFAULT_PERIOD, DEFAULT_PERIOD, 0), atr_enabled=false, atr_length_range=(DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0), mode=DEFAULT_MODE, kernel=None)
)]
pub fn candle_strength_oscillator_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    atr_enabled: bool,
    atr_length_range: (usize, usize, usize),
    mode: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            candle_strength_oscillator_batch_with_kernel(
                open,
                high,
                low,
                close,
                &CandleStrengthOscillatorBatchRange {
                    period: period_range,
                    atr_length: atr_length_range,
                },
                &CandleStrengthOscillatorParams {
                    period: None,
                    atr_enabled: Some(atr_enabled),
                    atr_length: None,
                    mode: Some(mode.to_string()),
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "strength",
        output
            .strength
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "highs",
        output
            .highs
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lows",
        output
            .lows
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "mid",
        output
            .mid
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
            .map(|combo| combo.period.unwrap_or(DEFAULT_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "atr_enableds",
        output
            .combos
            .iter()
            .map(|combo| combo.atr_enabled.unwrap_or(false))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "atr_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "modes",
        output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .mode
                    .clone()
                    .unwrap_or_else(|| DEFAULT_MODE.to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_candle_strength_oscillator_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(candle_strength_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(candle_strength_oscillator_batch_py, m)?)?;
    m.add_class::<CandleStrengthOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleStrengthOscillatorBatchConfig {
    pub period_range: Vec<usize>,
    pub atr_enabled: Option<bool>,
    pub atr_length_range: Vec<usize>,
    pub mode: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = candle_strength_oscillator_js)]
pub fn candle_strength_oscillator_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    atr_enabled: bool,
    atr_length: usize,
    mode: &str,
) -> Result<JsValue, JsValue> {
    let input = CandleStrengthOscillatorInput::from_slices(
        open,
        high,
        low,
        close,
        CandleStrengthOscillatorParams {
            period: Some(period),
            atr_enabled: Some(atr_enabled),
            atr_length: Some(atr_length),
            mode: Some(mode.to_string()),
        },
    );
    let out = candle_strength_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("strength"),
        &serde_wasm_bindgen::to_value(&out.strength).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("highs"),
        &serde_wasm_bindgen::to_value(&out.highs).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("lows"),
        &serde_wasm_bindgen::to_value(&out.lows).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("mid"),
        &serde_wasm_bindgen::to_value(&out.mid).unwrap(),
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
#[wasm_bindgen(js_name = candle_strength_oscillator_batch_js)]
pub fn candle_strength_oscillator_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: CandleStrengthOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.period_range.len() != 3 || config.atr_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = candle_strength_oscillator_batch_with_kernel(
        open,
        high,
        low,
        close,
        &CandleStrengthOscillatorBatchRange {
            period: (
                config.period_range[0],
                config.period_range[1],
                config.period_range[2],
            ),
            atr_length: (
                config.atr_length_range[0],
                config.atr_length_range[1],
                config.atr_length_range[2],
            ),
        },
        &CandleStrengthOscillatorParams {
            period: None,
            atr_enabled: Some(config.atr_enabled.unwrap_or(false)),
            atr_length: None,
            mode: Some(config.mode.unwrap_or_else(|| DEFAULT_MODE.to_string())),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("strength"),
        &serde_wasm_bindgen::to_value(&out.strength).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("highs"),
        &serde_wasm_bindgen::to_value(&out.highs).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("lows"),
        &serde_wasm_bindgen::to_value(&out.lows).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("mid"),
        &serde_wasm_bindgen::to_value(&out.mid).unwrap(),
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
pub fn candle_strength_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(6 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn candle_strength_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 6 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn candle_strength_oscillator_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    atr_enabled: bool,
    atr_length: usize,
    mode: &str,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to candle_strength_oscillator_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 6 * len);
        let (dst_strength, rest) = out.split_at_mut(len);
        let (dst_highs, rest) = rest.split_at_mut(len);
        let (dst_lows, rest) = rest.split_at_mut(len);
        let (dst_mid, rest) = rest.split_at_mut(len);
        let (dst_long_signal, dst_short_signal) = rest.split_at_mut(len);
        let input = CandleStrengthOscillatorInput::from_slices(
            open,
            high,
            low,
            close,
            CandleStrengthOscillatorParams {
                period: Some(period),
                atr_enabled: Some(atr_enabled),
                atr_length: Some(atr_length),
                mode: Some(mode.to_string()),
            },
        );
        candle_strength_oscillator_into_slice(
            dst_strength,
            dst_highs,
            dst_lows,
            dst_mid,
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
pub fn candle_strength_oscillator_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    atr_enabled: bool,
    atr_length_start: usize,
    atr_length_end: usize,
    atr_length_step: usize,
    mode: &str,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to candle_strength_oscillator_batch_into",
        ));
    }
    let sweep = CandleStrengthOscillatorBatchRange {
        period: (period_start, period_end, period_step),
        atr_length: (atr_length_start, atr_length_end, atr_length_step),
    };
    let fixed = CandleStrengthOscillatorParams {
        period: None,
        atr_enabled: Some(atr_enabled),
        atr_length: None,
        mode: Some(mode.to_string()),
    };
    let combos =
        expand_grid_checked(&sweep, &fixed).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|value| value.checked_mul(6))
        .ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in candle_strength_oscillator_batch_into")
        })?;
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_strength, rest) = out.split_at_mut(split);
        let (dst_highs, rest) = rest.split_at_mut(split);
        let (dst_lows, rest) = rest.split_at_mut(split);
        let (dst_mid, rest) = rest.split_at_mut(split);
        let (dst_long_signal, dst_short_signal) = rest.split_at_mut(split);
        candle_strength_oscillator_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            &fixed,
            Kernel::Auto,
            false,
            dst_strength,
            dst_highs,
            dst_lows,
            dst_mid,
            dst_long_signal,
            dst_short_signal,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn candle_strength_oscillator_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    atr_enabled: bool,
    atr_length: usize,
    mode: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = candle_strength_oscillator_js(
        open,
        high,
        low,
        close,
        period,
        atr_enabled,
        atr_length,
        mode,
    )?;
    crate::write_wasm_object_f64_outputs("candle_strength_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn candle_strength_oscillator_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = candle_strength_oscillator_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "candle_strength_oscillator_batch_output_into_js",
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

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.09 + (x * 0.13).sin() * 1.4 + (x * 0.031).cos() * 0.6
            })
            .collect();
        let open: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c - ((i as f64) * 0.19).sin() * 0.7)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.4 + ((i as f64) * 0.07).cos().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.4 - ((i as f64) * 0.09).sin().abs() * 0.2)
            .collect();
        (open, high, low, close)
    }

    fn naive_wma(data: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        if period <= 1 {
            for (i, &value) in data.iter().enumerate() {
                if value.is_finite() {
                    out[i] = value;
                }
            }
            return out;
        }
        let denom = (period * (period + 1) / 2) as f64;
        let mut window = VecDeque::with_capacity(period);
        for (i, &value) in data.iter().enumerate() {
            if !value.is_finite() {
                window.clear();
                continue;
            }
            if window.len() == period {
                window.pop_front();
            }
            window.push_back(value);
            if window.len() == period {
                let mut sum = 0.0;
                for (j, &x) in window.iter().enumerate() {
                    sum += x * (j + 1) as f64;
                }
                out[i] = sum / denom;
            }
        }
        out
    }

    fn naive_hma(data: &[f64], period: usize) -> Vec<f64> {
        let half = (period / 2).max(1);
        let sqrt = sqrt_period(period).max(1);
        let full = naive_wma(data, period.max(1));
        let half = naive_wma(data, half);
        let mut delta = vec![f64::NAN; data.len()];
        for i in 0..data.len() {
            if half[i].is_finite() && full[i].is_finite() {
                delta[i] = 2.0f64.mul_add(half[i], -full[i]);
            }
        }
        naive_wma(&delta, sqrt)
    }

    fn naive_atr(high: &[f64], low: &[f64], close: &[f64], length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; high.len()];
        let alpha = 1.0 / length as f64;
        let mut prev_close = f64::NAN;
        let mut warm_sum = 0.0;
        let mut warm_count = 0usize;
        let mut rma = f64::NAN;
        let mut seeded = false;
        for i in 0..high.len() {
            if !high[i].is_finite() || !low[i].is_finite() || !close[i].is_finite() {
                prev_close = f64::NAN;
                warm_sum = 0.0;
                warm_count = 0;
                rma = f64::NAN;
                seeded = false;
                continue;
            }
            let tr = if prev_close.is_nan() {
                high[i] - low[i]
            } else {
                let up = if high[i] > prev_close {
                    high[i]
                } else {
                    prev_close
                };
                let dn = if low[i] < prev_close {
                    low[i]
                } else {
                    prev_close
                };
                up - dn
            };
            prev_close = close[i];
            if !seeded {
                warm_sum += tr;
                warm_count += 1;
                if warm_count == length {
                    rma = warm_sum * alpha;
                    seeded = true;
                    out[i] = rma;
                }
            } else {
                rma = alpha.mul_add(tr - rma, rma);
                out[i] = rma;
            }
        }
        out
    }

    fn naive_levels(
        strength: &[f64],
        period: usize,
        mode: CandleStrengthOscillatorMode,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut highs = vec![f64::NAN; strength.len()];
        let mut lows = vec![f64::NAN; strength.len()];
        let mut mid = vec![f64::NAN; strength.len()];
        let mut window = VecDeque::with_capacity(period.max(1));
        for (i, &value) in strength.iter().enumerate() {
            if !value.is_finite() {
                window.clear();
                continue;
            }
            if window.len() == period {
                window.pop_front();
            }
            window.push_back(value);
            if window.len() < period {
                continue;
            }
            match mode {
                CandleStrengthOscillatorMode::Bollinger => {
                    let mean = window.iter().sum::<f64>() / period as f64;
                    let var = window
                        .iter()
                        .map(|&x| {
                            let d = x - mean;
                            d * d
                        })
                        .sum::<f64>()
                        / period as f64;
                    let std = if var > 0.0 { var.sqrt() } else { 0.0 };
                    highs[i] = mean + BOLLINGER_STD_MULTIPLIER * std;
                    lows[i] = mean - BOLLINGER_STD_MULTIPLIER * std;
                    mid[i] = mean;
                }
                CandleStrengthOscillatorMode::Donchian => {
                    let mut hi = f64::NEG_INFINITY;
                    let mut lo = f64::INFINITY;
                    for &x in &window {
                        hi = hi.max(x);
                        lo = lo.min(x);
                    }
                    highs[i] = hi;
                    lows[i] = lo;
                    mid[i] = 0.5 * (hi + lo);
                }
            }
        }
        (highs, lows, mid)
    }

    fn naive_candle_strength(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        period: usize,
        atr_enabled: bool,
        atr_length: usize,
        mode: CandleStrengthOscillatorMode,
    ) -> CandleStrengthOscillatorOutput {
        let atr = if atr_enabled {
            naive_atr(high, low, close, atr_length)
        } else {
            vec![1.0; close.len()]
        };
        let mut score = vec![f64::NAN; close.len()];
        for i in 0..close.len() {
            let range = high[i] - low[i];
            if !is_valid_quad(open[i], high[i], low[i], close[i])
                || !atr[i].is_finite()
                || range.abs() <= f64::EPSILON
            {
                continue;
            }
            let body = (close[i] - open[i]).abs();
            let sign = if close[i] > open[i] { 1.0 } else { -1.0 };
            score[i] = sign * body / range * atr[i] * 100.0;
        }
        let strength = naive_hma(&score, period);
        let (highs, lows, mid) = naive_levels(&strength, period, mode);
        let mut long_signal = vec![0.0; close.len()];
        let mut short_signal = vec![0.0; close.len()];
        for i in 1..close.len() {
            if strength[i].is_finite()
                && mid[i].is_finite()
                && strength[i - 1].is_finite()
                && mid[i - 1].is_finite()
            {
                if strength[i - 1] <= mid[i - 1] && strength[i] > mid[i] {
                    long_signal[i] = 1.0;
                }
                if strength[i - 1] >= mid[i - 1] && strength[i] < mid[i] {
                    short_signal[i] = 1.0;
                }
            }
        }
        CandleStrengthOscillatorOutput {
            strength,
            highs,
            lows,
            mid,
            long_signal,
            short_signal,
        }
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
    fn candle_strength_oscillator_matches_naive_bollinger() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(256);
        let input = CandleStrengthOscillatorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            CandleStrengthOscillatorParams {
                period: Some(30),
                atr_enabled: Some(false),
                atr_length: Some(20),
                mode: Some("bollinger".to_string()),
            },
        );
        let out = candle_strength_oscillator_with_kernel(&input, Kernel::Scalar)?;
        let expected = naive_candle_strength(
            &open,
            &high,
            &low,
            &close,
            30,
            false,
            20,
            CandleStrengthOscillatorMode::Bollinger,
        );
        assert_series_close(&out.strength, &expected.strength, 1e-10);
        assert_series_close(&out.highs, &expected.highs, 1e-10);
        assert_series_close(&out.lows, &expected.lows, 1e-10);
        assert_series_close(&out.mid, &expected.mid, 1e-10);
        assert_series_close(&out.long_signal, &expected.long_signal, 1e-10);
        assert_series_close(&out.short_signal, &expected.short_signal, 1e-10);
        Ok(())
    }

    #[test]
    fn candle_strength_oscillator_matches_naive_donchian_with_atr() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(256);
        let input = CandleStrengthOscillatorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            CandleStrengthOscillatorParams {
                period: Some(20),
                atr_enabled: Some(true),
                atr_length: Some(14),
                mode: Some("donchian".to_string()),
            },
        );
        let out = candle_strength_oscillator_with_kernel(&input, Kernel::Scalar)?;
        let expected = naive_candle_strength(
            &open,
            &high,
            &low,
            &close,
            20,
            true,
            14,
            CandleStrengthOscillatorMode::Donchian,
        );
        assert_series_close(&out.strength, &expected.strength, 1e-10);
        assert_series_close(&out.highs, &expected.highs, 1e-10);
        assert_series_close(&out.lows, &expected.lows, 1e-10);
        assert_series_close(&out.mid, &expected.mid, 1e-10);
        assert_series_close(&out.long_signal, &expected.long_signal, 1e-10);
        assert_series_close(&out.short_signal, &expected.short_signal, 1e-10);
        Ok(())
    }

    #[test]
    fn candle_strength_oscillator_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(220);
        let input = CandleStrengthOscillatorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            CandleStrengthOscillatorParams {
                period: Some(24),
                atr_enabled: Some(true),
                atr_length: Some(10),
                mode: Some("bollinger".to_string()),
            },
        );
        let batch = candle_strength_oscillator(&input)?;
        let mut stream = CandleStrengthOscillatorStream::try_new(input.params.clone())?;
        let mut strength = Vec::with_capacity(close.len());
        let mut highs = Vec::with_capacity(close.len());
        let mut lows = Vec::with_capacity(close.len());
        let mut mid = Vec::with_capacity(close.len());
        let mut long_signal = Vec::with_capacity(close.len());
        let mut short_signal = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            if let Some(point) = stream.update(open[i], high[i], low[i], close[i]) {
                strength.push(point.strength);
                highs.push(point.highs);
                lows.push(point.lows);
                mid.push(point.mid);
                long_signal.push(point.long_signal);
                short_signal.push(point.short_signal);
            } else {
                strength.push(f64::NAN);
                highs.push(f64::NAN);
                lows.push(f64::NAN);
                mid.push(f64::NAN);
                long_signal.push(0.0);
                short_signal.push(0.0);
            }
        }
        assert_series_close(&strength, &batch.strength, 1e-12);
        assert_series_close(&highs, &batch.highs, 1e-12);
        assert_series_close(&lows, &batch.lows, 1e-12);
        assert_series_close(&mid, &batch.mid, 1e-12);
        assert_series_close(&long_signal, &batch.long_signal, 1e-12);
        assert_series_close(&short_signal, &batch.short_signal, 1e-12);
        Ok(())
    }

    #[test]
    fn candle_strength_oscillator_into_slice_matches_direct() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(180);
        let input = CandleStrengthOscillatorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            CandleStrengthOscillatorParams {
                period: Some(18),
                atr_enabled: Some(false),
                atr_length: Some(12),
                mode: Some("donchian".to_string()),
            },
        );
        let baseline = candle_strength_oscillator(&input)?;
        let mut strength = vec![f64::NAN; close.len()];
        let mut highs = vec![f64::NAN; close.len()];
        let mut lows = vec![f64::NAN; close.len()];
        let mut mid = vec![f64::NAN; close.len()];
        let mut long_signal = vec![0.0; close.len()];
        let mut short_signal = vec![0.0; close.len()];
        candle_strength_oscillator_into_slice(
            &mut strength,
            &mut highs,
            &mut lows,
            &mut mid,
            &mut long_signal,
            &mut short_signal,
            &input,
            Kernel::Scalar,
        )?;
        assert_series_close(&strength, &baseline.strength, 1e-12);
        assert_series_close(&highs, &baseline.highs, 1e-12);
        assert_series_close(&lows, &baseline.lows, 1e-12);
        assert_series_close(&mid, &baseline.mid, 1e-12);
        assert_series_close(&long_signal, &baseline.long_signal, 1e-12);
        assert_series_close(&short_signal, &baseline.short_signal, 1e-12);
        Ok(())
    }

    #[test]
    fn candle_strength_oscillator_batch_and_dispatch_outputs() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(192);
        let batch = candle_strength_oscillator_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &CandleStrengthOscillatorBatchRange {
                period: (20, 24, 2),
                atr_length: (14, 14, 0),
            },
            &CandleStrengthOscillatorParams {
                period: None,
                atr_enabled: Some(true),
                atr_length: None,
                mode: Some("bollinger".to_string()),
            },
            Kernel::Scalar,
        )?;
        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, close.len());
        let params0 = [
            ParamKV {
                key: "period",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "atr_enabled",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "atr_length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "mode",
                value: ParamValue::EnumString("bollinger"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params0 }];
        let dispatch = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "candle_strength_oscillator",
            output_id: Some("mid"),
            combos: &combos,
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(dispatch.rows, 1);
        assert_eq!(dispatch.cols, close.len());
        let values = dispatch.values_f64.as_ref().expect("f64 output");
        assert_series_close(values, &batch.mid[..close.len()], 1e-12);
        Ok(())
    }
}
