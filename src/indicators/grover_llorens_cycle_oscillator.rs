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

use crate::indicators::rsi::{RsiParams, RsiStream};
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 100;
const DEFAULT_MULT: f64 = 10.0;
const DEFAULT_SOURCE: &str = "close";
const DEFAULT_SMOOTH: bool = true;
const DEFAULT_RSI_PERIOD: usize = 20;
const FLOAT_TOL: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Open,
    High,
    Low,
    Close,
    Hl2,
    Hlc3,
    Ohlc4,
    Hlcc4,
}

impl SourceKind {
    #[inline]
    fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("open") {
            Some(Self::Open)
        } else if value.eq_ignore_ascii_case("high") {
            Some(Self::High)
        } else if value.eq_ignore_ascii_case("low") {
            Some(Self::Low)
        } else if value.eq_ignore_ascii_case("close") {
            Some(Self::Close)
        } else if value.eq_ignore_ascii_case("hl2") {
            Some(Self::Hl2)
        } else if value.eq_ignore_ascii_case("hlc3") {
            Some(Self::Hlc3)
        } else if value.eq_ignore_ascii_case("ohlc4") {
            Some(Self::Ohlc4)
        } else if value.eq_ignore_ascii_case("hlcc4") || value.eq_ignore_ascii_case("hlcc") {
            Some(Self::Hlcc4)
        } else {
            None
        }
    }

    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::High => "high",
            Self::Low => "low",
            Self::Close => "close",
            Self::Hl2 => "hl2",
            Self::Hlc3 => "hlc3",
            Self::Ohlc4 => "ohlc4",
            Self::Hlcc4 => "hlcc4",
        }
    }

    #[inline]
    fn needs_open(self) -> bool {
        matches!(self, Self::Open | Self::Ohlc4)
    }

    #[inline]
    fn value(self, open: f64, high: f64, low: f64, close: f64) -> f64 {
        match self {
            Self::Open => open,
            Self::High => high,
            Self::Low => low,
            Self::Close => close,
            Self::Hl2 => 0.5 * (high + low),
            Self::Hlc3 => (high + low + close) / 3.0,
            Self::Ohlc4 => (open + high + low + close) * 0.25,
            Self::Hlcc4 => (high + low + close + close) * 0.25,
        }
    }
}

#[derive(Debug, Clone)]
pub enum GroverLlorensCycleOscillatorData<'a> {
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
pub struct GroverLlorensCycleOscillatorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct GroverLlorensCycleOscillatorParams {
    pub length: Option<usize>,
    pub mult: Option<f64>,
    pub source: Option<String>,
    pub smooth: Option<bool>,
    pub rsi_period: Option<usize>,
}

impl Default for GroverLlorensCycleOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            mult: Some(DEFAULT_MULT),
            source: Some(DEFAULT_SOURCE.to_string()),
            smooth: Some(DEFAULT_SMOOTH),
            rsi_period: Some(DEFAULT_RSI_PERIOD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GroverLlorensCycleOscillatorInput<'a> {
    pub data: GroverLlorensCycleOscillatorData<'a>,
    pub params: GroverLlorensCycleOscillatorParams,
}

impl<'a> GroverLlorensCycleOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: GroverLlorensCycleOscillatorParams) -> Self {
        Self {
            data: GroverLlorensCycleOscillatorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: GroverLlorensCycleOscillatorParams,
    ) -> Self {
        Self {
            data: GroverLlorensCycleOscillatorData::Slices {
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
        Self::from_candles(candles, GroverLlorensCycleOscillatorParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(DEFAULT_MULT)
    }

    #[inline]
    pub fn get_source(&self) -> &str {
        self.params.source.as_deref().unwrap_or(DEFAULT_SOURCE)
    }

    #[inline]
    pub fn get_smooth(&self) -> bool {
        self.params.smooth.unwrap_or(DEFAULT_SMOOTH)
    }

    #[inline]
    pub fn get_rsi_period(&self) -> usize {
        self.params.rsi_period.unwrap_or(DEFAULT_RSI_PERIOD)
    }
}

#[derive(Clone, Debug)]
pub struct GroverLlorensCycleOscillatorBuilder {
    length: Option<usize>,
    mult: Option<f64>,
    source: Option<SourceKind>,
    smooth: Option<bool>,
    rsi_period: Option<usize>,
    kernel: Kernel,
}

impl Default for GroverLlorensCycleOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            mult: None,
            source: None,
            smooth: None,
            rsi_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl GroverLlorensCycleOscillatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline]
    pub fn mult(mut self, mult: f64) -> Self {
        self.mult = Some(mult);
        self
    }

    #[inline]
    pub fn source(mut self, source: &str) -> Result<Self, GroverLlorensCycleOscillatorError> {
        self.source = Some(parse_source(source)?);
        Ok(self)
    }

    #[inline]
    pub fn smooth(mut self, smooth: bool) -> Self {
        self.smooth = Some(smooth);
        self
    }

    #[inline]
    pub fn rsi_period(mut self, rsi_period: usize) -> Self {
        self.rsi_period = Some(rsi_period);
        self
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
    ) -> Result<GroverLlorensCycleOscillatorOutput, GroverLlorensCycleOscillatorError> {
        let input = GroverLlorensCycleOscillatorInput::from_candles(
            candles,
            GroverLlorensCycleOscillatorParams {
                length: self.length,
                mult: self.mult,
                source: Some(
                    self.source
                        .unwrap_or(SourceKind::Close)
                        .as_str()
                        .to_string(),
                ),
                smooth: self.smooth,
                rsi_period: self.rsi_period,
            },
        );
        grover_llorens_cycle_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<GroverLlorensCycleOscillatorOutput, GroverLlorensCycleOscillatorError> {
        let input = GroverLlorensCycleOscillatorInput::from_slices(
            open,
            high,
            low,
            close,
            GroverLlorensCycleOscillatorParams {
                length: self.length,
                mult: self.mult,
                source: Some(
                    self.source
                        .unwrap_or(SourceKind::Close)
                        .as_str()
                        .to_string(),
                ),
                smooth: self.smooth,
                rsi_period: self.rsi_period,
            },
        );
        grover_llorens_cycle_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<GroverLlorensCycleOscillatorStream, GroverLlorensCycleOscillatorError> {
        GroverLlorensCycleOscillatorStream::try_new(GroverLlorensCycleOscillatorParams {
            length: self.length,
            mult: self.mult,
            source: Some(
                self.source
                    .unwrap_or(SourceKind::Close)
                    .as_str()
                    .to_string(),
            ),
            smooth: self.smooth,
            rsi_period: self.rsi_period,
        })
    }
}

#[derive(Debug, Error)]
pub enum GroverLlorensCycleOscillatorError {
    #[error("grover_llorens_cycle_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("grover_llorens_cycle_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "grover_llorens_cycle_oscillator: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}"
    )]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "grover_llorens_cycle_oscillator: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error("grover_llorens_cycle_oscillator: Invalid mult: {mult}")]
    InvalidMult { mult: f64 },
    #[error(
        "grover_llorens_cycle_oscillator: Invalid source: {source_name}. Supported: open, high, low, close, hl2, hlc3, ohlc4, hlcc4"
    )]
    InvalidSource { source_name: String },
    #[error("grover_llorens_cycle_oscillator: Invalid rsi_period: {rsi_period}")]
    InvalidRsiPeriod { rsi_period: usize },
    #[error(
        "grover_llorens_cycle_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "grover_llorens_cycle_oscillator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "grover_llorens_cycle_oscillator: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("grover_llorens_cycle_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    length: usize,
    mult: f64,
    source: SourceKind,
    smooth: bool,
    rsi_period: usize,
    ema_alpha: f64,
}

#[derive(Debug, Clone)]
pub struct GroverLlorensCycleOscillatorStream {
    params: ResolvedParams,
    rsi_stream: RsiStream,
    max_prev: VecDeque<(usize, f64)>,
    min_prev: VecDeque<(usize, f64)>,
    valid_src_count: usize,
    prev_src: Option<f64>,
    prev_close: Option<f64>,
    atr_seed_count: usize,
    atr_seed_sum: f64,
    atr_value: Option<f64>,
    os: i8,
    prev_diff: Option<f64>,
    prev_ts: Option<f64>,
    last_event_step: Option<f64>,
    bars_since_event: usize,
    ema_value: Option<f64>,
}

impl GroverLlorensCycleOscillatorStream {
    pub fn try_new(
        params: GroverLlorensCycleOscillatorParams,
    ) -> Result<Self, GroverLlorensCycleOscillatorError> {
        let params = resolve_params(&params, 0)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        Self {
            params,
            rsi_stream: RsiStream::try_new(RsiParams {
                period: Some(params.rsi_period),
            })
            .expect("resolved RSI params must be valid"),
            max_prev: VecDeque::with_capacity(params.length),
            min_prev: VecDeque::with_capacity(params.length),
            valid_src_count: 0,
            prev_src: None,
            prev_close: None,
            atr_seed_count: 0,
            atr_seed_sum: 0.0,
            atr_value: None,
            os: 0,
            prev_diff: None,
            prev_ts: None,
            last_event_step: None,
            bars_since_event: 0,
            ema_value: None,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new_resolved(self.params);
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.params
            .length
            .max(self.params.rsi_period)
            .saturating_sub(1)
    }

    #[inline]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<f64> {
        if !valid_bar(self.params.source, open, high, low, close) {
            self.reset();
            return None;
        }

        let src = self.params.source.value(open, high, low, close);
        self.update_source(src, high, low, close)
    }

    #[inline]
    fn update_close(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if !(high.is_finite() && low.is_finite() && close.is_finite()) {
            self.reset();
            return None;
        }

        self.update_source(close, high, low, close)
    }

    #[inline]
    fn update_source(&mut self, src: f64, high: f64, low: f64, close: f64) -> Option<f64> {
        let diff = match self.prev_src {
            Some(prev_src) => src - self.prev_ts.unwrap_or(prev_src),
            None => f64::NAN,
        };

        let atr = self.update_atr(high, low, close);

        let rising = if self.valid_src_count >= self.params.length {
            self.max_prev
                .front()
                .map(|(_, v)| src > *v + FLOAT_TOL)
                .unwrap_or(false)
        } else {
            false
        };
        let falling = if self.valid_src_count >= self.params.length {
            self.min_prev
                .front()
                .map(|(_, v)| src < *v - FLOAT_TOL)
                .unwrap_or(false)
        } else {
            false
        };

        let prev_os = self.os;
        let new_os = if rising {
            1
        } else if falling {
            -1
        } else {
            prev_os
        };

        let prev_diff = self.prev_diff.unwrap_or(f64::NAN);
        let rise = new_os - prev_os == 2 && prev_diff.is_finite() && prev_diff < 0.0;
        let fall = new_os - prev_os == -2 && prev_diff.is_finite() && prev_diff > 0.0;
        let up = prev_diff.is_finite() && prev_diff <= 0.0 && diff.is_finite() && diff > 0.0;
        let dn = prev_diff.is_finite() && prev_diff >= 0.0 && diff.is_finite() && diff < 0.0;
        let event = up || dn || rise || fall;

        let mut ts = f64::NAN;
        if let Some(atr_value) = atr {
            let step = atr_value / self.params.length as f64;
            if event {
                self.last_event_step = Some(step);
                self.bars_since_event = 0;
            } else if self.last_event_step.is_some() {
                self.bars_since_event = self.bars_since_event.saturating_add(1);
            }

            let prev_ts_or_src = self.prev_ts.unwrap_or(src);
            if up {
                ts = prev_ts_or_src - atr_value * self.params.mult;
            } else if dn {
                ts = prev_ts_or_src + atr_value * self.params.mult;
            } else if rise {
                ts = src - atr_value * self.params.mult;
            } else if fall {
                ts = src + atr_value * self.params.mult;
            } else if let Some(event_step) = self.last_event_step {
                ts = prev_ts_or_src
                    + signum_with_tol(diff) * event_step * self.bars_since_event as f64;
            }
        }

        let result = if ts.is_finite() {
            let osc = src - ts;
            let smoothed = if self.params.smooth {
                let next = match self.ema_value {
                    Some(prev) => self
                        .params
                        .ema_alpha
                        .mul_add(osc, (1.0 - self.params.ema_alpha) * prev),
                    None => osc,
                };
                self.ema_value = Some(next);
                next
            } else {
                osc
            };
            self.rsi_stream.update(smoothed)
        } else {
            None
        };

        self.prev_src = Some(src);
        self.prev_diff = if diff.is_finite() { Some(diff) } else { None };
        self.prev_ts = if ts.is_finite() { Some(ts) } else { None };
        self.os = new_os;
        self.push_source(src);

        result
    }

    #[inline]
    fn update_atr(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = match self.prev_close {
            Some(prev_close) => {
                let hl = high - low;
                let hc = (high - prev_close).abs();
                let lc = (low - prev_close).abs();
                hl.max(hc).max(lc)
            }
            None => high - low,
        };
        self.prev_close = Some(close);

        if self.atr_seed_count < self.params.length {
            self.atr_seed_count += 1;
            self.atr_seed_sum += tr;
            if self.atr_seed_count == self.params.length {
                self.atr_value = Some(self.atr_seed_sum / self.params.length as f64);
            }
        } else if let Some(prev_atr) = self.atr_value {
            let next =
                ((self.params.length - 1) as f64 * prev_atr + tr) / self.params.length as f64;
            self.atr_value = Some(next);
        }

        self.atr_value
    }

    #[inline]
    fn push_source(&mut self, src: f64) {
        let idx = self.valid_src_count;
        while let Some(&(_, back)) = self.max_prev.back() {
            if back <= src {
                self.max_prev.pop_back();
            } else {
                break;
            }
        }
        self.max_prev.push_back((idx, src));

        while let Some(&(_, back)) = self.min_prev.back() {
            if back >= src {
                self.min_prev.pop_back();
            } else {
                break;
            }
        }
        self.min_prev.push_back((idx, src));

        self.valid_src_count += 1;
        let window_start = self.valid_src_count.saturating_sub(self.params.length);
        while let Some(&(front_idx, _)) = self.max_prev.front() {
            if front_idx < window_start {
                self.max_prev.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(front_idx, _)) = self.min_prev.front() {
            if front_idx < window_start {
                self.min_prev.pop_front();
            } else {
                break;
            }
        }
    }
}

#[inline]
pub fn grover_llorens_cycle_oscillator(
    input: &GroverLlorensCycleOscillatorInput,
) -> Result<GroverLlorensCycleOscillatorOutput, GroverLlorensCycleOscillatorError> {
    grover_llorens_cycle_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn signum_with_tol(value: f64) -> f64 {
    if !value.is_finite() || value.abs() <= FLOAT_TOL {
        0.0
    } else {
        value.signum()
    }
}

#[inline(always)]
fn valid_bar(source: SourceKind, open: f64, high: f64, low: f64, close: f64) -> bool {
    if !(high.is_finite() && low.is_finite() && close.is_finite()) {
        return false;
    }
    if source.needs_open() && !open.is_finite() {
        return false;
    }
    source.value(open, high, low, close).is_finite()
}

#[inline(always)]
fn scan_valid_bars(
    source: SourceKind,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> (usize, usize) {
    let len = close.len();
    let mut first = len;
    let mut count = 0usize;
    for i in 0..len {
        if valid_bar(source, open[i], high[i], low[i], close[i]) {
            if first == len {
                first = i;
            }
            count += 1;
        }
    }
    (first, count)
}

#[inline(always)]
fn resolve_params(
    params: &GroverLlorensCycleOscillatorParams,
    data_len: usize,
) -> Result<ResolvedParams, GroverLlorensCycleOscillatorError> {
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    let mult = params.mult.unwrap_or(DEFAULT_MULT);
    let source_str = params.source.as_deref().unwrap_or(DEFAULT_SOURCE);
    let source = parse_source(source_str)?;
    let smooth = params.smooth.unwrap_or(DEFAULT_SMOOTH);
    let rsi_period = params.rsi_period.unwrap_or(DEFAULT_RSI_PERIOD);

    if length == 0 || (data_len != 0 && length > data_len) {
        return Err(GroverLlorensCycleOscillatorError::InvalidLength { length, data_len });
    }
    if !mult.is_finite() {
        return Err(GroverLlorensCycleOscillatorError::InvalidMult { mult });
    }
    if rsi_period == 0 {
        return Err(GroverLlorensCycleOscillatorError::InvalidRsiPeriod { rsi_period });
    }

    Ok(ResolvedParams {
        length,
        mult,
        source,
        smooth,
        rsi_period,
        ema_alpha: 2.0 / (rsi_period as f64 + 1.0),
    })
}

#[inline(always)]
fn parse_source(source: &str) -> Result<SourceKind, GroverLlorensCycleOscillatorError> {
    SourceKind::parse(source).ok_or_else(|| GroverLlorensCycleOscillatorError::InvalidSource {
        source_name: source.to_string(),
    })
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a GroverLlorensCycleOscillatorInput,
) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
    match &input.data {
        GroverLlorensCycleOscillatorData::Candles { candles } => {
            (&candles.open, &candles.high, &candles.low, &candles.close)
        }
        GroverLlorensCycleOscillatorData::Slices {
            open,
            high,
            low,
            close,
        } => (*open, *high, *low, *close),
    }
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a GroverLlorensCycleOscillatorInput,
    _kernel: Kernel,
) -> Result<
    (&'a [f64], &'a [f64], &'a [f64], &'a [f64], ResolvedParams),
    GroverLlorensCycleOscillatorError,
> {
    let (open, high, low, close) = input_slices(input);
    let len = close.len();
    if len == 0 {
        return Err(GroverLlorensCycleOscillatorError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(
            GroverLlorensCycleOscillatorError::InconsistentSliceLengths {
                open_len: open.len(),
                high_len: high.len(),
                low_len: low.len(),
                close_len: len,
            },
        );
    }

    let params = resolve_params(&input.params, len)?;
    let (first, valid) = scan_valid_bars(params.source, open, high, low, close);
    if first >= len {
        return Err(GroverLlorensCycleOscillatorError::AllValuesNaN);
    }

    let needed = params.length.max(params.rsi_period);
    if valid < needed {
        return Err(GroverLlorensCycleOscillatorError::NotEnoughValidData { needed, valid });
    }

    Ok((open, high, low, close, params))
}

#[inline(always)]
fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: ResolvedParams,
    out: &mut [f64],
) {
    if params.length == DEFAULT_LENGTH
        && params.mult == DEFAULT_MULT
        && matches!(params.source, SourceKind::Close)
        && params.smooth == DEFAULT_SMOOTH
        && params.rsi_period == DEFAULT_RSI_PERIOD
    {
        compute_row_default_close(high, low, close, out);
        return;
    }

    let mut stream = GroverLlorensCycleOscillatorStream::new_resolved(params);
    if matches!(params.source, SourceKind::Close) {
        for i in 0..close.len() {
            out[i] = stream
                .update_close(high[i], low[i], close[i])
                .unwrap_or(f64::NAN);
        }
    } else {
        for i in 0..close.len() {
            out[i] = stream
                .update(open[i], high[i], low[i], close[i])
                .unwrap_or(f64::NAN);
        }
    }
}

#[inline(always)]
fn compute_row_default_close(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    const CAP: usize = 128;
    const MASK: usize = CAP - 1;
    let ema_alpha = 2.0 / (DEFAULT_RSI_PERIOD as f64 + 1.0);
    let ema_omalpha = 1.0 - ema_alpha;
    let mut rsi_stream = RsiStream::try_new(RsiParams {
        period: Some(DEFAULT_RSI_PERIOD),
    })
    .expect("default RSI params must be valid");

    let mut max_idx = [0usize; CAP];
    let mut min_idx = [0usize; CAP];
    let mut max_val = [0.0; CAP];
    let mut min_val = [0.0; CAP];
    let mut max_head = 0usize;
    let mut max_tail = 0usize;
    let mut min_head = 0usize;
    let mut min_tail = 0usize;
    let mut valid_src_count = 0usize;
    let mut prev_src: Option<f64> = None;
    let mut prev_close: Option<f64> = None;
    let mut atr_seed_count = 0usize;
    let mut atr_seed_sum = 0.0;
    let mut atr_value: Option<f64> = None;
    let mut os = 0i8;
    let mut prev_diff: Option<f64> = None;
    let mut prev_ts: Option<f64> = None;
    let mut last_event_step: Option<f64> = None;
    let mut bars_since_event = 0usize;
    let mut ema_value: Option<f64> = None;

    for i in 0..close.len() {
        let high_i = high[i];
        let low_i = low[i];
        let close_i = close[i];
        if !(high_i.is_finite() && low_i.is_finite() && close_i.is_finite()) {
            rsi_stream = RsiStream::try_new(RsiParams {
                period: Some(DEFAULT_RSI_PERIOD),
            })
            .expect("default RSI params must be valid");
            max_head = 0;
            max_tail = 0;
            min_head = 0;
            min_tail = 0;
            valid_src_count = 0;
            prev_src = None;
            prev_close = None;
            atr_seed_count = 0;
            atr_seed_sum = 0.0;
            atr_value = None;
            os = 0;
            prev_diff = None;
            prev_ts = None;
            last_event_step = None;
            bars_since_event = 0;
            ema_value = None;
            out[i] = f64::NAN;
            continue;
        }

        let src = close_i;
        let diff = match prev_src {
            Some(previous) => src - prev_ts.unwrap_or(previous),
            None => f64::NAN,
        };

        let tr = match prev_close {
            Some(previous_close) => {
                let hl = high_i - low_i;
                let hc = (high_i - previous_close).abs();
                let lc = (low_i - previous_close).abs();
                hl.max(hc).max(lc)
            }
            None => high_i - low_i,
        };
        prev_close = Some(close_i);

        if atr_seed_count < DEFAULT_LENGTH {
            atr_seed_count += 1;
            atr_seed_sum += tr;
            if atr_seed_count == DEFAULT_LENGTH {
                atr_value = Some(atr_seed_sum / DEFAULT_LENGTH as f64);
            }
        } else if let Some(previous_atr) = atr_value {
            atr_value =
                Some(((DEFAULT_LENGTH - 1) as f64 * previous_atr + tr) / DEFAULT_LENGTH as f64);
        }

        let rising = if valid_src_count >= DEFAULT_LENGTH && max_head < max_tail {
            src > max_val[max_head & MASK] + FLOAT_TOL
        } else {
            false
        };
        let falling = if valid_src_count >= DEFAULT_LENGTH && min_head < min_tail {
            src < min_val[min_head & MASK] - FLOAT_TOL
        } else {
            false
        };

        let prev_os = os;
        let new_os = if rising {
            1
        } else if falling {
            -1
        } else {
            prev_os
        };

        let previous_diff = prev_diff.unwrap_or(f64::NAN);
        let rise = new_os - prev_os == 2 && previous_diff.is_finite() && previous_diff < 0.0;
        let fall = new_os - prev_os == -2 && previous_diff.is_finite() && previous_diff > 0.0;
        let up =
            previous_diff.is_finite() && previous_diff <= 0.0 && diff.is_finite() && diff > 0.0;
        let dn =
            previous_diff.is_finite() && previous_diff >= 0.0 && diff.is_finite() && diff < 0.0;
        let event = up || dn || rise || fall;

        let mut ts = f64::NAN;
        if let Some(atr) = atr_value {
            let step = atr / DEFAULT_LENGTH as f64;
            if event {
                last_event_step = Some(step);
                bars_since_event = 0;
            } else if last_event_step.is_some() {
                bars_since_event = bars_since_event.saturating_add(1);
            }

            let prev_ts_or_src = prev_ts.unwrap_or(src);
            if up {
                ts = prev_ts_or_src - atr * DEFAULT_MULT;
            } else if dn {
                ts = prev_ts_or_src + atr * DEFAULT_MULT;
            } else if rise {
                ts = src - atr * DEFAULT_MULT;
            } else if fall {
                ts = src + atr * DEFAULT_MULT;
            } else if let Some(event_step) = last_event_step {
                ts = prev_ts_or_src + signum_with_tol(diff) * event_step * bars_since_event as f64;
            }
        }

        let result = if ts.is_finite() {
            let osc = src - ts;
            let next = match ema_value {
                Some(previous) => ema_alpha.mul_add(osc, ema_omalpha * previous),
                None => osc,
            };
            ema_value = Some(next);
            rsi_stream.update(next)
        } else {
            None
        };

        prev_src = Some(src);
        prev_diff = if diff.is_finite() { Some(diff) } else { None };
        prev_ts = if ts.is_finite() { Some(ts) } else { None };
        os = new_os;

        while max_tail > max_head {
            let pos = (max_tail - 1) & MASK;
            if max_val[pos] <= src {
                max_tail -= 1;
            } else {
                break;
            }
        }
        let max_pos = max_tail & MASK;
        max_idx[max_pos] = valid_src_count;
        max_val[max_pos] = src;
        max_tail += 1;

        while min_tail > min_head {
            let pos = (min_tail - 1) & MASK;
            if min_val[pos] >= src {
                min_tail -= 1;
            } else {
                break;
            }
        }
        let min_pos = min_tail & MASK;
        min_idx[min_pos] = valid_src_count;
        min_val[min_pos] = src;
        min_tail += 1;

        valid_src_count += 1;
        let window_start = valid_src_count.saturating_sub(DEFAULT_LENGTH);
        while max_head < max_tail && max_idx[max_head & MASK] < window_start {
            max_head += 1;
        }
        while min_head < min_tail && min_idx[min_head & MASK] < window_start {
            min_head += 1;
        }

        out[i] = result.unwrap_or(f64::NAN);
    }
}

pub fn grover_llorens_cycle_oscillator_with_kernel(
    input: &GroverLlorensCycleOscillatorInput,
    kernel: Kernel,
) -> Result<GroverLlorensCycleOscillatorOutput, GroverLlorensCycleOscillatorError> {
    let (open, high, low, close, params) = prepare_input(input, kernel)?;
    let mut values = alloc_uninit_f64(close.len());
    compute_row(open, high, low, close, params, &mut values);
    Ok(GroverLlorensCycleOscillatorOutput { values })
}

#[inline]
pub fn grover_llorens_cycle_oscillator_into_slice(
    dst: &mut [f64],
    input: &GroverLlorensCycleOscillatorInput,
    kernel: Kernel,
) -> Result<(), GroverLlorensCycleOscillatorError> {
    let (open, high, low, close, params) = prepare_input(input, kernel)?;
    if dst.len() != close.len() {
        return Err(GroverLlorensCycleOscillatorError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }
    compute_row(open, high, low, close, params, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn grover_llorens_cycle_oscillator_into(
    input: &GroverLlorensCycleOscillatorInput,
    out: &mut [f64],
) -> Result<(), GroverLlorensCycleOscillatorError> {
    grover_llorens_cycle_oscillator_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct GroverLlorensCycleOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub mult: (f64, f64, f64),
    pub source: String,
    pub smooth: bool,
    pub rsi_period: (usize, usize, usize),
}

impl Default for GroverLlorensCycleOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            mult: (DEFAULT_MULT, DEFAULT_MULT, 0.0),
            source: DEFAULT_SOURCE.to_string(),
            smooth: DEFAULT_SMOOTH,
            rsi_period: (DEFAULT_RSI_PERIOD, DEFAULT_RSI_PERIOD, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GroverLlorensCycleOscillatorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GroverLlorensCycleOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct GroverLlorensCycleOscillatorBatchBuilder {
    range: GroverLlorensCycleOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for GroverLlorensCycleOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: GroverLlorensCycleOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl GroverLlorensCycleOscillatorBatchBuilder {
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
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step);
        self
    }

    #[inline]
    pub fn source(mut self, source: &str) -> Result<Self, GroverLlorensCycleOscillatorError> {
        self.range.source = parse_source(source)?.as_str().to_string();
        Ok(self)
    }

    #[inline]
    pub fn smooth(mut self, smooth: bool) -> Self {
        self.range.smooth = smooth;
        self
    }

    #[inline]
    pub fn rsi_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rsi_period = (start, end, step);
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<GroverLlorensCycleOscillatorBatchOutput, GroverLlorensCycleOscillatorError> {
        self.apply_slices(&candles.open, &candles.high, &candles.low, &candles.close)
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<GroverLlorensCycleOscillatorBatchOutput, GroverLlorensCycleOscillatorError> {
        grover_llorens_cycle_oscillator_batch_with_kernel(
            open,
            high,
            low,
            close,
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_grid(
    range: &GroverLlorensCycleOscillatorBatchRange,
) -> Result<Vec<GroverLlorensCycleOscillatorParams>, GroverLlorensCycleOscillatorError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, GroverLlorensCycleOscillatorError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            while value <= end {
                out.push(value);
                match value.checked_add(step) {
                    Some(next) if next > value => value = next,
                    _ => break,
                }
            }
        } else {
            let mut value = start;
            loop {
                out.push(value);
                if value == end {
                    break;
                }
                let next = value.saturating_sub(step);
                if next >= value || next < end {
                    break;
                }
                value = next;
            }
        }
        if out.is_empty() {
            return Err(GroverLlorensCycleOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, GroverLlorensCycleOscillatorError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(GroverLlorensCycleOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        if step.abs() <= FLOAT_TOL || (start - end).abs() <= FLOAT_TOL {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            while value <= end + FLOAT_TOL {
                out.push(value);
                value += step.abs();
                if out.len() > 1_000_000 {
                    break;
                }
            }
        } else {
            let mut value = start;
            while value >= end - FLOAT_TOL {
                out.push(value);
                value -= step.abs();
                if out.len() > 1_000_000 {
                    break;
                }
            }
        }
        if out.is_empty() {
            return Err(GroverLlorensCycleOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let lengths = axis_usize(range.length)?;
    let mults = axis_f64(range.mult)?;
    let rsi_periods = axis_usize(range.rsi_period)?;
    let source = parse_source(&range.source)?.as_str().to_string();

    let mut combos = Vec::with_capacity(lengths.len() * mults.len() * rsi_periods.len());
    for &length in &lengths {
        for &mult in &mults {
            for &rsi_period in &rsi_periods {
                combos.push(GroverLlorensCycleOscillatorParams {
                    length: Some(length),
                    mult: Some(mult),
                    source: Some(source.clone()),
                    smooth: Some(range.smooth),
                    rsi_period: Some(rsi_period),
                });
            }
        }
    }
    Ok(combos)
}

pub fn grover_llorens_cycle_oscillator_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GroverLlorensCycleOscillatorBatchRange,
    kernel: Kernel,
) -> Result<GroverLlorensCycleOscillatorBatchOutput, GroverLlorensCycleOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(GroverLlorensCycleOscillatorError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    grover_llorens_cycle_oscillator_batch_par_slice(
        open,
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline]
pub fn grover_llorens_cycle_oscillator_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GroverLlorensCycleOscillatorBatchRange,
    kernel: Kernel,
) -> Result<GroverLlorensCycleOscillatorBatchOutput, GroverLlorensCycleOscillatorError> {
    grover_llorens_cycle_oscillator_batch_inner(open, high, low, close, sweep, kernel, false)
}

#[inline]
pub fn grover_llorens_cycle_oscillator_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GroverLlorensCycleOscillatorBatchRange,
    kernel: Kernel,
) -> Result<GroverLlorensCycleOscillatorBatchOutput, GroverLlorensCycleOscillatorError> {
    grover_llorens_cycle_oscillator_batch_inner(open, high, low, close, sweep, kernel, true)
}

#[inline(always)]
fn grover_llorens_cycle_oscillator_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GroverLlorensCycleOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<GroverLlorensCycleOscillatorBatchOutput, GroverLlorensCycleOscillatorError> {
    let combos = expand_grid(sweep)?;
    let len = close.len();
    if len == 0 {
        return Err(GroverLlorensCycleOscillatorError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(
            GroverLlorensCycleOscillatorError::InconsistentSliceLengths {
                open_len: open.len(),
                high_len: high.len(),
                low_len: low.len(),
                close_len: len,
            },
        );
    }

    let source = parse_source(&sweep.source)?;
    let (first, valid) = scan_valid_bars(source, open, high, low, close);
    if first >= len {
        return Err(GroverLlorensCycleOscillatorError::AllValuesNaN);
    }
    let needed = combos
        .iter()
        .map(|combo| {
            combo
                .length
                .unwrap_or(DEFAULT_LENGTH)
                .max(combo.rsi_period.unwrap_or(DEFAULT_RSI_PERIOD))
        })
        .max()
        .unwrap_or(0);
    if valid < needed {
        return Err(GroverLlorensCycleOscillatorError::NotEnoughValidData { needed, valid });
    }

    let rows = combos.len();
    let cols = len;
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let params = resolve_params(&combos[row], cols).expect("validated combo");
                compute_row(open, high, low, close, params, out_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let params = resolve_params(&combos[row], cols).expect("validated combo");
            compute_row(open, high, low, close, params, out_row);
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let params = resolve_params(&combos[row], cols)?;
            compute_row(open, high, low, close, params, out_row);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(GroverLlorensCycleOscillatorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn grover_llorens_cycle_oscillator_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GroverLlorensCycleOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<GroverLlorensCycleOscillatorParams>, GroverLlorensCycleOscillatorError> {
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(GroverLlorensCycleOscillatorError::EmptyInputData);
    }
    if open.len() != cols || high.len() != cols || low.len() != cols {
        return Err(
            GroverLlorensCycleOscillatorError::InconsistentSliceLengths {
                open_len: open.len(),
                high_len: high.len(),
                low_len: low.len(),
                close_len: cols,
            },
        );
    }
    let total =
        rows.checked_mul(cols)
            .ok_or(GroverLlorensCycleOscillatorError::OutputLengthMismatch {
                expected: usize::MAX,
                got: out.len(),
            })?;
    if out.len() != total {
        return Err(GroverLlorensCycleOscillatorError::OutputLengthMismatch {
            expected: total,
            got: out.len(),
        });
    }

    let source = parse_source(&sweep.source)?;
    let (first, valid) = scan_valid_bars(source, open, high, low, close);
    if first >= cols {
        return Err(GroverLlorensCycleOscillatorError::AllValuesNaN);
    }
    let needed = combos
        .iter()
        .map(|combo| {
            combo
                .length
                .unwrap_or(DEFAULT_LENGTH)
                .max(combo.rsi_period.unwrap_or(DEFAULT_RSI_PERIOD))
        })
        .max()
        .unwrap_or(0);
    if valid < needed {
        return Err(GroverLlorensCycleOscillatorError::NotEnoughValidData { needed, valid });
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let params = resolve_params(&combos[row], cols).expect("validated combo");
                compute_row(open, high, low, close, params, out_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let params = resolve_params(&combos[row], cols).expect("validated combo");
            compute_row(open, high, low, close, params, out_row);
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let params = resolve_params(&combos[row], cols)?;
            compute_row(open, high, low, close, params, out_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "grover_llorens_cycle_oscillator")]
#[pyo3(signature = (open, high, low, close, length=DEFAULT_LENGTH, mult=DEFAULT_MULT, source=DEFAULT_SOURCE, smooth=DEFAULT_SMOOTH, rsi_period=DEFAULT_RSI_PERIOD, kernel=None))]
pub fn grover_llorens_cycle_oscillator_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    mult: f64,
    source: &str,
    smooth: bool,
    rsi_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let kern = validate_kernel(kernel, false)?;
    let input = GroverLlorensCycleOscillatorInput::from_slices(
        open,
        high,
        low,
        close,
        GroverLlorensCycleOscillatorParams {
            length: Some(length),
            mult: Some(mult),
            source: Some(source.to_string()),
            smooth: Some(smooth),
            rsi_period: Some(rsi_period),
        },
    );
    let output = py
        .allow_threads(|| grover_llorens_cycle_oscillator_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "GroverLlorensCycleOscillatorStream")]
pub struct GroverLlorensCycleOscillatorStreamPy {
    inner: GroverLlorensCycleOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl GroverLlorensCycleOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, mult=DEFAULT_MULT, source=DEFAULT_SOURCE, smooth=DEFAULT_SMOOTH, rsi_period=DEFAULT_RSI_PERIOD))]
    fn new(
        length: usize,
        mult: f64,
        source: &str,
        smooth: bool,
        rsi_period: usize,
    ) -> PyResult<Self> {
        let inner =
            GroverLlorensCycleOscillatorStream::try_new(GroverLlorensCycleOscillatorParams {
                length: Some(length),
                mult: Some(mult),
                source: Some(source.to_string()),
                smooth: Some(smooth),
                rsi_period: Some(rsi_period),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<f64> {
        self.inner.update(open, high, low, close)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "grover_llorens_cycle_oscillator_batch")]
#[pyo3(signature = (open, high, low, close, length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0), mult_range=(DEFAULT_MULT, DEFAULT_MULT, 0.0), source=DEFAULT_SOURCE, smooth=DEFAULT_SMOOTH, rsi_period_range=(DEFAULT_RSI_PERIOD, DEFAULT_RSI_PERIOD, 0), kernel=None))]
pub fn grover_llorens_cycle_oscillator_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    source: &str,
    smooth: bool,
    rsi_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let kern = validate_kernel(kernel, true)?;
    let sweep = GroverLlorensCycleOscillatorBatchRange {
        length: length_range,
        mult: mult_range,
        source: source.to_string(),
        smooth,
        rsi_period: rsi_period_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let values_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let values_slice = unsafe { values_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            grover_llorens_cycle_oscillator_batch_inner_into(
                open,
                high,
                low,
                close,
                &sweep,
                batch.to_non_batch(),
                true,
                values_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", values_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "mults",
        combos
            .iter()
            .map(|combo| combo.mult.unwrap_or(DEFAULT_MULT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "sources",
        combos
            .iter()
            .map(|combo| combo.source.as_deref().unwrap_or(DEFAULT_SOURCE))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "smooth_flags",
        combos
            .iter()
            .map(|combo| combo.smooth.unwrap_or(DEFAULT_SMOOTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "rsi_periods",
        combos
            .iter()
            .map(|combo| combo.rsi_period.unwrap_or(DEFAULT_RSI_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_grover_llorens_cycle_oscillator_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        grover_llorens_cycle_oscillator_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        grover_llorens_cycle_oscillator_batch_py,
        module
    )?)?;
    module.add_class::<GroverLlorensCycleOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
struct GroverLlorensCycleOscillatorJsOutput {
    values: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "grover_llorens_cycle_oscillator_js")]
pub fn grover_llorens_cycle_oscillator_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    mult: f64,
    source: &str,
    smooth: bool,
    rsi_period: usize,
) -> Result<JsValue, JsValue> {
    let input = GroverLlorensCycleOscillatorInput::from_slices(
        open,
        high,
        low,
        close,
        GroverLlorensCycleOscillatorParams {
            length: Some(length),
            mult: Some(mult),
            source: Some(source.to_string()),
            smooth: Some(smooth),
            rsi_period: Some(rsi_period),
        },
    );
    let out =
        grover_llorens_cycle_oscillator(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&GroverLlorensCycleOscillatorJsOutput { values: out.values })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn grover_llorens_cycle_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn grover_llorens_cycle_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn grover_llorens_cycle_oscillator_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    values_ptr: *mut f64,
    len: usize,
    length: usize,
    mult: f64,
    source: &str,
    smooth: bool,
    rsi_period: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || values_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = GroverLlorensCycleOscillatorInput::from_slices(
            open,
            high,
            low,
            close,
            GroverLlorensCycleOscillatorParams {
                length: Some(length),
                mult: Some(mult),
                source: Some(source.to_string()),
                smooth: Some(smooth),
                rsi_period: Some(rsi_period),
            },
        );
        let out = std::slice::from_raw_parts_mut(values_ptr, len);
        grover_llorens_cycle_oscillator_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct GroverLlorensCycleOscillatorBatchConfig {
    pub length_range: Option<(usize, usize, usize)>,
    pub mult_range: Option<(f64, f64, f64)>,
    pub source: Option<String>,
    pub smooth: Option<bool>,
    pub rsi_period_range: Option<(usize, usize, usize)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct GroverLlorensCycleOscillatorBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GroverLlorensCycleOscillatorParams>,
    pub lengths: Vec<usize>,
    pub mults: Vec<f64>,
    pub sources: Vec<String>,
    pub smooth_flags: Vec<bool>,
    pub rsi_periods: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "grover_llorens_cycle_oscillator_batch_js")]
pub fn grover_llorens_cycle_oscillator_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: GroverLlorensCycleOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = GroverLlorensCycleOscillatorBatchRange {
        length: config
            .length_range
            .unwrap_or((DEFAULT_LENGTH, DEFAULT_LENGTH, 0)),
        mult: config
            .mult_range
            .unwrap_or((DEFAULT_MULT, DEFAULT_MULT, 0.0)),
        source: config.source.unwrap_or_else(|| DEFAULT_SOURCE.to_string()),
        smooth: config.smooth.unwrap_or(DEFAULT_SMOOTH),
        rsi_period: config
            .rsi_period_range
            .unwrap_or((DEFAULT_RSI_PERIOD, DEFAULT_RSI_PERIOD, 0)),
    };
    let out = grover_llorens_cycle_oscillator_batch_inner(
        open,
        high,
        low,
        close,
        &sweep,
        detect_best_batch_kernel().to_non_batch(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&GroverLlorensCycleOscillatorBatchJsOutput {
        lengths: out
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        mults: out
            .combos
            .iter()
            .map(|combo| combo.mult.unwrap_or(DEFAULT_MULT))
            .collect(),
        sources: out
            .combos
            .iter()
            .map(|combo| {
                combo
                    .source
                    .clone()
                    .unwrap_or_else(|| DEFAULT_SOURCE.to_string())
            })
            .collect(),
        smooth_flags: out
            .combos
            .iter()
            .map(|combo| combo.smooth.unwrap_or(DEFAULT_SMOOTH))
            .collect(),
        rsi_periods: out
            .combos
            .iter()
            .map(|combo| combo.rsi_period.unwrap_or(DEFAULT_RSI_PERIOD))
            .collect(),
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn grover_llorens_cycle_oscillator_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    mult_start: f64,
    mult_end: f64,
    mult_step: f64,
    source: &str,
    smooth: bool,
    rsi_period_start: usize,
    rsi_period_end: usize,
    rsi_period_step: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = GroverLlorensCycleOscillatorBatchRange {
        length: (length_start, length_end, length_step),
        mult: (mult_start, mult_end, mult_step),
        source: source.to_string(),
        smooth,
        rsi_period: (rsi_period_start, rsi_period_end, rsi_period_step),
    };
    let rows = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        grover_llorens_cycle_oscillator_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            detect_best_batch_kernel().to_non_batch(),
            false,
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn grover_llorens_cycle_oscillator_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    mult: f64,
    source: &str,
    smooth: bool,
    rsi_period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = grover_llorens_cycle_oscillator_js(
        open, high, low, close, length, mult, source, smooth, rsi_period,
    )?;
    crate::write_wasm_object_f64_outputs(
        "grover_llorens_cycle_oscillator_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn grover_llorens_cycle_oscillator_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = grover_llorens_cycle_oscillator_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "grover_llorens_cycle_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = vec![f64::NAN; len];
        let mut high = vec![f64::NAN; len];
        let mut low = vec![f64::NAN; len];
        let mut close = vec![f64::NAN; len];
        let mut prev = 100.0;
        for i in 2..len {
            let x = i as f64;
            let wave = (x * 0.11).sin() * 2.4 + (x * 0.037).cos() * 1.3;
            let o = prev + wave * 0.35;
            let c = o + (x * 0.19).sin() * 1.1 - (x * 0.07).cos() * 0.4;
            let h = o.max(c) + 0.6 + (x * 0.03).sin().abs() * 0.25;
            let l = o.min(c) - 0.6 - (x * 0.02).cos().abs() * 0.25;
            open[i] = o;
            high[i] = h;
            low[i] = l;
            close[i] = c;
            prev = c;
        }
        (open, high, low, close)
    }

    #[test]
    fn grover_llorens_cycle_oscillator_output_contract() {
        let (open, high, low, close) = sample_ohlc(512);
        let input = GroverLlorensCycleOscillatorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GroverLlorensCycleOscillatorParams::default(),
        );
        let out = grover_llorens_cycle_oscillator(&input).expect("glco");
        assert_eq!(out.values.len(), close.len());
        let first_valid = out
            .values
            .iter()
            .position(|v| v.is_finite())
            .expect("finite");
        assert!(first_valid >= 100);
        assert!(out
            .values
            .iter()
            .skip(first_valid + 32)
            .all(|v| v.is_finite()));
    }

    #[test]
    fn grover_llorens_cycle_oscillator_rejects_invalid_parameters() {
        let (open, high, low, close) = sample_ohlc(64);
        let bad_len = GroverLlorensCycleOscillatorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GroverLlorensCycleOscillatorParams {
                length: Some(0),
                ..GroverLlorensCycleOscillatorParams::default()
            },
        );
        assert!(matches!(
            grover_llorens_cycle_oscillator(&bad_len),
            Err(GroverLlorensCycleOscillatorError::InvalidLength { .. })
        ));

        let bad_source = GroverLlorensCycleOscillatorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GroverLlorensCycleOscillatorParams {
                source: Some("bad".to_string()),
                ..GroverLlorensCycleOscillatorParams::default()
            },
        );
        assert!(matches!(
            grover_llorens_cycle_oscillator(&bad_source),
            Err(GroverLlorensCycleOscillatorError::InvalidSource { .. })
        ));
    }

    #[test]
    fn grover_llorens_cycle_oscillator_stream_matches_batch_with_reset() {
        let (mut open, mut high, mut low, mut close) = sample_ohlc(256);
        open[120] = f64::NAN;
        high[120] = f64::NAN;
        low[120] = f64::NAN;
        close[120] = f64::NAN;

        let params = GroverLlorensCycleOscillatorParams {
            length: Some(60),
            mult: Some(8.0),
            source: Some("hlc3".to_string()),
            smooth: Some(true),
            rsi_period: Some(14),
        };
        let input = GroverLlorensCycleOscillatorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            params.clone(),
        );
        let batch = grover_llorens_cycle_oscillator(&input).expect("batch");
        let mut stream = GroverLlorensCycleOscillatorStream::try_new(params).expect("stream");

        let mut streamed = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            streamed.push(
                stream
                    .update(open[i], high[i], low[i], close[i])
                    .unwrap_or(f64::NAN),
            );
        }

        for i in 0..close.len() {
            if batch.values[i].is_nan() {
                assert!(streamed[i].is_nan(), "expected NaN at {i}");
            } else {
                assert!(
                    (batch.values[i] - streamed[i]).abs() <= 1e-12,
                    "mismatch at {i}: {} vs {}",
                    batch.values[i],
                    streamed[i]
                );
            }
        }
    }

    #[test]
    fn grover_llorens_cycle_oscillator_batch_single_param_matches_single() {
        let (open, high, low, close) = sample_ohlc(192);
        let sweep = GroverLlorensCycleOscillatorBatchRange {
            length: (50, 50, 0),
            mult: (8.0, 8.0, 0.0),
            source: "close".to_string(),
            smooth: true,
            rsi_period: (14, 14, 0),
        };
        let batch = grover_llorens_cycle_oscillator_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &sweep,
            Kernel::Auto,
        )
        .expect("batch");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());

        let single =
            grover_llorens_cycle_oscillator(&GroverLlorensCycleOscillatorInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                GroverLlorensCycleOscillatorParams {
                    length: Some(50),
                    mult: Some(8.0),
                    source: Some("close".to_string()),
                    smooth: Some(true),
                    rsi_period: Some(14),
                },
            ))
            .expect("single");

        assert_eq!(batch.values.len(), close.len());
        for i in 0..close.len() {
            if single.values[i].is_nan() {
                assert!(batch.values[i].is_nan(), "expected NaN at {i}");
            } else {
                assert!((single.values[i] - batch.values[i]).abs() <= 1e-12);
            }
        }
    }
}
