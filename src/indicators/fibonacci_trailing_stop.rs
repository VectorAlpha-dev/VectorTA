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
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LEFT_BARS: usize = 20;
const DEFAULT_RIGHT_BARS: usize = 1;
const DEFAULT_LEVEL: f64 = -0.382;
const DEFAULT_TRIGGER: &str = "close";
const FLOAT_TOL: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TriggerMode {
    Close,
    Wick,
}

impl TriggerMode {
    #[inline(always)]
    fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("close") {
            Some(Self::Close)
        } else if value.eq_ignore_ascii_case("wick") {
            Some(Self::Wick)
        } else {
            None
        }
    }

    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Close => "close",
            Self::Wick => "wick",
        }
    }
}

#[derive(Debug, Clone)]
pub enum FibonacciTrailingStopData<'a> {
    Candles(&'a Candles),
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct FibonacciTrailingStopOutput {
    pub trailing_stop: Vec<f64>,
    pub long_stop: Vec<f64>,
    pub short_stop: Vec<f64>,
    pub direction: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct FibonacciTrailingStopPoint {
    pub trailing_stop: f64,
    pub long_stop: f64,
    pub short_stop: f64,
    pub direction: f64,
}

impl FibonacciTrailingStopPoint {
    #[inline(always)]
    fn nan() -> Self {
        Self {
            trailing_stop: f64::NAN,
            long_stop: f64::NAN,
            short_stop: f64::NAN,
            direction: f64::NAN,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct FibonacciTrailingStopParams {
    pub left_bars: Option<usize>,
    pub right_bars: Option<usize>,
    pub level: Option<f64>,
    pub trigger: Option<String>,
}

impl Default for FibonacciTrailingStopParams {
    fn default() -> Self {
        Self {
            left_bars: Some(DEFAULT_LEFT_BARS),
            right_bars: Some(DEFAULT_RIGHT_BARS),
            level: Some(DEFAULT_LEVEL),
            trigger: Some(DEFAULT_TRIGGER.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FibonacciTrailingStopInput<'a> {
    pub data: FibonacciTrailingStopData<'a>,
    pub params: FibonacciTrailingStopParams,
}

impl<'a> FibonacciTrailingStopInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: FibonacciTrailingStopParams) -> Self {
        Self {
            data: FibonacciTrailingStopData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: FibonacciTrailingStopParams,
    ) -> Self {
        Self {
            data: FibonacciTrailingStopData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FibonacciTrailingStopParams::default())
    }

    #[inline]
    pub fn as_slices(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            FibonacciTrailingStopData::Candles(candles) => {
                (&candles.high, &candles.low, &candles.close)
            }
            FibonacciTrailingStopData::Slices { high, low, close } => (high, low, close),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FibonacciTrailingStopBuilder {
    left_bars: Option<usize>,
    right_bars: Option<usize>,
    level: Option<f64>,
    trigger: Option<TriggerMode>,
    kernel: Kernel,
}

impl FibonacciTrailingStopBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn left_bars(mut self, value: usize) -> Self {
        self.left_bars = Some(value);
        self
    }

    #[inline]
    pub fn right_bars(mut self, value: usize) -> Self {
        self.right_bars = Some(value);
        self
    }

    #[inline]
    pub fn level(mut self, value: f64) -> Self {
        self.level = Some(value);
        self
    }

    #[inline]
    pub fn trigger(mut self, value: &str) -> Result<Self, FibonacciTrailingStopError> {
        self.trigger = Some(TriggerMode::parse(value).ok_or_else(|| {
            FibonacciTrailingStopError::InvalidTrigger {
                trigger: value.to_string(),
            }
        })?);
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
    ) -> Result<FibonacciTrailingStopOutput, FibonacciTrailingStopError> {
        let input = FibonacciTrailingStopInput::from_candles(
            candles,
            FibonacciTrailingStopParams {
                left_bars: self.left_bars,
                right_bars: self.right_bars,
                level: self.level,
                trigger: Some(
                    self.trigger
                        .unwrap_or(TriggerMode::Close)
                        .as_str()
                        .to_string(),
                ),
            },
        );
        fibonacci_trailing_stop_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FibonacciTrailingStopOutput, FibonacciTrailingStopError> {
        let input = FibonacciTrailingStopInput::from_slices(
            high,
            low,
            close,
            FibonacciTrailingStopParams {
                left_bars: self.left_bars,
                right_bars: self.right_bars,
                level: self.level,
                trigger: Some(
                    self.trigger
                        .unwrap_or(TriggerMode::Close)
                        .as_str()
                        .to_string(),
                ),
            },
        );
        fibonacci_trailing_stop_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<FibonacciTrailingStopStream, FibonacciTrailingStopError> {
        FibonacciTrailingStopStream::try_new(FibonacciTrailingStopParams {
            left_bars: self.left_bars,
            right_bars: self.right_bars,
            level: self.level,
            trigger: Some(
                self.trigger
                    .unwrap_or(TriggerMode::Close)
                    .as_str()
                    .to_string(),
            ),
        })
    }
}

#[derive(Debug, Error)]
pub enum FibonacciTrailingStopError {
    #[error("fibonacci_trailing_stop: Input data slice is empty.")]
    EmptyInputData,
    #[error("fibonacci_trailing_stop: Input slice lengths differ: high={high_len}, low={low_len}, close={close_len}.")]
    MismatchedInputLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("fibonacci_trailing_stop: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "fibonacci_trailing_stop: Invalid left_bars: left_bars = {left_bars}, data length = {data_len}"
    )]
    InvalidLeftBars { left_bars: usize, data_len: usize },
    #[error(
        "fibonacci_trailing_stop: Invalid right_bars: right_bars = {right_bars}, data length = {data_len}"
    )]
    InvalidRightBars { right_bars: usize, data_len: usize },
    #[error("fibonacci_trailing_stop: Invalid level: {level}")]
    InvalidLevel { level: f64 },
    #[error("fibonacci_trailing_stop: Invalid trigger: {trigger}")]
    InvalidTrigger { trigger: String },
    #[error("fibonacci_trailing_stop: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fibonacci_trailing_stop: Output length mismatch: expected = {expected}")]
    OutputLengthMismatch { expected: usize },
    #[error("fibonacci_trailing_stop: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("fibonacci_trailing_stop: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    left_bars: usize,
    right_bars: usize,
    left_small: usize,
    right_small: usize,
    level: f64,
    trigger: TriggerMode,
}

#[derive(Clone, Copy, Debug)]
struct PivotPoint {
    price: f64,
    dir: i8,
}

#[derive(Clone, Debug)]
struct CoreState {
    trigger: TriggerMode,
    level: f64,
    dir: i8,
    st: f64,
    max_level: f64,
    min_level: f64,
    pivots: Vec<PivotPoint>,
}

impl CoreState {
    #[inline]
    fn new(high: f64, low: f64, close: f64, params: ResolvedParams) -> Self {
        Self {
            trigger: params.trigger,
            level: params.level,
            dir: 0,
            st: close,
            max_level: high,
            min_level: low,
            pivots: Vec::with_capacity(3),
        }
    }

    #[inline]
    fn update_pivots(&mut self, ph: Option<f64>, pl: Option<f64>) {
        if let Some(value) = ph {
            if let Some(first) = self.pivots.first_mut() {
                if first.dir > 0 && value > first.price {
                    first.price = value;
                } else if first.dir < 0 && value > first.price {
                    self.pivots.insert(
                        0,
                        PivotPoint {
                            price: value,
                            dir: 1,
                        },
                    );
                }
            } else {
                self.pivots.push(PivotPoint {
                    price: value,
                    dir: 1,
                });
            }
        }

        if let Some(value) = pl {
            if let Some(first) = self.pivots.first_mut() {
                if first.dir < 0 && value < first.price {
                    first.price = value;
                } else if first.dir > 0 && value < first.price {
                    self.pivots.insert(
                        0,
                        PivotPoint {
                            price: value,
                            dir: -1,
                        },
                    );
                }
            } else {
                self.pivots.push(PivotPoint {
                    price: value,
                    dir: -1,
                });
            }
        }

        if self.pivots.len() > 3 {
            self.pivots.truncate(3);
        }
    }

    #[inline]
    fn apply_bar(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        ph: Option<f64>,
        pl: Option<f64>,
    ) -> FibonacciTrailingStopPoint {
        self.update_pivots(ph, pl);

        if self.pivots.len() >= 2 {
            let p0 = self.pivots[0].price;
            let p1 = self.pivots[1].price;
            let mut max_value = p0.max(p1);
            let mut min_value = p0.min(p1);
            if self.pivots.len() == 2 {
                self.st = (max_value + min_value) * 0.5;
            }
            let dif = max_value - min_value;
            max_value += dif * self.level;
            min_value -= dif * self.level;
            self.max_level = max_value;
            self.min_level = min_value;
        }

        let price = match self.trigger {
            TriggerMode::Close => close,
            TriggerMode::Wick => {
                if self.dir < 1 {
                    high
                } else {
                    low
                }
            }
        };

        if self.dir < 1 {
            if price > self.st {
                self.st = self.min_level;
                self.dir = 1;
            } else {
                self.st = self.st.min(self.max_level);
            }
        }

        if self.dir > -1 {
            if price < self.st {
                self.st = self.max_level;
                self.dir = -1;
            } else {
                self.st = self.st.max(self.min_level);
            }
        }

        FibonacciTrailingStopPoint {
            trailing_stop: self.st,
            long_stop: if self.dir == 1 { self.st } else { f64::NAN },
            short_stop: if self.dir == -1 { self.st } else { f64::NAN },
            direction: self.dir as f64,
        }
    }
}

#[inline(always)]
fn first_valid_ohlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    for i in 0..high.len() {
        if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
            return i;
        }
    }
    high.len()
}

#[inline(always)]
fn max_consecutive_valid_ohlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut run = 0usize;
    for i in 0..high.len() {
        if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
            run += 1;
            if run > best {
                best = run;
            }
        } else {
            run = 0;
        }
    }
    best
}

#[inline(always)]
fn canonical_trigger_name(trigger: Option<&str>) -> String {
    trigger.unwrap_or(DEFAULT_TRIGGER).to_ascii_lowercase()
}

#[inline]
fn resolve_params(
    params: &FibonacciTrailingStopParams,
    data_len: Option<usize>,
) -> Result<ResolvedParams, FibonacciTrailingStopError> {
    let left_bars = params.left_bars.unwrap_or(DEFAULT_LEFT_BARS);
    let right_bars = params.right_bars.unwrap_or(DEFAULT_RIGHT_BARS);
    let level = params.level.unwrap_or(DEFAULT_LEVEL);
    let trigger_name = canonical_trigger_name(params.trigger.as_deref());
    let trigger = TriggerMode::parse(&trigger_name).ok_or_else(|| {
        FibonacciTrailingStopError::InvalidTrigger {
            trigger: trigger_name.clone(),
        }
    })?;

    if left_bars == 0 {
        return Err(FibonacciTrailingStopError::InvalidLeftBars {
            left_bars,
            data_len: data_len.unwrap_or(0),
        });
    }
    if right_bars == 0 {
        return Err(FibonacciTrailingStopError::InvalidRightBars {
            right_bars,
            data_len: data_len.unwrap_or(0),
        });
    }
    if !level.is_finite() {
        return Err(FibonacciTrailingStopError::InvalidLevel { level });
    }
    if let Some(len) = data_len {
        let needed = left_bars + right_bars + 1;
        if needed > len {
            return Err(FibonacciTrailingStopError::NotEnoughValidData { needed, valid: len });
        }
    }

    Ok(ResolvedParams {
        left_bars,
        right_bars,
        left_small: ((left_bars + 1) / 2).max(1),
        right_small: ((right_bars + 1) / 2).max(1),
        level,
        trigger,
    })
}

#[inline(always)]
fn confirmed_pivot_high_at(data: &[f64], idx: usize, left: usize, right: usize) -> Option<f64> {
    if idx < right {
        return None;
    }
    let center = idx - right;
    if center < left || center + right >= data.len() {
        return None;
    }
    let candidate = data[center];
    if !candidate.is_finite() {
        return None;
    }
    for &value in &data[(center - left)..=(center + right)] {
        if !value.is_finite() || value > candidate {
            return None;
        }
    }
    Some(candidate)
}

#[inline(always)]
fn confirmed_pivot_low_at(data: &[f64], idx: usize, left: usize, right: usize) -> Option<f64> {
    if idx < right {
        return None;
    }
    let center = idx - right;
    if center < left || center + right >= data.len() {
        return None;
    }
    let candidate = data[center];
    if !candidate.is_finite() {
        return None;
    }
    for &value in &data[(center - left)..=(center + right)] {
        if !value.is_finite() || value < candidate {
            return None;
        }
    }
    Some(candidate)
}

#[inline(always)]
fn buffer_pivot_high(data: &VecDeque<f64>, left: usize, right: usize) -> Option<f64> {
    if data.len() < left + right + 1 {
        return None;
    }
    let center = data.len() - 1 - right;
    let candidate = data[center];
    if !candidate.is_finite() {
        return None;
    }
    for i in (center - left)..=(center + right) {
        let value = data[i];
        if !value.is_finite() || value > candidate {
            return None;
        }
    }
    Some(candidate)
}

#[inline(always)]
fn buffer_pivot_low(data: &VecDeque<f64>, left: usize, right: usize) -> Option<f64> {
    if data.len() < left + right + 1 {
        return None;
    }
    let center = data.len() - 1 - right;
    let candidate = data[center];
    if !candidate.is_finite() {
        return None;
    }
    for i in (center - left)..=(center + right) {
        let value = data[i];
        if !value.is_finite() || value < candidate {
            return None;
        }
    }
    Some(candidate)
}

#[derive(Clone, Debug)]
pub struct FibonacciTrailingStopStream {
    params: ResolvedParams,
    state: Option<CoreState>,
    high_buf: VecDeque<f64>,
    low_buf: VecDeque<f64>,
    max_window: usize,
}

impl FibonacciTrailingStopStream {
    #[inline]
    pub fn try_new(
        params: FibonacciTrailingStopParams,
    ) -> Result<Self, FibonacciTrailingStopError> {
        let params = resolve_params(&params, None)?;
        let max_window = (params.left_bars + params.right_bars + 1)
            .max(params.left_small + params.right_small + 1);
        Ok(Self {
            params,
            state: None,
            high_buf: VecDeque::with_capacity(max_window),
            low_buf: VecDeque::with_capacity(max_window),
            max_window,
        })
    }

    #[inline]
    pub fn reset(&mut self) {
        self.state = None;
        self.high_buf.clear();
        self.low_buf.clear();
    }

    #[inline]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<FibonacciTrailingStopPoint> {
        if !(high.is_finite() && low.is_finite() && close.is_finite()) {
            self.reset();
            return None;
        }

        self.high_buf.push_back(high);
        self.low_buf.push_back(low);
        if self.high_buf.len() > self.max_window {
            self.high_buf.pop_front();
            self.low_buf.pop_front();
        }

        let ph = buffer_pivot_high(
            &self.high_buf,
            self.params.left_bars,
            self.params.right_bars,
        );
        let pl = buffer_pivot_low(&self.low_buf, self.params.left_bars, self.params.right_bars);

        let state = self
            .state
            .get_or_insert_with(|| CoreState::new(high, low, close, self.params));
        Some(state.apply_bar(high, low, close, ph, pl))
    }
}

#[inline]
fn fibonacci_trailing_stop_prepare<'a>(
    input: &'a FibonacciTrailingStopInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], ResolvedParams), FibonacciTrailingStopError> {
    let (high, low, close) = input.as_slices();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FibonacciTrailingStopError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(FibonacciTrailingStopError::MismatchedInputLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let first = first_valid_ohlc(high, low, close);
    if first >= close.len() {
        return Err(FibonacciTrailingStopError::AllValuesNaN);
    }

    let params = resolve_params(&input.params, Some(close.len()))?;
    let needed = params.left_bars + params.right_bars + 1;
    let valid = max_consecutive_valid_ohlc(high, low, close);
    if valid < needed {
        return Err(FibonacciTrailingStopError::NotEnoughValidData { needed, valid });
    }

    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    Ok((high, low, close, params))
}

fn fibonacci_trailing_stop_row_from_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: ResolvedParams,
    trailing_stop: &mut [f64],
    long_stop: &mut [f64],
    short_stop: &mut [f64],
    direction: &mut [f64],
) {
    let mut state: Option<CoreState> = None;
    for i in 0..close.len() {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if !(h.is_finite() && l.is_finite() && c.is_finite()) {
            state = None;
            trailing_stop[i] = f64::NAN;
            long_stop[i] = f64::NAN;
            short_stop[i] = f64::NAN;
            direction[i] = f64::NAN;
            continue;
        }

        let ph = confirmed_pivot_high_at(high, i, params.left_bars, params.right_bars);
        let pl = confirmed_pivot_low_at(low, i, params.left_bars, params.right_bars);

        let point = state
            .get_or_insert_with(|| CoreState::new(h, l, c, params))
            .apply_bar(h, l, c, ph, pl);

        trailing_stop[i] = point.trailing_stop;
        long_stop[i] = point.long_stop;
        short_stop[i] = point.short_stop;
        direction[i] = point.direction;
    }
}

#[inline]
pub fn fibonacci_trailing_stop(
    input: &FibonacciTrailingStopInput,
) -> Result<FibonacciTrailingStopOutput, FibonacciTrailingStopError> {
    fibonacci_trailing_stop_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn fibonacci_trailing_stop_with_kernel(
    input: &FibonacciTrailingStopInput,
    kernel: Kernel,
) -> Result<FibonacciTrailingStopOutput, FibonacciTrailingStopError> {
    let (high, low, close, params) = fibonacci_trailing_stop_prepare(input, kernel)?;
    let len = close.len();
    let mut trailing_stop = alloc_uninit_f64(len);
    let mut long_stop = alloc_uninit_f64(len);
    let mut short_stop = alloc_uninit_f64(len);
    let mut direction = alloc_uninit_f64(len);
    fibonacci_trailing_stop_row_from_slices(
        high,
        low,
        close,
        params,
        &mut trailing_stop,
        &mut long_stop,
        &mut short_stop,
        &mut direction,
    );
    Ok(FibonacciTrailingStopOutput {
        trailing_stop,
        long_stop,
        short_stop,
        direction,
    })
}

#[inline]
pub fn fibonacci_trailing_stop_into_slices(
    trailing_stop: &mut [f64],
    long_stop: &mut [f64],
    short_stop: &mut [f64],
    direction: &mut [f64],
    input: &FibonacciTrailingStopInput,
    kernel: Kernel,
) -> Result<(), FibonacciTrailingStopError> {
    let expected = input.as_slices().2.len();
    if trailing_stop.len() != expected
        || long_stop.len() != expected
        || short_stop.len() != expected
        || direction.len() != expected
    {
        return Err(FibonacciTrailingStopError::OutputLengthMismatch { expected });
    }
    let (high, low, close, params) = fibonacci_trailing_stop_prepare(input, kernel)?;
    fibonacci_trailing_stop_row_from_slices(
        high,
        low,
        close,
        params,
        trailing_stop,
        long_stop,
        short_stop,
        direction,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn fibonacci_trailing_stop_into(
    input: &FibonacciTrailingStopInput,
    trailing_stop: &mut [f64],
    long_stop: &mut [f64],
    short_stop: &mut [f64],
    direction: &mut [f64],
) -> Result<(), FibonacciTrailingStopError> {
    fibonacci_trailing_stop_into_slices(
        trailing_stop,
        long_stop,
        short_stop,
        direction,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct FibonacciTrailingStopBatchRange {
    pub left_bars: (usize, usize, usize),
    pub right_bars: (usize, usize, usize),
    pub level: (f64, f64, f64),
    pub trigger: Option<String>,
}

impl Default for FibonacciTrailingStopBatchRange {
    fn default() -> Self {
        Self {
            left_bars: (DEFAULT_LEFT_BARS, DEFAULT_LEFT_BARS, 0),
            right_bars: (DEFAULT_RIGHT_BARS, DEFAULT_RIGHT_BARS, 0),
            level: (DEFAULT_LEVEL, DEFAULT_LEVEL, 0.0),
            trigger: Some(DEFAULT_TRIGGER.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FibonacciTrailingStopBatchOutput {
    pub trailing_stop: Vec<f64>,
    pub long_stop: Vec<f64>,
    pub short_stop: Vec<f64>,
    pub direction: Vec<f64>,
    pub combos: Vec<FibonacciTrailingStopParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct FibonacciTrailingStopBatchBuilder {
    range: FibonacciTrailingStopBatchRange,
    kernel: Kernel,
}

impl FibonacciTrailingStopBatchBuilder {
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
    pub fn left_bars_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.left_bars = (start, end, step);
        self
    }

    #[inline]
    pub fn right_bars_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.right_bars = (start, end, step);
        self
    }

    #[inline]
    pub fn level_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.level = (start, end, step);
        self
    }

    #[inline]
    pub fn trigger<T: Into<String>>(mut self, trigger: T) -> Self {
        self.range.trigger = Some(trigger.into());
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FibonacciTrailingStopBatchOutput, FibonacciTrailingStopError> {
        fibonacci_trailing_stop_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<FibonacciTrailingStopBatchOutput, FibonacciTrailingStopError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close)
    }
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, FibonacciTrailingStopError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            let next = value.saturating_add(step);
            if next == value {
                break;
            }
            value = next;
        }
    } else {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_sub(step);
            if next == value || next < end {
                break;
            }
            value = next;
        }
    }
    if out.is_empty() {
        return Err(FibonacciTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, FibonacciTrailingStopError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(FibonacciTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(FibonacciTrailingStopError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(FibonacciTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end + FLOAT_TOL {
        out.push(value.min(end));
        value += step;
    }
    if (out.last().copied().unwrap_or(start) - end).abs() > 1e-9 {
        return Err(FibonacciTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid_fibonacci_trailing_stop(
    sweep: &FibonacciTrailingStopBatchRange,
) -> Result<Vec<FibonacciTrailingStopParams>, FibonacciTrailingStopError> {
    let left_values = expand_axis_usize(sweep.left_bars)?;
    let right_values = expand_axis_usize(sweep.right_bars)?;
    let level_values = expand_axis_f64(sweep.level.0, sweep.level.1, sweep.level.2)?;
    let trigger_name = canonical_trigger_name(sweep.trigger.as_deref());
    let mut combos = Vec::with_capacity(
        left_values
            .len()
            .saturating_mul(right_values.len())
            .saturating_mul(level_values.len()),
    );
    for left_bars in left_values {
        for &right_bars in &right_values {
            for &level in &level_values {
                let params = FibonacciTrailingStopParams {
                    left_bars: Some(left_bars),
                    right_bars: Some(right_bars),
                    level: Some(level),
                    trigger: Some(trigger_name.clone()),
                };
                let _ = resolve_params(&params, None)?;
                combos.push(params);
            }
        }
    }
    Ok(combos)
}

#[inline]
pub fn fibonacci_trailing_stop_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FibonacciTrailingStopBatchRange,
    kernel: Kernel,
) -> Result<FibonacciTrailingStopBatchOutput, FibonacciTrailingStopError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(FibonacciTrailingStopError::InvalidKernelForBatch(other)),
    };
    fibonacci_trailing_stop_batch_par_slices(high, low, close, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn fibonacci_trailing_stop_batch_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FibonacciTrailingStopBatchRange,
    kernel: Kernel,
) -> Result<FibonacciTrailingStopBatchOutput, FibonacciTrailingStopError> {
    fibonacci_trailing_stop_batch_inner(high, low, close, sweep, kernel, false)
}

#[inline]
pub fn fibonacci_trailing_stop_batch_par_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FibonacciTrailingStopBatchRange,
    kernel: Kernel,
) -> Result<FibonacciTrailingStopBatchOutput, FibonacciTrailingStopError> {
    fibonacci_trailing_stop_batch_inner(high, low, close, sweep, kernel, true)
}

pub fn fibonacci_trailing_stop_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FibonacciTrailingStopBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<FibonacciTrailingStopBatchOutput, FibonacciTrailingStopError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FibonacciTrailingStopError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(FibonacciTrailingStopError::MismatchedInputLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let first = first_valid_ohlc(high, low, close);
    if first >= close.len() {
        return Err(FibonacciTrailingStopError::AllValuesNaN);
    }

    let combos = expand_grid_fibonacci_trailing_stop(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(FibonacciTrailingStopError::OutputLengthMismatch {
            expected: usize::MAX,
        })?;
    let resolved = combos
        .iter()
        .map(|params| resolve_params(params, Some(cols)))
        .collect::<Result<Vec<_>, _>>()?;
    let max_valid = max_consecutive_valid_ohlc(high, low, close);
    for params in &resolved {
        let needed = params.left_bars + params.right_bars + 1;
        if max_valid < needed {
            return Err(FibonacciTrailingStopError::NotEnoughValidData {
                needed,
                valid: max_valid,
            });
        }
    }

    let zero_prefixes = vec![0usize; rows];
    let mut trailing_stop_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut trailing_stop_mu, cols, &zero_prefixes);
    let mut trailing_stop_guard = ManuallyDrop::new(trailing_stop_mu);
    let trailing_stop_out = unsafe {
        std::slice::from_raw_parts_mut(trailing_stop_guard.as_mut_ptr() as *mut f64, total)
    };

    let mut long_stop_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut long_stop_mu, cols, &zero_prefixes);
    let mut long_stop_guard = ManuallyDrop::new(long_stop_mu);
    let long_stop_out =
        unsafe { std::slice::from_raw_parts_mut(long_stop_guard.as_mut_ptr() as *mut f64, total) };

    let mut short_stop_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut short_stop_mu, cols, &zero_prefixes);
    let mut short_stop_guard = ManuallyDrop::new(short_stop_mu);
    let short_stop_out =
        unsafe { std::slice::from_raw_parts_mut(short_stop_guard.as_mut_ptr() as *mut f64, total) };

    let mut direction_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut direction_mu, cols, &zero_prefixes);
    let mut direction_guard = ManuallyDrop::new(direction_mu);
    let direction_out =
        unsafe { std::slice::from_raw_parts_mut(direction_guard.as_mut_ptr() as *mut f64, total) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let trailing_stop_ptr = trailing_stop_out.as_mut_ptr() as usize;
            let long_stop_ptr = long_stop_out.as_mut_ptr() as usize;
            let short_stop_ptr = short_stop_out.as_mut_ptr() as usize;
            let direction_ptr = direction_out.as_mut_ptr() as usize;
            resolved
                .par_iter()
                .enumerate()
                .for_each(|(row, params)| unsafe {
                    let start = row * cols;
                    fibonacci_trailing_stop_row_from_slices(
                        high,
                        low,
                        close,
                        *params,
                        std::slice::from_raw_parts_mut(
                            (trailing_stop_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (long_stop_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (short_stop_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (direction_ptr as *mut f64).add(start),
                            cols,
                        ),
                    );
                });
        }

        #[cfg(target_arch = "wasm32")]
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            fibonacci_trailing_stop_row_from_slices(
                high,
                low,
                close,
                *params,
                &mut trailing_stop_out[start..end],
                &mut long_stop_out[start..end],
                &mut short_stop_out[start..end],
                &mut direction_out[start..end],
            );
        }
    } else {
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            fibonacci_trailing_stop_row_from_slices(
                high,
                low,
                close,
                *params,
                &mut trailing_stop_out[start..end],
                &mut long_stop_out[start..end],
                &mut short_stop_out[start..end],
                &mut direction_out[start..end],
            );
        }
    }

    let trailing_stop = unsafe {
        Vec::from_raw_parts(
            trailing_stop_guard.as_mut_ptr() as *mut f64,
            trailing_stop_guard.len(),
            trailing_stop_guard.capacity(),
        )
    };
    let long_stop = unsafe {
        Vec::from_raw_parts(
            long_stop_guard.as_mut_ptr() as *mut f64,
            long_stop_guard.len(),
            long_stop_guard.capacity(),
        )
    };
    let short_stop = unsafe {
        Vec::from_raw_parts(
            short_stop_guard.as_mut_ptr() as *mut f64,
            short_stop_guard.len(),
            short_stop_guard.capacity(),
        )
    };
    let direction = unsafe {
        Vec::from_raw_parts(
            direction_guard.as_mut_ptr() as *mut f64,
            direction_guard.len(),
            direction_guard.capacity(),
        )
    };
    core::mem::forget(trailing_stop_guard);
    core::mem::forget(long_stop_guard);
    core::mem::forget(short_stop_guard);
    core::mem::forget(direction_guard);

    Ok(FibonacciTrailingStopBatchOutput {
        trailing_stop,
        long_stop,
        short_stop,
        direction,
        combos,
        rows,
        cols,
    })
}

pub fn fibonacci_trailing_stop_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FibonacciTrailingStopBatchRange,
    kernel: Kernel,
    parallel: bool,
    trailing_stop: &mut [f64],
    long_stop: &mut [f64],
    short_stop: &mut [f64],
    direction: &mut [f64],
) -> Result<Vec<FibonacciTrailingStopParams>, FibonacciTrailingStopError> {
    let out = fibonacci_trailing_stop_batch_inner(high, low, close, sweep, kernel, parallel)?;
    let total = out.rows * out.cols;
    if trailing_stop.len() != total
        || long_stop.len() != total
        || short_stop.len() != total
        || direction.len() != total
    {
        return Err(FibonacciTrailingStopError::OutputLengthMismatch { expected: total });
    }
    trailing_stop.copy_from_slice(&out.trailing_stop);
    long_stop.copy_from_slice(&out.long_stop);
    short_stop.copy_from_slice(&out.short_stop);
    direction.copy_from_slice(&out.direction);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "fibonacci_trailing_stop")]
#[pyo3(signature = (
    high,
    low,
    close,
    left_bars=DEFAULT_LEFT_BARS,
    right_bars=DEFAULT_RIGHT_BARS,
    level=DEFAULT_LEVEL,
    trigger=DEFAULT_TRIGGER,
    kernel=None
))]
pub fn fibonacci_trailing_stop_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    left_bars: usize,
    right_bars: usize,
    level: f64,
    trigger: &str,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = FibonacciTrailingStopInput::from_slices(
        high,
        low,
        close,
        FibonacciTrailingStopParams {
            left_bars: Some(left_bars),
            right_bars: Some(right_bars),
            level: Some(level),
            trigger: Some(trigger.to_string()),
        },
    );
    let out = py
        .allow_threads(|| fibonacci_trailing_stop_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.trailing_stop.into_pyarray(py),
        out.long_stop.into_pyarray(py),
        out.short_stop.into_pyarray(py),
        out.direction.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "FibonacciTrailingStopStream")]
pub struct FibonacciTrailingStopStreamPy {
    stream: FibonacciTrailingStopStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl FibonacciTrailingStopStreamPy {
    #[new]
    #[pyo3(signature = (
        left_bars=DEFAULT_LEFT_BARS,
        right_bars=DEFAULT_RIGHT_BARS,
        level=DEFAULT_LEVEL,
        trigger=DEFAULT_TRIGGER
    ))]
    fn new(left_bars: usize, right_bars: usize, level: f64, trigger: &str) -> PyResult<Self> {
        let stream = FibonacciTrailingStopStream::try_new(FibonacciTrailingStopParams {
            left_bars: Some(left_bars),
            right_bars: Some(right_bars),
            level: Some(level),
            trigger: Some(trigger.to_string()),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(high, low, close).map(|point| {
            (
                point.trailing_stop,
                point.long_stop,
                point.short_stop,
                point.direction,
            )
        })
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "fibonacci_trailing_stop_batch")]
#[pyo3(signature = (
    high,
    low,
    close,
    left_bars_range=(DEFAULT_LEFT_BARS, DEFAULT_LEFT_BARS, 0),
    right_bars_range=(DEFAULT_RIGHT_BARS, DEFAULT_RIGHT_BARS, 0),
    level_range=(DEFAULT_LEVEL, DEFAULT_LEVEL, 0.0),
    trigger=DEFAULT_TRIGGER,
    kernel=None
))]
pub fn fibonacci_trailing_stop_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    left_bars_range: (usize, usize, usize),
    right_bars_range: (usize, usize, usize),
    level_range: (f64, f64, f64),
    trigger: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = FibonacciTrailingStopBatchRange {
        left_bars: left_bars_range,
        right_bars: right_bars_range,
        level: level_range,
        trigger: Some(trigger.to_string()),
    };
    let combos = expand_grid_fibonacci_trailing_stop(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let trailing_stop_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let long_stop_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let short_stop_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let direction_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let trailing_stop_slice = unsafe { trailing_stop_arr.as_slice_mut()? };
    let long_stop_slice = unsafe { long_stop_arr.as_slice_mut()? };
    let short_stop_slice = unsafe { short_stop_arr.as_slice_mut()? };
    let direction_slice = unsafe { direction_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch_kernel = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            fibonacci_trailing_stop_batch_inner_into(
                high,
                low,
                close,
                &sweep,
                batch_kernel.to_non_batch(),
                true,
                trailing_stop_slice,
                long_stop_slice,
                short_stop_slice,
                direction_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("trailing_stop", trailing_stop_arr.reshape((rows, cols))?)?;
    dict.set_item("long_stop", long_stop_arr.reshape((rows, cols))?)?;
    dict.set_item("short_stop", short_stop_arr.reshape((rows, cols))?)?;
    dict.set_item("direction", direction_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "left_bars",
        combos
            .iter()
            .map(|combo| combo.left_bars.unwrap_or(DEFAULT_LEFT_BARS) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "right_bars",
        combos
            .iter()
            .map(|combo| combo.right_bars.unwrap_or(DEFAULT_RIGHT_BARS) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "levels",
        combos
            .iter()
            .map(|combo| combo.level.unwrap_or(DEFAULT_LEVEL))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_fibonacci_trailing_stop_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fibonacci_trailing_stop_py, module)?)?;
    module.add_function(wrap_pyfunction!(fibonacci_trailing_stop_batch_py, module)?)?;
    module.add_class::<FibonacciTrailingStopStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct FibonacciTrailingStopJsOutput {
    pub trailing_stop: Vec<f64>,
    pub long_stop: Vec<f64>,
    pub short_stop: Vec<f64>,
    pub direction: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "fibonacci_trailing_stop_js")]
pub fn fibonacci_trailing_stop_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    left_bars: usize,
    right_bars: usize,
    level: f64,
    trigger: String,
) -> Result<JsValue, JsValue> {
    let input = FibonacciTrailingStopInput::from_slices(
        high,
        low,
        close,
        FibonacciTrailingStopParams {
            left_bars: Some(left_bars),
            right_bars: Some(right_bars),
            level: Some(level),
            trigger: Some(trigger),
        },
    );
    let out = fibonacci_trailing_stop(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&FibonacciTrailingStopJsOutput {
        trailing_stop: out.trailing_stop,
        long_stop: out.long_stop,
        short_stop: out.short_stop,
        direction: out.direction,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_trailing_stop_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_trailing_stop_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn has_duplicate_ptrs(ptrs: &[usize]) -> bool {
    for i in 0..ptrs.len() {
        for j in (i + 1)..ptrs.len() {
            if ptrs[i] == ptrs[j] {
                return true;
            }
        }
    }
    false
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_trailing_stop_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    trailing_stop_ptr: *mut f64,
    long_stop_ptr: *mut f64,
    short_stop_ptr: *mut f64,
    direction_ptr: *mut f64,
    len: usize,
    left_bars: usize,
    right_bars: usize,
    level: f64,
    trigger: String,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || trailing_stop_ptr.is_null()
        || long_stop_ptr.is_null()
        || short_stop_ptr.is_null()
        || direction_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = FibonacciTrailingStopInput::from_slices(
            high,
            low,
            close,
            FibonacciTrailingStopParams {
                left_bars: Some(left_bars),
                right_bars: Some(right_bars),
                level: Some(level),
                trigger: Some(trigger),
            },
        );

        let output_ptrs = [
            trailing_stop_ptr as usize,
            long_stop_ptr as usize,
            short_stop_ptr as usize,
            direction_ptr as usize,
        ];
        let need_temp = output_ptrs.iter().any(|&ptr| {
            ptr == high_ptr as usize || ptr == low_ptr as usize || ptr == close_ptr as usize
        }) || has_duplicate_ptrs(&output_ptrs);

        if need_temp {
            let mut trailing_stop = vec![0.0; len];
            let mut long_stop = vec![0.0; len];
            let mut short_stop = vec![0.0; len];
            let mut direction = vec![0.0; len];
            fibonacci_trailing_stop_into_slices(
                &mut trailing_stop,
                &mut long_stop,
                &mut short_stop,
                &mut direction,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(trailing_stop_ptr, len).copy_from_slice(&trailing_stop);
            std::slice::from_raw_parts_mut(long_stop_ptr, len).copy_from_slice(&long_stop);
            std::slice::from_raw_parts_mut(short_stop_ptr, len).copy_from_slice(&short_stop);
            std::slice::from_raw_parts_mut(direction_ptr, len).copy_from_slice(&direction);
        } else {
            fibonacci_trailing_stop_into_slices(
                std::slice::from_raw_parts_mut(trailing_stop_ptr, len),
                std::slice::from_raw_parts_mut(long_stop_ptr, len),
                std::slice::from_raw_parts_mut(short_stop_ptr, len),
                std::slice::from_raw_parts_mut(direction_ptr, len),
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
pub struct FibonacciTrailingStopBatchJsConfig {
    pub left_bars_range: Option<(usize, usize, usize)>,
    pub right_bars_range: Option<(usize, usize, usize)>,
    pub level_range: Option<(f64, f64, f64)>,
    pub trigger: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct FibonacciTrailingStopBatchJsOutput {
    pub trailing_stop: Vec<f64>,
    pub long_stop: Vec<f64>,
    pub short_stop: Vec<f64>,
    pub direction: Vec<f64>,
    pub combos: Vec<FibonacciTrailingStopParams>,
    pub left_bars: Vec<usize>,
    pub right_bars: Vec<usize>,
    pub levels: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "fibonacci_trailing_stop_batch_js")]
pub fn fibonacci_trailing_stop_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: FibonacciTrailingStopBatchJsConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = FibonacciTrailingStopBatchRange {
        left_bars: config
            .left_bars_range
            .unwrap_or((DEFAULT_LEFT_BARS, DEFAULT_LEFT_BARS, 0)),
        right_bars: config
            .right_bars_range
            .unwrap_or((DEFAULT_RIGHT_BARS, DEFAULT_RIGHT_BARS, 0)),
        level: config
            .level_range
            .unwrap_or((DEFAULT_LEVEL, DEFAULT_LEVEL, 0.0)),
        trigger: config.trigger.or_else(|| Some(DEFAULT_TRIGGER.to_string())),
    };
    let out = fibonacci_trailing_stop_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&FibonacciTrailingStopBatchJsOutput {
        left_bars: out
            .combos
            .iter()
            .map(|combo| combo.left_bars.unwrap_or(DEFAULT_LEFT_BARS))
            .collect(),
        right_bars: out
            .combos
            .iter()
            .map(|combo| combo.right_bars.unwrap_or(DEFAULT_RIGHT_BARS))
            .collect(),
        levels: out
            .combos
            .iter()
            .map(|combo| combo.level.unwrap_or(DEFAULT_LEVEL))
            .collect(),
        trailing_stop: out.trailing_stop,
        long_stop: out.long_stop,
        short_stop: out.short_stop,
        direction: out.direction,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_trailing_stop_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    trailing_stop_ptr: *mut f64,
    long_stop_ptr: *mut f64,
    short_stop_ptr: *mut f64,
    direction_ptr: *mut f64,
    len: usize,
    left_bars_start: usize,
    left_bars_end: usize,
    left_bars_step: usize,
    right_bars_start: usize,
    right_bars_end: usize,
    right_bars_step: usize,
    level_start: f64,
    level_end: f64,
    level_step: f64,
    trigger: String,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || trailing_stop_ptr.is_null()
        || long_stop_ptr.is_null()
        || short_stop_ptr.is_null()
        || direction_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = FibonacciTrailingStopBatchRange {
        left_bars: (left_bars_start, left_bars_end, left_bars_step),
        right_bars: (right_bars_start, right_bars_end, right_bars_step),
        level: (level_start, level_end, level_step),
        trigger: Some(trigger),
    };

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let combos = expand_grid_fibonacci_trailing_stop(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

        let output_ptrs = [
            trailing_stop_ptr as usize,
            long_stop_ptr as usize,
            short_stop_ptr as usize,
            direction_ptr as usize,
        ];
        let need_temp = output_ptrs.iter().any(|&ptr| {
            ptr == high_ptr as usize || ptr == low_ptr as usize || ptr == close_ptr as usize
        }) || has_duplicate_ptrs(&output_ptrs);

        if need_temp {
            let mut trailing_stop = vec![0.0; total];
            let mut long_stop = vec![0.0; total];
            let mut short_stop = vec![0.0; total];
            let mut direction = vec![0.0; total];
            let rows = fibonacci_trailing_stop_batch_inner_into(
                high,
                low,
                close,
                &sweep,
                Kernel::Auto,
                false,
                &mut trailing_stop,
                &mut long_stop,
                &mut short_stop,
                &mut direction,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .len();
            std::slice::from_raw_parts_mut(trailing_stop_ptr, total)
                .copy_from_slice(&trailing_stop);
            std::slice::from_raw_parts_mut(long_stop_ptr, total).copy_from_slice(&long_stop);
            std::slice::from_raw_parts_mut(short_stop_ptr, total).copy_from_slice(&short_stop);
            std::slice::from_raw_parts_mut(direction_ptr, total).copy_from_slice(&direction);
            Ok(rows)
        } else {
            let rows = fibonacci_trailing_stop_batch_inner_into(
                high,
                low,
                close,
                &sweep,
                Kernel::Auto,
                false,
                std::slice::from_raw_parts_mut(trailing_stop_ptr, total),
                std::slice::from_raw_parts_mut(long_stop_ptr, total),
                std::slice::from_raw_parts_mut(short_stop_ptr, total),
                std::slice::from_raw_parts_mut(direction_ptr, total),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .len();
            Ok(rows)
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_trailing_stop_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    left_bars: usize,
    right_bars: usize,
    level: f64,
    trigger: String,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        fibonacci_trailing_stop_js(high, low, close, left_bars, right_bars, level, trigger)?;
    crate::write_wasm_object_f64_outputs("fibonacci_trailing_stop_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_trailing_stop_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = fibonacci_trailing_stop_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "fibonacci_trailing_stop_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    fn sample_candles(length: usize) -> Candles {
        let close = (0..length)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.03 + (x * 0.18).sin() * 4.0 + (x * 0.051).cos() * 1.2
            })
            .collect::<Vec<_>>();
        let open = close.iter().map(|v| v - 0.25).collect::<Vec<_>>();
        let high = close
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.9 + (i as f64 * 0.07).cos().abs() * 0.3)
            .collect::<Vec<_>>();
        let low = close
            .iter()
            .enumerate()
            .map(|(i, v)| v - 0.85 - (i as f64 * 0.05).sin().abs() * 0.25)
            .collect::<Vec<_>>();
        let volume = vec![1_000.0; length];
        Candles::new((0..length as i64).collect(), open, high, low, close, volume)
    }

    fn assert_series_eq(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&lhs, &rhs) in left.iter().zip(right.iter()) {
            assert!(
                (lhs.is_nan() && rhs.is_nan()) || (lhs - rhs).abs() <= tol,
                "series mismatch: left={lhs:?}, right={rhs:?}"
            );
        }
    }

    #[test]
    fn fibonacci_trailing_stop_output_contract() {
        let candles = sample_candles(320);
        let out =
            fibonacci_trailing_stop(&FibonacciTrailingStopInput::with_default_candles(&candles))
                .unwrap();
        assert_eq!(out.trailing_stop.len(), candles.close.len());
        assert_eq!(out.long_stop.len(), candles.close.len());
        assert_eq!(out.short_stop.len(), candles.close.len());
        assert_eq!(out.direction.len(), candles.close.len());
        assert!(out.trailing_stop[0].is_finite());
        assert!(out.direction.iter().filter(|v| v.is_finite()).all(|v| {
            (*v + 1.0).abs() <= FLOAT_TOL || v.abs() <= FLOAT_TOL || (*v - 1.0).abs() <= FLOAT_TOL
        }));
    }

    #[test]
    fn fibonacci_trailing_stop_rejects_invalid_params() {
        let candles = sample_candles(16);
        let err = fibonacci_trailing_stop(&FibonacciTrailingStopInput::from_candles(
            &candles,
            FibonacciTrailingStopParams {
                left_bars: Some(0),
                right_bars: Some(1),
                level: Some(DEFAULT_LEVEL),
                trigger: Some(DEFAULT_TRIGGER.to_string()),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            FibonacciTrailingStopError::InvalidLeftBars { .. }
        ));

        let err = fibonacci_trailing_stop(&FibonacciTrailingStopInput::from_candles(
            &candles,
            FibonacciTrailingStopParams {
                left_bars: Some(4),
                right_bars: Some(1),
                level: Some(f64::NAN),
                trigger: Some(DEFAULT_TRIGGER.to_string()),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            FibonacciTrailingStopError::InvalidLevel { .. }
        ));
    }

    #[test]
    fn fibonacci_trailing_stop_builder_matches_direct() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(320);
        let direct = fibonacci_trailing_stop(&FibonacciTrailingStopInput::from_candles(
            &candles,
            FibonacciTrailingStopParams {
                left_bars: Some(12),
                right_bars: Some(2),
                level: Some(-0.236),
                trigger: Some("wick".to_string()),
            },
        ))?;
        let built = FibonacciTrailingStopBuilder::new()
            .left_bars(12)
            .right_bars(2)
            .level(-0.236)
            .trigger("wick")?
            .apply(&candles)?;

        assert_series_eq(&built.trailing_stop, &direct.trailing_stop, 1e-12);
        assert_series_eq(&built.long_stop, &direct.long_stop, 1e-12);
        assert_series_eq(&built.short_stop, &direct.short_stop, 1e-12);
        assert_series_eq(&built.direction, &direct.direction, 1e-12);
        Ok(())
    }

    #[test]
    fn fibonacci_trailing_stop_stream_matches_batch_with_reset() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(240);
        let mut high = candles.high.clone();
        let mut low = candles.low.clone();
        let mut close = candles.close.clone();
        high[120] = f64::NAN;
        low[120] = f64::NAN;
        close[120] = f64::NAN;

        let batch = fibonacci_trailing_stop(&FibonacciTrailingStopInput::from_slices(
            &high,
            &low,
            &close,
            FibonacciTrailingStopParams {
                left_bars: Some(10),
                right_bars: Some(2),
                level: Some(-0.382),
                trigger: Some("close".to_string()),
            },
        ))?;

        let mut stream = FibonacciTrailingStopBuilder::new()
            .left_bars(10)
            .right_bars(2)
            .level(-0.382)
            .into_stream()?;

        let mut trailing_stop = Vec::with_capacity(close.len());
        let mut long_stop = Vec::with_capacity(close.len());
        let mut short_stop = Vec::with_capacity(close.len());
        let mut direction = Vec::with_capacity(close.len());

        for i in 0..close.len() {
            match stream.update(high[i], low[i], close[i]) {
                Some(point) => {
                    trailing_stop.push(point.trailing_stop);
                    long_stop.push(point.long_stop);
                    short_stop.push(point.short_stop);
                    direction.push(point.direction);
                }
                None => {
                    trailing_stop.push(f64::NAN);
                    long_stop.push(f64::NAN);
                    short_stop.push(f64::NAN);
                    direction.push(f64::NAN);
                }
            }
        }

        assert_series_eq(&trailing_stop, &batch.trailing_stop, 1e-12);
        assert_series_eq(&long_stop, &batch.long_stop, 1e-12);
        assert_series_eq(&short_stop, &batch.short_stop, 1e-12);
        assert_series_eq(&direction, &batch.direction, 1e-12);
        Ok(())
    }

    #[test]
    fn fibonacci_trailing_stop_into_matches_main_api() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(192);
        let input = FibonacciTrailingStopInput::from_candles(
            &candles,
            FibonacciTrailingStopParams {
                left_bars: Some(14),
                right_bars: Some(1),
                level: Some(-0.382),
                trigger: Some("close".to_string()),
            },
        );
        let direct = fibonacci_trailing_stop(&input)?;
        let mut trailing_stop = vec![f64::NAN; candles.close.len()];
        let mut long_stop = vec![f64::NAN; candles.close.len()];
        let mut short_stop = vec![f64::NAN; candles.close.len()];
        let mut direction = vec![f64::NAN; candles.close.len()];

        fibonacci_trailing_stop_into_slices(
            &mut trailing_stop,
            &mut long_stop,
            &mut short_stop,
            &mut direction,
            &input,
            Kernel::Auto,
        )?;

        assert_series_eq(&trailing_stop, &direct.trailing_stop, 1e-12);
        assert_series_eq(&long_stop, &direct.long_stop, 1e-12);
        assert_series_eq(&short_stop, &direct.short_stop, 1e-12);
        assert_series_eq(&direction, &direct.direction, 1e-12);
        Ok(())
    }

    #[test]
    fn fibonacci_trailing_stop_batch_single_param_matches_single() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(200);
        let batch = FibonacciTrailingStopBatchBuilder::new()
            .left_bars_range(12, 12, 0)
            .right_bars_range(2, 2, 0)
            .level_range(-0.236, -0.236, 0.0)
            .trigger("wick")
            .apply_candles(&candles)?;
        let single = fibonacci_trailing_stop(&FibonacciTrailingStopInput::from_candles(
            &candles,
            FibonacciTrailingStopParams {
                left_bars: Some(12),
                right_bars: Some(2),
                level: Some(-0.236),
                trigger: Some("wick".to_string()),
            },
        ))?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, candles.close.len());
        assert_eq!(batch.combos.len(), 1);
        assert_series_eq(
            &batch.trailing_stop[..batch.cols],
            &single.trailing_stop,
            1e-12,
        );
        assert_series_eq(&batch.long_stop[..batch.cols], &single.long_stop, 1e-12);
        assert_series_eq(&batch.short_stop[..batch.cols], &single.short_stop, 1e-12);
        assert_series_eq(&batch.direction[..batch.cols], &single.direction, 1e-12);
        Ok(())
    }

    #[test]
    fn fibonacci_trailing_stop_batch_metadata() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(180);
        let out = FibonacciTrailingStopBatchBuilder::new()
            .left_bars_range(10, 12, 2)
            .right_bars_range(1, 2, 1)
            .level_range(-0.382, -0.236, 0.146)
            .apply_candles(&candles)?;

        assert_eq!(out.rows, 8);
        assert_eq!(out.cols, candles.close.len());
        assert_eq!(out.trailing_stop.len(), out.rows * out.cols);
        assert_eq!(out.long_stop.len(), out.rows * out.cols);
        assert_eq!(out.short_stop.len(), out.rows * out.cols);
        assert_eq!(out.direction.len(), out.rows * out.cols);
        assert_eq!(out.combos.len(), out.rows);
        Ok(())
    }
}
