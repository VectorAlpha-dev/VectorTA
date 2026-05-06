#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn statistical_trailing_stop_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    data_length: usize,
    normalization_length: usize,
    base_level: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = statistical_trailing_stop_js(
        high,
        low,
        close,
        data_length,
        normalization_length,
        base_level,
    )?;
    crate::write_wasm_object_f64_outputs("statistical_trailing_stop_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn statistical_trailing_stop_batch_unified_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = statistical_trailing_stop_batch_unified_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "statistical_trailing_stop_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_DATA_LENGTH: usize = 10;
const DEFAULT_NORMALIZATION_LENGTH: usize = 100;
const DEFAULT_BASE_LEVEL: &str = "level2";
const MIN_DATA_LENGTH: usize = 1;
const MIN_NORMALIZATION_LENGTH: usize = 10;
const BIAS_BEARISH: u8 = 0;
const BIAS_BULLISH: u8 = 1;

#[inline(always)]
fn hlc3(high: f64, low: f64, close: f64) -> f64 {
    (high + low + close) / 3.0
}

#[inline(always)]
fn floor_positive(value: f64) -> f64 {
    if value > 0.0 {
        value
    } else {
        f64::MIN_POSITIVE
    }
}

#[inline(always)]
fn close_source(candles: &Candles) -> &[f64] {
    &candles.close
}

#[inline(always)]
fn high_source(candles: &Candles) -> &[f64] {
    &candles.high
}

#[inline(always)]
fn low_source(candles: &Candles) -> &[f64] {
    &candles.low
}

#[derive(Debug, Clone)]
pub enum StatisticalTrailingStopData<'a> {
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
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct StatisticalTrailingStopOutput {
    pub level: Vec<f64>,
    pub anchor: Vec<f64>,
    pub bias: Vec<f64>,
    pub changed: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct StatisticalTrailingStopParams {
    pub data_length: Option<usize>,
    pub normalization_length: Option<usize>,
    pub base_level: Option<String>,
}

impl Default for StatisticalTrailingStopParams {
    fn default() -> Self {
        Self {
            data_length: Some(DEFAULT_DATA_LENGTH),
            normalization_length: Some(DEFAULT_NORMALIZATION_LENGTH),
            base_level: Some(DEFAULT_BASE_LEVEL.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StatisticalTrailingStopInput<'a> {
    pub data: StatisticalTrailingStopData<'a>,
    pub params: StatisticalTrailingStopParams,
}

impl<'a> StatisticalTrailingStopInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles, params: StatisticalTrailingStopParams) -> Self {
        Self {
            data: StatisticalTrailingStopData::Candles { candles },
            params,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: StatisticalTrailingStopParams,
    ) -> Self {
        Self {
            data: StatisticalTrailingStopData::Slices { high, low, close },
            params,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, StatisticalTrailingStopParams::default())
    }

    #[inline(always)]
    pub fn get_data_length(&self) -> usize {
        self.params.data_length.unwrap_or(DEFAULT_DATA_LENGTH)
    }

    #[inline(always)]
    pub fn get_normalization_length(&self) -> usize {
        self.params
            .normalization_length
            .unwrap_or(DEFAULT_NORMALIZATION_LENGTH)
    }

    #[inline(always)]
    pub fn get_base_level(&self) -> &str {
        self.params
            .base_level
            .as_deref()
            .unwrap_or(DEFAULT_BASE_LEVEL)
    }

    #[inline(always)]
    fn as_hlc(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            StatisticalTrailingStopData::Candles { candles } => (
                high_source(candles),
                low_source(candles),
                close_source(candles),
            ),
            StatisticalTrailingStopData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

impl<'a> AsRef<[f64]> for StatisticalTrailingStopInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        self.as_hlc().2
    }
}

#[derive(Clone, Debug)]
pub struct StatisticalTrailingStopBuilder {
    data_length: Option<usize>,
    normalization_length: Option<usize>,
    base_level: Option<String>,
    kernel: Kernel,
}

impl Default for StatisticalTrailingStopBuilder {
    fn default() -> Self {
        Self {
            data_length: None,
            normalization_length: None,
            base_level: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StatisticalTrailingStopBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn data_length(mut self, value: usize) -> Self {
        self.data_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn normalization_length(mut self, value: usize) -> Self {
        self.normalization_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn base_level<S: Into<String>>(mut self, value: S) -> Self {
        self.base_level = Some(value.into());
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> StatisticalTrailingStopParams {
        StatisticalTrailingStopParams {
            data_length: self.data_length,
            normalization_length: self.normalization_length,
            base_level: self.base_level,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<StatisticalTrailingStopOutput, StatisticalTrailingStopError> {
        let kernel = self.kernel;
        let params = self.params();
        statistical_trailing_stop_with_kernel(
            &StatisticalTrailingStopInput::from_candles(candles, params),
            kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<StatisticalTrailingStopOutput, StatisticalTrailingStopError> {
        let kernel = self.kernel;
        let params = self.params();
        statistical_trailing_stop_with_kernel(
            &StatisticalTrailingStopInput::from_slices(high, low, close, params),
            kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<StatisticalTrailingStopStream, StatisticalTrailingStopError> {
        StatisticalTrailingStopStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum StatisticalTrailingStopError {
    #[error("statistical_trailing_stop: input data slice is empty.")]
    EmptyInputData,
    #[error("statistical_trailing_stop: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "statistical_trailing_stop: inconsistent data lengths - high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    DataLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "statistical_trailing_stop: invalid period: data_length = {data_length}, normalization_length = {normalization_length}, data length = {data_len}"
    )]
    InvalidPeriod {
        data_length: usize,
        normalization_length: usize,
        data_len: usize,
    },
    #[error(
        "statistical_trailing_stop: invalid base level: {base_level}. expected one of level0, level1, level2, level3"
    )]
    InvalidBaseLevel { base_level: String },
    #[error(
        "statistical_trailing_stop: not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "statistical_trailing_stop: output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "statistical_trailing_stop: invalid range for {axis}: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        axis: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("statistical_trailing_stop: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    data_length: usize,
    normalization_length: usize,
    base_level_index: usize,
    all_finite: bool,
}

#[derive(Clone, Debug)]
struct MonoDeque {
    idx: Vec<usize>,
    vals: Vec<f64>,
    head: usize,
    descending: bool,
}

impl MonoDeque {
    #[inline(always)]
    fn new(descending: bool, cap: usize) -> Self {
        Self {
            idx: Vec::with_capacity(cap),
            vals: Vec::with_capacity(cap),
            head: 0,
            descending,
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.idx.clear();
        self.vals.clear();
        self.head = 0;
    }

    #[inline(always)]
    fn push(&mut self, index: usize, value: f64) {
        while self.idx.len() > self.head {
            let last = *self.vals.last().unwrap();
            let remove = if self.descending {
                last <= value
            } else {
                last >= value
            };
            if !remove {
                break;
            }
            self.idx.pop();
            self.vals.pop();
        }
        self.idx.push(index);
        self.vals.push(value);
    }

    #[inline(always)]
    fn expire(&mut self, min_index: usize) {
        while self.head < self.idx.len() && self.idx[self.head] < min_index {
            self.head += 1;
        }
        if self.head > 64 && self.head * 2 >= self.idx.len() {
            self.idx.drain(..self.head);
            self.vals.drain(..self.head);
            self.head = 0;
        }
    }

    #[inline(always)]
    fn front_value(&self) -> f64 {
        self.vals[self.head]
    }
}

#[derive(Clone, Debug)]
struct RingHistory {
    values: Vec<f64>,
    head: usize,
    count: usize,
}

impl RingHistory {
    #[inline(always)]
    fn new(cap: usize) -> Self {
        Self {
            values: vec![0.0; cap.max(1)],
            head: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }

    #[inline(always)]
    fn push(&mut self, value: f64) {
        self.values[self.head] = value;
        self.head += 1;
        if self.head == self.values.len() {
            self.head = 0;
        }
        if self.count < self.values.len() {
            self.count += 1;
        }
    }

    #[inline(always)]
    fn oldest(&self) -> f64 {
        self.values[self.head]
    }
}

#[derive(Clone, Debug)]
struct RollingStats {
    ring: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
    sum_sq: f64,
    inv_len: f64,
}

impl RollingStats {
    #[inline(always)]
    fn new(length: usize) -> Self {
        let len = length.max(1);
        Self {
            ring: vec![0.0; len],
            head: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            inv_len: 1.0 / len as f64,
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    #[inline(always)]
    fn push(&mut self, value: f64) -> Option<(f64, f64)> {
        let value_sq = value * value;
        if self.count < self.ring.len() {
            self.ring[self.head] = value;
            self.head += 1;
            if self.head == self.ring.len() {
                self.head = 0;
            }
            self.count += 1;
            self.sum += value;
            self.sum_sq += value_sq;
        } else {
            let old = self.ring[self.head];
            self.ring[self.head] = value;
            self.head += 1;
            if self.head == self.ring.len() {
                self.head = 0;
            }
            self.sum += value - old;
            self.sum_sq += value_sq - old * old;
        }
        if self.count < self.ring.len() {
            return None;
        }
        let mean = self.sum * self.inv_len;
        let variance = (self.sum_sq * self.inv_len - mean * mean).max(0.0);
        Some((mean, variance.sqrt()))
    }
}

type StatisticalTrailingStopValues = (f64, f64, f64, f64);

#[derive(Clone, Debug)]
struct StatisticalTrailingStopState {
    data_length: usize,
    base_level_index: usize,
    valid_run: usize,
    max_high: MonoDeque,
    min_low: MonoDeque,
    close_history: RingHistory,
    stats: RollingStats,
    bias: u8,
    delta: f64,
    level: f64,
    extreme: f64,
    anchor: f64,
    index: usize,
}

impl StatisticalTrailingStopState {
    #[inline(always)]
    fn new(data_length: usize, normalization_length: usize, base_level_index: usize) -> Self {
        Self {
            data_length,
            base_level_index,
            valid_run: 0,
            max_high: MonoDeque::new(true, data_length + 2),
            min_low: MonoDeque::new(false, data_length + 2),
            close_history: RingHistory::new(data_length + 2),
            stats: RollingStats::new(normalization_length),
            bias: BIAS_BEARISH,
            delta: f64::NAN,
            level: f64::NAN,
            extreme: f64::NAN,
            anchor: f64::NAN,
            index: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.valid_run = 0;
        self.max_high.clear();
        self.min_low.clear();
        self.close_history.clear();
        self.stats.clear();
        self.bias = BIAS_BEARISH;
        self.delta = f64::NAN;
        self.level = f64::NAN;
        self.extreme = f64::NAN;
        self.anchor = f64::NAN;
        self.index = 0;
    }

    #[inline(always)]
    fn update(
        &mut self,
        index: usize,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<StatisticalTrailingStopValues> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.reset();
            return None;
        }

        self.update_finite(index, high, low, close)
    }

    #[inline(always)]
    fn update_finite(
        &mut self,
        index: usize,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<StatisticalTrailingStopValues> {
        self.valid_run += 1;
        self.max_high.push(index, high);
        self.min_low.push(index, low);
        let window_start = index + 1 - self.valid_run.min(self.data_length);
        self.max_high.expire(window_start);
        self.min_low.expire(window_start);
        self.close_history.push(close);

        if self.valid_run < self.data_length + 2 {
            return None;
        }

        let previous_close = self.close_history.oldest();
        let highest = self.max_high.front_value();
        let lowest = self.min_low.front_value();
        let tr = (highest - lowest)
            .max((highest - previous_close).abs())
            .max((lowest - previous_close).abs());
        let (mean, stdev) = self.stats.push(floor_positive(tr).ln())?;
        let delta = (mean + self.base_level_index as f64 * stdev).exp();
        self.delta = delta;

        let current_hlc3 = hlc3(high, low, close);
        if !self.level.is_finite() {
            self.level = if self.bias == BIAS_BEARISH {
                current_hlc3 + delta
            } else {
                (current_hlc3 - delta).max(0.0)
            };
        }

        if self.bias == BIAS_BEARISH {
            if self.extreme.is_finite() {
                self.extreme = self.extreme.min(low);
            }
            self.level = self.level.min(current_hlc3 + delta);
        } else {
            if self.extreme.is_finite() {
                self.extreme = self.extreme.max(high);
            }
            self.level = self.level.max((current_hlc3 - delta).max(0.0));
        }

        let triggered = (self.bias == BIAS_BEARISH && close >= self.level)
            || (self.bias == BIAS_BULLISH && close <= self.level);
        let mut changed = 0.0;

        if triggered {
            self.anchor = close;
            self.index = index;
            self.bias = if self.bias == BIAS_BEARISH {
                BIAS_BULLISH
            } else {
                BIAS_BEARISH
            };
            self.level = if self.bias == BIAS_BEARISH {
                current_hlc3 + delta
            } else {
                (current_hlc3 - delta).max(0.0)
            };
            self.extreme = if self.bias == BIAS_BEARISH { low } else { high };
            changed = 1.0;
        }

        Some((self.level, self.anchor, self.bias as f64, changed))
    }
}

#[derive(Clone, Debug)]
pub struct StatisticalTrailingStopStream {
    params: StatisticalTrailingStopParams,
    state: StatisticalTrailingStopState,
    index: usize,
}

impl StatisticalTrailingStopStream {
    #[inline(always)]
    pub fn try_new(
        params: StatisticalTrailingStopParams,
    ) -> Result<Self, StatisticalTrailingStopError> {
        let data_length = params.data_length.unwrap_or(DEFAULT_DATA_LENGTH);
        let normalization_length = params
            .normalization_length
            .unwrap_or(DEFAULT_NORMALIZATION_LENGTH);
        validate_periods(data_length, normalization_length, usize::MAX)?;
        let base_level_index =
            parse_base_level(params.base_level.as_deref().unwrap_or(DEFAULT_BASE_LEVEL))?;
        Ok(Self {
            state: StatisticalTrailingStopState::new(
                data_length,
                normalization_length,
                base_level_index,
            ),
            params,
            index: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        let output = self.state.update(self.index, high, low, close);
        self.index = self.index.saturating_add(1);
        output
    }

    #[inline(always)]
    pub fn params(&self) -> &StatisticalTrailingStopParams {
        &self.params
    }
}

#[derive(Clone, Debug)]
pub struct StatisticalTrailingStopBatchRange {
    pub data_length: (usize, usize, usize),
    pub normalization_length: (usize, usize, usize),
    pub base_level: (String, String, usize),
}

impl Default for StatisticalTrailingStopBatchRange {
    fn default() -> Self {
        Self {
            data_length: (DEFAULT_DATA_LENGTH, DEFAULT_DATA_LENGTH, 0),
            normalization_length: (
                DEFAULT_NORMALIZATION_LENGTH,
                DEFAULT_NORMALIZATION_LENGTH,
                0,
            ),
            base_level: (
                DEFAULT_BASE_LEVEL.to_string(),
                DEFAULT_BASE_LEVEL.to_string(),
                0,
            ),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StatisticalTrailingStopBatchBuilder {
    range: StatisticalTrailingStopBatchRange,
    kernel: Kernel,
}

#[derive(Clone, Debug)]
pub struct StatisticalTrailingStopBatchOutput {
    pub level: Vec<f64>,
    pub anchor: Vec<f64>,
    pub bias: Vec<f64>,
    pub changed: Vec<f64>,
    pub combos: Vec<StatisticalTrailingStopParams>,
    pub rows: usize,
    pub cols: usize,
}

impl StatisticalTrailingStopBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn data_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.data_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn normalization_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.normalization_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn base_level_range<S: Into<String>>(mut self, start: S, end: S, step: usize) -> Self {
        self.range.base_level = (start.into(), end.into(), step);
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<StatisticalTrailingStopBatchOutput, StatisticalTrailingStopError> {
        statistical_trailing_stop_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<StatisticalTrailingStopBatchOutput, StatisticalTrailingStopError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close)
    }
}

#[inline(always)]
fn validate_periods(
    data_length: usize,
    normalization_length: usize,
    data_len: usize,
) -> Result<(), StatisticalTrailingStopError> {
    if data_length < MIN_DATA_LENGTH
        || normalization_length < MIN_NORMALIZATION_LENGTH
        || data_length + normalization_length + 1 > data_len
    {
        return Err(StatisticalTrailingStopError::InvalidPeriod {
            data_length,
            normalization_length,
            data_len,
        });
    }
    Ok(())
}

#[inline(always)]
fn parse_base_level(value: &str) -> Result<usize, StatisticalTrailingStopError> {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '_', '-'], "");
    match normalized.as_str() {
        "level0" | "0" => Ok(0),
        "level1" | "1" => Ok(1),
        "level2" | "2" => Ok(2),
        "level3" | "3" => Ok(3),
        _ => Err(StatisticalTrailingStopError::InvalidBaseLevel {
            base_level: value.to_string(),
        }),
    }
}

#[inline(always)]
fn canonical_base_level(value: &str) -> Result<&'static str, StatisticalTrailingStopError> {
    match parse_base_level(value)? {
        0 => Ok("level0"),
        1 => Ok("level1"),
        2 => Ok("level2"),
        _ => Ok("level3"),
    }
}

#[inline(always)]
fn analyze_valid_segments(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(usize, usize), StatisticalTrailingStopError> {
    let mut first_valid = None;
    let mut current = 0usize;
    let mut max_run = 0usize;
    for i in 0..close.len() {
        let valid = high[i].is_finite() && low[i].is_finite() && close[i].is_finite();
        if valid {
            if first_valid.is_none() {
                first_valid = Some(i);
            }
            current += 1;
            max_run = max_run.max(current);
        } else {
            current = 0;
        }
    }
    let first_valid = first_valid.ok_or(StatisticalTrailingStopError::AllValuesNaN)?;
    Ok((first_valid, max_run))
}

fn prepare_input<'a>(
    input: &'a StatisticalTrailingStopInput<'a>,
    _kernel: Kernel,
) -> Result<PreparedInput<'a>, StatisticalTrailingStopError> {
    let (high, low, close) = input.as_hlc();
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(StatisticalTrailingStopError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(StatisticalTrailingStopError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let data_length = input.get_data_length();
    let normalization_length = input.get_normalization_length();
    validate_periods(data_length, normalization_length, close.len())?;
    let base_level_index = parse_base_level(input.get_base_level())?;
    let (first_valid, max_run) = analyze_valid_segments(high, low, close)?;
    let needed = data_length + normalization_length + 1;
    if max_run < needed {
        return Err(StatisticalTrailingStopError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }

    Ok(PreparedInput {
        high,
        low,
        close,
        data_length,
        normalization_length,
        base_level_index,
        all_finite: first_valid == 0 && max_run == close.len(),
    })
}

#[inline(always)]
fn compute_row(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    data_length: usize,
    normalization_length: usize,
    base_level_index: usize,
    all_finite: bool,
    level_out: &mut [f64],
    anchor_out: &mut [f64],
    bias_out: &mut [f64],
    changed_out: &mut [f64],
) -> Result<(), StatisticalTrailingStopError> {
    let len = close.len();
    let expected = len;
    if level_out.len() != expected
        || anchor_out.len() != expected
        || bias_out.len() != expected
        || changed_out.len() != expected
    {
        return Err(StatisticalTrailingStopError::OutputLengthMismatch {
            expected,
            got: level_out
                .len()
                .max(anchor_out.len())
                .max(bias_out.len())
                .max(changed_out.len()),
        });
    }

    let mut state =
        StatisticalTrailingStopState::new(data_length, normalization_length, base_level_index);

    if all_finite {
        for i in 0..len {
            write_statistical_trailing_stop_values(
                i,
                state.update_finite(i, high[i], low[i], close[i]),
                level_out,
                anchor_out,
                bias_out,
                changed_out,
            );
        }
        return Ok(());
    }

    for i in 0..len {
        write_statistical_trailing_stop_values(
            i,
            state.update(i, high[i], low[i], close[i]),
            level_out,
            anchor_out,
            bias_out,
            changed_out,
        );
    }
    Ok(())
}

#[inline(always)]
fn write_statistical_trailing_stop_values(
    i: usize,
    values: Option<StatisticalTrailingStopValues>,
    level_out: &mut [f64],
    anchor_out: &mut [f64],
    bias_out: &mut [f64],
    changed_out: &mut [f64],
) {
    if let Some((level, anchor, bias, changed)) = values {
        level_out[i] = level;
        anchor_out[i] = anchor;
        bias_out[i] = bias;
        changed_out[i] = changed;
    } else {
        level_out[i] = f64::NAN;
        anchor_out[i] = f64::NAN;
        bias_out[i] = f64::NAN;
        changed_out[i] = f64::NAN;
    }
}

#[inline]
pub fn statistical_trailing_stop(
    input: &StatisticalTrailingStopInput,
) -> Result<StatisticalTrailingStopOutput, StatisticalTrailingStopError> {
    statistical_trailing_stop_with_kernel(input, Kernel::Auto)
}

pub fn statistical_trailing_stop_with_kernel(
    input: &StatisticalTrailingStopInput,
    kernel: Kernel,
) -> Result<StatisticalTrailingStopOutput, StatisticalTrailingStopError> {
    let prepared = prepare_input(input, kernel)?;
    let len = prepared.close.len();
    let mut level = alloc_uninit_f64(len);
    let mut anchor = alloc_uninit_f64(len);
    let mut bias = alloc_uninit_f64(len);
    let mut changed = alloc_uninit_f64(len);
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.data_length,
        prepared.normalization_length,
        prepared.base_level_index,
        prepared.all_finite,
        &mut level,
        &mut anchor,
        &mut bias,
        &mut changed,
    )?;
    Ok(StatisticalTrailingStopOutput {
        level,
        anchor,
        bias,
        changed,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn statistical_trailing_stop_into(
    level_out: &mut [f64],
    anchor_out: &mut [f64],
    bias_out: &mut [f64],
    changed_out: &mut [f64],
    input: &StatisticalTrailingStopInput,
) -> Result<(), StatisticalTrailingStopError> {
    statistical_trailing_stop_into_slice(
        level_out,
        anchor_out,
        bias_out,
        changed_out,
        input,
        Kernel::Auto,
    )
}

pub fn statistical_trailing_stop_into_slice(
    level_out: &mut [f64],
    anchor_out: &mut [f64],
    bias_out: &mut [f64],
    changed_out: &mut [f64],
    input: &StatisticalTrailingStopInput,
    kernel: Kernel,
) -> Result<(), StatisticalTrailingStopError> {
    let prepared = prepare_input(input, kernel)?;
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.data_length,
        prepared.normalization_length,
        prepared.base_level_index,
        prepared.all_finite,
        level_out,
        anchor_out,
        bias_out,
        changed_out,
    )
}

#[inline(always)]
pub fn expand_grid(
    sweep: &StatisticalTrailingStopBatchRange,
) -> Result<Vec<StatisticalTrailingStopParams>, StatisticalTrailingStopError> {
    fn axis_usize(
        axis: &'static str,
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, StatisticalTrailingStopError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let mut value = start;
            let stride = step.max(1);
            while value <= end {
                out.push(value);
                match value.checked_add(stride) {
                    Some(next) => value = next,
                    None => break,
                }
            }
        } else {
            let mut value = start as isize;
            let stop = end as isize;
            let stride = step.max(1) as isize;
            while value >= stop {
                out.push(value as usize);
                value -= stride;
            }
        }
        if out.is_empty() {
            return Err(StatisticalTrailingStopError::InvalidRange {
                axis,
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    fn axis_base_level(
        axis: &'static str,
        (start, end, step): (String, String, usize),
    ) -> Result<Vec<String>, StatisticalTrailingStopError> {
        let start_idx = parse_base_level(&start)?;
        let end_idx = parse_base_level(&end)?;
        if step == 0 || start_idx == end_idx {
            return Ok(vec![canonical_base_level(&start)?.to_string()]);
        }
        let mut out = Vec::new();
        if start_idx < end_idx {
            let mut idx = start_idx;
            let stride = step.max(1);
            while idx <= end_idx {
                out.push(
                    match idx {
                        0 => "level0",
                        1 => "level1",
                        2 => "level2",
                        _ => "level3",
                    }
                    .to_string(),
                );
                match idx.checked_add(stride) {
                    Some(next) => idx = next,
                    None => break,
                }
            }
        } else {
            let mut idx = start_idx as isize;
            let stop = end_idx as isize;
            let stride = step.max(1) as isize;
            while idx >= stop {
                out.push(
                    match idx as usize {
                        0 => "level0",
                        1 => "level1",
                        2 => "level2",
                        _ => "level3",
                    }
                    .to_string(),
                );
                idx -= stride;
            }
        }
        if out.is_empty() {
            return Err(StatisticalTrailingStopError::InvalidRange {
                axis,
                start,
                end,
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let data_lengths = axis_usize("data_length", sweep.data_length)?;
    let normalization_lengths = axis_usize("normalization_length", sweep.normalization_length)?;
    let base_levels = axis_base_level("base_level", sweep.base_level.clone())?;

    let cap = data_lengths
        .len()
        .checked_mul(normalization_lengths.len())
        .and_then(|v| v.checked_mul(base_levels.len()))
        .ok_or(StatisticalTrailingStopError::InvalidRange {
            axis: "grid",
            start: "cap".to_string(),
            end: "overflow".to_string(),
            step: "mul".to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &data_length in &data_lengths {
        for &normalization_length in &normalization_lengths {
            for base_level in &base_levels {
                out.push(StatisticalTrailingStopParams {
                    data_length: Some(data_length),
                    normalization_length: Some(normalization_length),
                    base_level: Some(base_level.clone()),
                });
            }
        }
    }
    Ok(out)
}

fn statistical_trailing_stop_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StatisticalTrailingStopBatchRange,
    parallel: bool,
    level_out: &mut [f64],
    anchor_out: &mut [f64],
    bias_out: &mut [f64],
    changed_out: &mut [f64],
) -> Result<Vec<StatisticalTrailingStopParams>, StatisticalTrailingStopError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(StatisticalTrailingStopError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(StatisticalTrailingStopError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or(StatisticalTrailingStopError::OutputLengthMismatch {
                expected: usize::MAX,
                got: level_out.len(),
            })?;
    if level_out.len() != expected
        || anchor_out.len() != expected
        || bias_out.len() != expected
        || changed_out.len() != expected
    {
        return Err(StatisticalTrailingStopError::OutputLengthMismatch {
            expected,
            got: level_out
                .len()
                .max(anchor_out.len())
                .max(bias_out.len())
                .max(changed_out.len()),
        });
    }
    let (_, max_run) = analyze_valid_segments(high, low, close)?;
    let all_finite = max_run == cols;

    for params in &combos {
        let data_length = params.data_length.unwrap_or(DEFAULT_DATA_LENGTH);
        let normalization_length = params
            .normalization_length
            .unwrap_or(DEFAULT_NORMALIZATION_LENGTH);
        validate_periods(data_length, normalization_length, cols)?;
        let needed = data_length + normalization_length + 1;
        if max_run < needed {
            return Err(StatisticalTrailingStopError::NotEnoughValidData {
                needed,
                valid: max_run,
            });
        }
        let _ = parse_base_level(params.base_level.as_deref().unwrap_or(DEFAULT_BASE_LEVEL))?;
    }

    let do_row = |row: usize,
                  level_row: &mut [f64],
                  anchor_row: &mut [f64],
                  bias_row: &mut [f64],
                  changed_row: &mut [f64]| {
        compute_row(
            high,
            low,
            close,
            combos[row].data_length.unwrap_or(DEFAULT_DATA_LENGTH),
            combos[row]
                .normalization_length
                .unwrap_or(DEFAULT_NORMALIZATION_LENGTH),
            parse_base_level(
                combos[row]
                    .base_level
                    .as_deref()
                    .unwrap_or(DEFAULT_BASE_LEVEL),
            )?,
            all_finite,
            level_row,
            anchor_row,
            bias_row,
            changed_row,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            level_out
                .par_chunks_mut(cols)
                .zip(anchor_out.par_chunks_mut(cols))
                .zip(bias_out.par_chunks_mut(cols))
                .zip(changed_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(
                    |(row, (((level_row, anchor_row), bias_row), changed_row))| {
                        do_row(row, level_row, anchor_row, bias_row, changed_row)
                    },
                )?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((level_row, anchor_row), bias_row), changed_row)) in level_out
                .chunks_mut(cols)
                .zip(anchor_out.chunks_mut(cols))
                .zip(bias_out.chunks_mut(cols))
                .zip(changed_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, level_row, anchor_row, bias_row, changed_row)?;
            }
        }
    } else {
        for (row, (((level_row, anchor_row), bias_row), changed_row)) in level_out
            .chunks_mut(cols)
            .zip(anchor_out.chunks_mut(cols))
            .zip(bias_out.chunks_mut(cols))
            .zip(changed_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, level_row, anchor_row, bias_row, changed_row)?;
        }
    }

    Ok(combos)
}

pub fn statistical_trailing_stop_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StatisticalTrailingStopBatchRange,
    kernel: Kernel,
) -> Result<StatisticalTrailingStopBatchOutput, StatisticalTrailingStopError> {
    match kernel {
        Kernel::Auto => {
            let _ = detect_best_batch_kernel();
        }
        k if !k.is_batch() => return Err(StatisticalTrailingStopError::InvalidKernelForBatch(k)),
        _ => {}
    }
    statistical_trailing_stop_batch_par_slice(high, low, close, sweep, Kernel::ScalarBatch)
}

pub fn statistical_trailing_stop_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StatisticalTrailingStopBatchRange,
    _kernel: Kernel,
) -> Result<StatisticalTrailingStopBatchOutput, StatisticalTrailingStopError> {
    statistical_trailing_stop_batch_impl(high, low, close, sweep, false)
}

pub fn statistical_trailing_stop_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StatisticalTrailingStopBatchRange,
    _kernel: Kernel,
) -> Result<StatisticalTrailingStopBatchOutput, StatisticalTrailingStopError> {
    statistical_trailing_stop_batch_impl(high, low, close, sweep, true)
}

fn statistical_trailing_stop_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StatisticalTrailingStopBatchRange,
    parallel: bool,
) -> Result<StatisticalTrailingStopBatchOutput, StatisticalTrailingStopError> {
    let rows = expand_grid(sweep)?.len();
    let cols = close.len();

    let mut level_mu = make_uninit_matrix(rows, cols);
    let mut anchor_mu = make_uninit_matrix(rows, cols);
    let mut bias_mu = make_uninit_matrix(rows, cols);
    let mut changed_mu = make_uninit_matrix(rows, cols);

    let mut level_guard = ManuallyDrop::new(level_mu);
    let mut anchor_guard = ManuallyDrop::new(anchor_mu);
    let mut bias_guard = ManuallyDrop::new(bias_mu);
    let mut changed_guard = ManuallyDrop::new(changed_mu);

    let level_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(level_guard.as_mut_ptr() as *mut f64, level_guard.len())
    };
    let anchor_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(anchor_guard.as_mut_ptr() as *mut f64, anchor_guard.len())
    };
    let bias_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(bias_guard.as_mut_ptr() as *mut f64, bias_guard.len())
    };
    let changed_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(changed_guard.as_mut_ptr() as *mut f64, changed_guard.len())
    };

    let combos = statistical_trailing_stop_batch_inner_into(
        high,
        low,
        close,
        sweep,
        parallel,
        level_out,
        anchor_out,
        bias_out,
        changed_out,
    )?;

    let level = unsafe {
        Vec::from_raw_parts(
            level_guard.as_mut_ptr() as *mut f64,
            level_guard.len(),
            level_guard.capacity(),
        )
    };
    let anchor = unsafe {
        Vec::from_raw_parts(
            anchor_guard.as_mut_ptr() as *mut f64,
            anchor_guard.len(),
            anchor_guard.capacity(),
        )
    };
    let bias = unsafe {
        Vec::from_raw_parts(
            bias_guard.as_mut_ptr() as *mut f64,
            bias_guard.len(),
            bias_guard.capacity(),
        )
    };
    let changed = unsafe {
        Vec::from_raw_parts(
            changed_guard.as_mut_ptr() as *mut f64,
            changed_guard.len(),
            changed_guard.capacity(),
        )
    };

    Ok(StatisticalTrailingStopBatchOutput {
        level,
        anchor,
        bias,
        changed,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "statistical_trailing_stop")]
#[pyo3(signature = (high, low, close, data_length=DEFAULT_DATA_LENGTH, normalization_length=DEFAULT_NORMALIZATION_LENGTH, base_level=DEFAULT_BASE_LEVEL, kernel=None))]
pub fn statistical_trailing_stop_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    data_length: usize,
    normalization_length: usize,
    base_level: &str,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = StatisticalTrailingStopInput::from_slices(
        high_slice,
        low_slice,
        close_slice,
        StatisticalTrailingStopParams {
            data_length: Some(data_length),
            normalization_length: Some(normalization_length),
            base_level: Some(base_level.to_string()),
        },
    );
    let output = py
        .allow_threads(|| statistical_trailing_stop_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.level.into_pyarray(py),
        output.anchor.into_pyarray(py),
        output.bias.into_pyarray(py),
        output.changed.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "statistical_trailing_stop_batch")]
#[pyo3(signature = (high, low, close, data_length_range=(DEFAULT_DATA_LENGTH, DEFAULT_DATA_LENGTH, 0), normalization_length_range=(DEFAULT_NORMALIZATION_LENGTH, DEFAULT_NORMALIZATION_LENGTH, 0), base_level_range=(DEFAULT_BASE_LEVEL.to_string(), DEFAULT_BASE_LEVEL.to_string(), 0usize), kernel=None))]
pub fn statistical_trailing_stop_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    data_length_range: (usize, usize, usize),
    normalization_length_range: (usize, usize, usize),
    base_level_range: (String, String, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = StatisticalTrailingStopBatchRange {
        data_length: data_length_range,
        normalization_length: normalization_length_range,
        base_level: base_level_range,
    };

    let rows = expand_grid(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .len();
    let cols = close_slice.len();
    let total = rows.checked_mul(cols).ok_or_else(|| {
        PyValueError::new_err("rows*cols overflow in statistical_trailing_stop_batch")
    })?;

    let level_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let anchor_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bias_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let changed_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let level_out = unsafe { level_arr.as_slice_mut()? };
    let anchor_out = unsafe { anchor_arr.as_slice_mut()? };
    let bias_out = unsafe { bias_arr.as_slice_mut()? };
    let changed_out = unsafe { changed_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            statistical_trailing_stop_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                !matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch),
                level_out,
                anchor_out,
                bias_out,
                changed_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("level", level_arr.reshape((rows, cols))?)?;
    dict.set_item("anchor", anchor_arr.reshape((rows, cols))?)?;
    dict.set_item("bias", bias_arr.reshape((rows, cols))?)?;
    dict.set_item("changed", changed_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "data_lengths",
        combos
            .iter()
            .map(|c| c.data_length.unwrap_or(DEFAULT_DATA_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "normalization_lengths",
        combos
            .iter()
            .map(|c| {
                c.normalization_length
                    .unwrap_or(DEFAULT_NORMALIZATION_LENGTH) as u64
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    let base_levels = PyList::empty(py);
    for combo in &combos {
        base_levels.append(combo.base_level.as_deref().unwrap_or(DEFAULT_BASE_LEVEL))?;
    }
    dict.set_item("base_levels", base_levels)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "StatisticalTrailingStopStream")]
pub struct StatisticalTrailingStopStreamPy {
    stream: StatisticalTrailingStopStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StatisticalTrailingStopStreamPy {
    #[new]
    #[pyo3(signature = (data_length=DEFAULT_DATA_LENGTH, normalization_length=DEFAULT_NORMALIZATION_LENGTH, base_level=DEFAULT_BASE_LEVEL))]
    fn new(data_length: usize, normalization_length: usize, base_level: &str) -> PyResult<Self> {
        let stream = StatisticalTrailingStopStream::try_new(StatisticalTrailingStopParams {
            data_length: Some(data_length),
            normalization_length: Some(normalization_length),
            base_level: Some(base_level.to_string()),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct StatisticalTrailingStopBatchConfig {
    pub data_length_range: (usize, usize, usize),
    pub normalization_length_range: (usize, usize, usize),
    pub base_level_range: (String, String, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct StatisticalTrailingStopBatchJsOutput {
    pub level: Vec<f64>,
    pub anchor: Vec<f64>,
    pub bias: Vec<f64>,
    pub changed: Vec<f64>,
    pub combos: Vec<StatisticalTrailingStopParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn statistical_trailing_stop_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    data_length: usize,
    normalization_length: usize,
    base_level: &str,
) -> Result<JsValue, JsValue> {
    let input = StatisticalTrailingStopInput::from_slices(
        high,
        low,
        close,
        StatisticalTrailingStopParams {
            data_length: Some(data_length),
            normalization_length: Some(normalization_length),
            base_level: Some(base_level.to_string()),
        },
    );
    let output = statistical_trailing_stop_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn statistical_trailing_stop_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn statistical_trailing_stop_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn statistical_trailing_stop_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    level_ptr: *mut f64,
    anchor_ptr: *mut f64,
    bias_ptr: *mut f64,
    changed_ptr: *mut f64,
    len: usize,
    data_length: usize,
    normalization_length: usize,
    base_level: &str,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || level_ptr.is_null()
        || anchor_ptr.is_null()
        || bias_ptr.is_null()
        || changed_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = StatisticalTrailingStopInput::from_slices(
            high,
            low,
            close,
            StatisticalTrailingStopParams {
                data_length: Some(data_length),
                normalization_length: Some(normalization_length),
                base_level: Some(base_level.to_string()),
            },
        );

        let aliased = [
            high_ptr as *const u8,
            low_ptr as *const u8,
            close_ptr as *const u8,
        ]
        .iter()
        .any(|&inp| {
            [
                level_ptr as *const u8,
                anchor_ptr as *const u8,
                bias_ptr as *const u8,
                changed_ptr as *const u8,
            ]
            .iter()
            .any(|&out| inp == out)
        }) || level_ptr == anchor_ptr
            || level_ptr == bias_ptr
            || level_ptr == changed_ptr
            || anchor_ptr == bias_ptr
            || anchor_ptr == changed_ptr
            || bias_ptr == changed_ptr;

        if aliased {
            let output = statistical_trailing_stop_with_kernel(&input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(level_ptr, len).copy_from_slice(&output.level);
            std::slice::from_raw_parts_mut(anchor_ptr, len).copy_from_slice(&output.anchor);
            std::slice::from_raw_parts_mut(bias_ptr, len).copy_from_slice(&output.bias);
            std::slice::from_raw_parts_mut(changed_ptr, len).copy_from_slice(&output.changed);
        } else {
            let level_out = std::slice::from_raw_parts_mut(level_ptr, len);
            let anchor_out = std::slice::from_raw_parts_mut(anchor_ptr, len);
            let bias_out = std::slice::from_raw_parts_mut(bias_ptr, len);
            let changed_out = std::slice::from_raw_parts_mut(changed_ptr, len);
            statistical_trailing_stop_into_slice(
                level_out,
                anchor_out,
                bias_out,
                changed_out,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = statistical_trailing_stop_batch)]
pub fn statistical_trailing_stop_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: StatisticalTrailingStopBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = StatisticalTrailingStopBatchRange {
        data_length: config.data_length_range,
        normalization_length: config.normalization_length_range,
        base_level: config.base_level_range,
    };
    let output =
        statistical_trailing_stop_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_output = StatisticalTrailingStopBatchJsOutput {
        level: output.level,
        anchor: output.anchor,
        bias: output.bias,
        changed: output.changed,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn statistical_trailing_stop_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    level_ptr: *mut f64,
    anchor_ptr: *mut f64,
    bias_ptr: *mut f64,
    changed_ptr: *mut f64,
    len: usize,
    data_length_start: usize,
    data_length_end: usize,
    data_length_step: usize,
    normalization_length_start: usize,
    normalization_length_end: usize,
    normalization_length_step: usize,
    base_level_start: &str,
    base_level_end: &str,
    base_level_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || level_ptr.is_null()
        || anchor_ptr.is_null()
        || bias_ptr.is_null()
        || changed_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let sweep = StatisticalTrailingStopBatchRange {
        data_length: (data_length_start, data_length_end, data_length_step),
        normalization_length: (
            normalization_length_start,
            normalization_length_end,
            normalization_length_step,
        ),
        base_level: (
            base_level_start.to_string(),
            base_level_end.to_string(),
            base_level_step,
        ),
    };
    let rows = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let level_out = std::slice::from_raw_parts_mut(level_ptr, total);
        let anchor_out = std::slice::from_raw_parts_mut(anchor_ptr, total);
        let bias_out = std::slice::from_raw_parts_mut(bias_ptr, total);
        let changed_out = std::slice::from_raw_parts_mut(changed_ptr, total);
        statistical_trailing_stop_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            false,
            level_out,
            anchor_out,
            bias_out,
            changed_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn trend_data(size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(size);
        let mut low = Vec::with_capacity(size);
        let mut close = Vec::with_capacity(size);
        for i in 0..size {
            let base = 100.0 + i as f64 * 0.9;
            high.push(base + 1.0 + (i % 3) as f64 * 0.1);
            low.push(base - 1.0 - (i % 2) as f64 * 0.1);
            close.push(base);
        }
        (high, low, close)
    }

    fn reversal_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close = vec![
            100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0, 95.5, 97.0, 99.0, 101.0, 104.0,
            108.0, 111.0, 113.0, 112.0, 111.0, 109.0, 107.0, 105.0, 103.0, 101.0, 99.0, 97.0, 95.0,
            94.0, 93.5, 93.0,
        ];
        let high = close.iter().map(|v| v + 0.8).collect::<Vec<_>>();
        let low = close.iter().map(|v| v - 0.8).collect::<Vec<_>>();
        (high, low, close)
    }

    fn arrays_eq_nan(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(x, y)| {
                (x.is_nan() && y.is_nan())
                    || (!x.is_nan() && !y.is_nan() && (*x - *y).abs() <= 1e-12)
            })
    }

    #[test]
    fn statistical_trailing_stop_into_matches_single() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = trend_data(160);
        let input = StatisticalTrailingStopInput::from_slices(
            &high,
            &low,
            &close,
            StatisticalTrailingStopParams::default(),
        );
        let single = statistical_trailing_stop(&input)?;

        let mut level = vec![0.0; close.len()];
        let mut anchor = vec![0.0; close.len()];
        let mut bias = vec![0.0; close.len()];
        let mut changed = vec![0.0; close.len()];
        statistical_trailing_stop_into_slice(
            &mut level,
            &mut anchor,
            &mut bias,
            &mut changed,
            &input,
            Kernel::Auto,
        )?;

        assert!(arrays_eq_nan(&single.level, &level));
        assert!(arrays_eq_nan(&single.anchor, &anchor));
        assert!(arrays_eq_nan(&single.bias, &bias));
        assert!(arrays_eq_nan(&single.changed, &changed));
        Ok(())
    }

    #[test]
    fn statistical_trailing_stop_stream_matches_batch() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = trend_data(170);
        let params = StatisticalTrailingStopParams::default();
        let input = StatisticalTrailingStopInput::from_slices(&high, &low, &close, params.clone());
        let batch = statistical_trailing_stop(&input)?;

        let mut stream = StatisticalTrailingStopStream::try_new(params)?;
        let mut level = Vec::with_capacity(close.len());
        let mut anchor = Vec::with_capacity(close.len());
        let mut bias = Vec::with_capacity(close.len());
        let mut changed = Vec::with_capacity(close.len());

        for i in 0..close.len() {
            if let Some((lvl, anc, bs, ch)) = stream.update(high[i], low[i], close[i]) {
                level.push(lvl);
                anchor.push(anc);
                bias.push(bs);
                changed.push(ch);
            } else {
                level.push(f64::NAN);
                anchor.push(f64::NAN);
                bias.push(f64::NAN);
                changed.push(f64::NAN);
            }
        }

        assert!(arrays_eq_nan(&batch.level, &level));
        assert!(arrays_eq_nan(&batch.anchor, &anchor));
        assert!(arrays_eq_nan(&batch.bias, &bias));
        assert!(arrays_eq_nan(&batch.changed, &changed));
        Ok(())
    }

    #[test]
    fn statistical_trailing_stop_reversal_behavior() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = reversal_data();
        let output = statistical_trailing_stop(&StatisticalTrailingStopInput::from_slices(
            &high,
            &low,
            &close,
            StatisticalTrailingStopParams {
                data_length: Some(3),
                normalization_length: Some(10),
                base_level: Some("level0".to_string()),
            },
        ))?;

        let changes = output
            .changed
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v == 1.0 { Some(i) } else { None })
            .collect::<Vec<_>>();
        assert!(!changes.is_empty());
        let first = changes[0];
        assert!(output.anchor[first].is_finite());
        assert_eq!(output.bias[first], 1.0);
        Ok(())
    }

    #[test]
    fn statistical_trailing_stop_nan_gap_restarts() -> Result<(), Box<dyn StdError>> {
        let (mut high, mut low, mut close) = trend_data(170);
        high[120] = f64::NAN;
        low[120] = f64::NAN;
        close[120] = f64::NAN;
        let output = statistical_trailing_stop(&StatisticalTrailingStopInput::from_slices(
            &high,
            &low,
            &close,
            StatisticalTrailingStopParams::default(),
        ))?;

        let restart_end =
            (120 + DEFAULT_DATA_LENGTH + DEFAULT_NORMALIZATION_LENGTH + 1).min(output.level.len());
        for i in 120..restart_end {
            assert!(output.level[i].is_nan());
            assert!(output.bias[i].is_nan());
        }
        Ok(())
    }

    #[test]
    fn statistical_trailing_stop_batch_matches_single() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = trend_data(170);
        let sweep = StatisticalTrailingStopBatchRange {
            data_length: (9, 10, 1),
            normalization_length: (10, 11, 1),
            base_level: ("level1".to_string(), "level2".to_string(), 1),
        };
        let batch = statistical_trailing_stop_batch_with_kernel(
            &high,
            &low,
            &close,
            &sweep,
            Kernel::ScalarBatch,
        )?;

        assert_eq!(batch.rows, 8);
        assert_eq!(batch.cols, close.len());
        for row in 0..batch.rows {
            let combo = &batch.combos[row];
            let single = statistical_trailing_stop(&StatisticalTrailingStopInput::from_slices(
                &high,
                &low,
                &close,
                combo.clone(),
            ))?;
            let start = row * batch.cols;
            let end = start + batch.cols;
            assert!(arrays_eq_nan(
                &batch.level[start..end],
                single.level.as_slice()
            ));
            assert!(arrays_eq_nan(
                &batch.anchor[start..end],
                single.anchor.as_slice()
            ));
            assert!(arrays_eq_nan(
                &batch.bias[start..end],
                single.bias.as_slice()
            ));
            assert!(arrays_eq_nan(
                &batch.changed[start..end],
                single.changed.as_slice()
            ));
        }
        Ok(())
    }

    #[test]
    fn statistical_trailing_stop_invalid_base_level_errors() {
        let (high, low, close) = trend_data(160);
        let input = StatisticalTrailingStopInput::from_slices(
            &high,
            &low,
            &close,
            StatisticalTrailingStopParams {
                data_length: Some(10),
                normalization_length: Some(100),
                base_level: Some("bad".to_string()),
            },
        );
        assert!(matches!(
            statistical_trailing_stop(&input),
            Err(StatisticalTrailingStopError::InvalidBaseLevel { .. })
        ));
    }

    #[test]
    fn statistical_trailing_stop_all_nan_errors() {
        let high = vec![f64::NAN; 200];
        let low = vec![f64::NAN; 200];
        let close = vec![f64::NAN; 200];
        let input = StatisticalTrailingStopInput::from_slices(
            &high,
            &low,
            &close,
            StatisticalTrailingStopParams::default(),
        );
        assert!(matches!(
            statistical_trailing_stop(&input),
            Err(StatisticalTrailingStopError::AllValuesNaN)
        ));
    }

    #[test]
    fn statistical_trailing_stop_default_candles_smoke() -> Result<(), Box<dyn StdError>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        let output = statistical_trailing_stop(
            &StatisticalTrailingStopInput::with_default_candles(&candles),
        )?;
        assert_eq!(output.level.len(), candles.close.len());
        assert_eq!(output.anchor.len(), candles.close.len());
        assert_eq!(output.bias.len(), candles.close.len());
        assert_eq!(output.changed.len(), candles.close.len());
        Ok(())
    }
}
