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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use thiserror::Error;

const DEFAULT_ENTRY_LEVEL: usize = 30;
const DEFAULT_EXIT_LEVEL: usize = 75;
const SMA13_WINDOW: usize = 13;
const RANGE75_WINDOW: usize = 75;
const OUTPUTS: usize = 6;

#[derive(Debug, Clone)]
pub enum CyberpunkValueTrendAnalyzerData<'a> {
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
pub struct CyberpunkValueTrendAnalyzerOutput {
    pub value_trend: Vec<f64>,
    pub value_trend_lag: Vec<f64>,
    pub deviation_index: Vec<f64>,
    pub overbought_signal: Vec<f64>,
    pub buy_signal: Vec<f64>,
    pub sell_signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CyberpunkValueTrendAnalyzerOutputField {
    ValueTrend,
    ValueTrendLag,
    DeviationIndex,
    OverboughtSignal,
    BuySignal,
    SellSignal,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct CyberpunkValueTrendAnalyzerParams {
    pub entry_level: Option<usize>,
    pub exit_level: Option<usize>,
}

impl Default for CyberpunkValueTrendAnalyzerParams {
    fn default() -> Self {
        Self {
            entry_level: Some(DEFAULT_ENTRY_LEVEL),
            exit_level: Some(DEFAULT_EXIT_LEVEL),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CyberpunkValueTrendAnalyzerInput<'a> {
    pub data: CyberpunkValueTrendAnalyzerData<'a>,
    pub params: CyberpunkValueTrendAnalyzerParams,
}

impl<'a> CyberpunkValueTrendAnalyzerInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: CyberpunkValueTrendAnalyzerParams) -> Self {
        Self {
            data: CyberpunkValueTrendAnalyzerData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: CyberpunkValueTrendAnalyzerParams,
    ) -> Self {
        Self {
            data: CyberpunkValueTrendAnalyzerData::Slices {
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
        Self::from_candles(candles, CyberpunkValueTrendAnalyzerParams::default())
    }

    #[inline]
    pub fn get_entry_level(&self) -> usize {
        self.params.entry_level.unwrap_or(DEFAULT_ENTRY_LEVEL)
    }

    #[inline]
    pub fn get_exit_level(&self) -> usize {
        self.params.exit_level.unwrap_or(DEFAULT_EXIT_LEVEL)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CyberpunkValueTrendAnalyzerBuilder {
    entry_level: Option<usize>,
    exit_level: Option<usize>,
    kernel: Kernel,
}

impl Default for CyberpunkValueTrendAnalyzerBuilder {
    fn default() -> Self {
        Self {
            entry_level: None,
            exit_level: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CyberpunkValueTrendAnalyzerBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn entry_level(mut self, value: usize) -> Self {
        self.entry_level = Some(value);
        self
    }

    #[inline(always)]
    pub fn exit_level(mut self, value: usize) -> Self {
        self.exit_level = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }
}

#[derive(Debug, Error)]
pub enum CyberpunkValueTrendAnalyzerError {
    #[error("cyberpunk_value_trend_analyzer: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "cyberpunk_value_trend_analyzer: Input length mismatch: open = {open_len}, high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    InputLengthMismatch {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("cyberpunk_value_trend_analyzer: All values are NaN.")]
    AllValuesNaN,
    #[error("cyberpunk_value_trend_analyzer: Invalid entry_level: {entry_level}")]
    InvalidEntryLevel { entry_level: usize },
    #[error("cyberpunk_value_trend_analyzer: Invalid exit_level: {exit_level}")]
    InvalidExitLevel { exit_level: usize },
    #[error(
        "cyberpunk_value_trend_analyzer: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "cyberpunk_value_trend_analyzer: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "cyberpunk_value_trend_analyzer: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error(
        "cyberpunk_value_trend_analyzer: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("cyberpunk_value_trend_analyzer: Invalid kernel: {0:?}")]
    InvalidKernel(Kernel),
    #[error("cyberpunk_value_trend_analyzer: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("cyberpunk_value_trend_analyzer: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
struct RollingSum {
    buf: [f64; SMA13_WINDOW],
    pos: usize,
    count: usize,
    sum: f64,
}

impl Default for RollingSum {
    fn default() -> Self {
        Self {
            buf: [0.0; SMA13_WINDOW],
            pos: 0,
            count: 0,
            sum: 0.0,
        }
    }
}

impl RollingSum {
    #[inline(always)]
    fn reset(&mut self) {
        self.pos = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn push(&mut self, value: f64) -> Option<f64> {
        if self.count < SMA13_WINDOW {
            self.buf[self.pos] = value;
            self.pos = (self.pos + 1) % SMA13_WINDOW;
            self.count += 1;
            self.sum += value;
            if self.count == SMA13_WINDOW {
                Some(self.sum / SMA13_WINDOW as f64)
            } else {
                None
            }
        } else {
            let old = self.buf[self.pos];
            self.buf[self.pos] = value;
            self.pos = (self.pos + 1) % SMA13_WINDOW;
            self.sum += value - old;
            Some(self.sum / SMA13_WINDOW as f64)
        }
    }
}

#[derive(Debug, Clone, Default)]
struct MonotonicQueue {
    data: VecDeque<(usize, f64)>,
}

impl MonotonicQueue {
    #[inline(always)]
    fn reset(&mut self) {
        self.data.clear();
    }

    #[inline(always)]
    fn push_min(&mut self, index: usize, value: f64) {
        while let Some((_, back)) = self.data.back() {
            if *back <= value {
                break;
            }
            self.data.pop_back();
        }
        self.data.push_back((index, value));
    }

    #[inline(always)]
    fn push_max(&mut self, index: usize, value: f64) {
        while let Some((_, back)) = self.data.back() {
            if *back >= value {
                break;
            }
            self.data.pop_back();
        }
        self.data.push_back((index, value));
    }

    #[inline(always)]
    fn prune(&mut self, min_index: usize) {
        while let Some((front_index, _)) = self.data.front() {
            if *front_index >= min_index {
                break;
            }
            self.data.pop_front();
        }
    }

    #[inline(always)]
    fn current(&self) -> Option<f64> {
        self.data.front().map(|(_, value)| *value)
    }
}

#[derive(Debug, Clone, Copy)]
struct WeightedSmaState {
    alpha: f64,
    value: Option<f64>,
}

impl WeightedSmaState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            alpha: 1.0 / length as f64,
            value: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.value = None;
    }

    #[inline(always)]
    fn update(&mut self, source: f64) -> f64 {
        if !source.is_finite() {
            self.value = None;
            return f64::NAN;
        }
        let next = match self.value {
            Some(prev) => self.alpha * source + (1.0 - self.alpha) * prev,
            None => source,
        };
        self.value = Some(next);
        next
    }
}

#[inline(always)]
fn is_valid_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
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
        if is_valid_ohlc(o, h, l, c) {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a CyberpunkValueTrendAnalyzerInput<'a>,
) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
    match &input.data {
        CyberpunkValueTrendAnalyzerData::Candles { candles } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        CyberpunkValueTrendAnalyzerData::Slices {
            open,
            high,
            low,
            close,
        } => (open, high, low, close),
    }
}

#[inline(always)]
fn validate_params_only(
    entry_level: usize,
    exit_level: usize,
) -> Result<(), CyberpunkValueTrendAnalyzerError> {
    if !(1..=100).contains(&entry_level) {
        return Err(CyberpunkValueTrendAnalyzerError::InvalidEntryLevel { entry_level });
    }
    if !(1..=100).contains(&exit_level) {
        return Err(CyberpunkValueTrendAnalyzerError::InvalidExitLevel { exit_level });
    }
    Ok(())
}

#[inline(always)]
fn validate_common(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    entry_level: usize,
    exit_level: usize,
) -> Result<(), CyberpunkValueTrendAnalyzerError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(CyberpunkValueTrendAnalyzerError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(CyberpunkValueTrendAnalyzerError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    validate_params_only(entry_level, exit_level)?;
    let longest = longest_valid_run(open, high, low, close);
    if longest == 0 {
        return Err(CyberpunkValueTrendAnalyzerError::AllValuesNaN);
    }
    if longest < RANGE75_WINDOW {
        return Err(CyberpunkValueTrendAnalyzerError::NotEnoughValidData {
            needed: RANGE75_WINDOW,
            valid: longest,
        });
    }
    Ok(())
}

#[inline(always)]
fn normalize_single_kernel(kernel: Kernel) -> Result<Kernel, CyberpunkValueTrendAnalyzerError> {
    match kernel {
        Kernel::Auto => Ok(detect_best_kernel()),
        Kernel::Scalar | Kernel::ScalarBatch => Ok(Kernel::Scalar),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => Ok(Kernel::Avx2),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => Ok(Kernel::Avx512),
        other => Err(CyberpunkValueTrendAnalyzerError::InvalidKernel(other)),
    }
}

#[inline(always)]
fn normalize_batch_kernel(kernel: Kernel) -> Result<Kernel, CyberpunkValueTrendAnalyzerError> {
    match kernel {
        Kernel::Auto => Ok(detect_best_batch_kernel()),
        Kernel::Scalar | Kernel::ScalarBatch => Ok(Kernel::ScalarBatch),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => Ok(Kernel::Avx2Batch),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => Ok(Kernel::Avx512Batch),
        other => Err(CyberpunkValueTrendAnalyzerError::InvalidKernelForBatch(
            other,
        )),
    }
}

#[inline(always)]
fn axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, CyberpunkValueTrendAnalyzerError> {
    validate_params_only(start, start)?;
    validate_params_only(end, end)?;
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(CyberpunkValueTrendAnalyzerError::InvalidRange {
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
        let next = current.checked_add(step).ok_or_else(|| {
            CyberpunkValueTrendAnalyzerError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            }
        })?;
        if next <= current {
            return Err(CyberpunkValueTrendAnalyzerError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current = next.min(end);
    }
    Ok(out)
}

#[inline(always)]
fn reset_core_state(
    sma13: &mut RollingSum,
    lowest75: &mut MonotonicQueue,
    highest75: &mut MonotonicQueue,
    close_norm_sma: &mut WeightedSmaState,
    smooth5: &mut WeightedSmaState,
    valid_run: &mut usize,
    prev_value_trend: &mut f64,
) {
    sma13.reset();
    lowest75.reset();
    highest75.reset();
    close_norm_sma.reset();
    smooth5.reset();
    *valid_run = 0;
    *prev_value_trend = f64::NAN;
}

#[inline(always)]
fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    entry_level: usize,
    exit_level: usize,
    out_value_trend: &mut [f64],
    out_value_trend_lag: &mut [f64],
    out_deviation_index: &mut [f64],
    out_overbought_signal: &mut [f64],
    out_buy_signal: &mut [f64],
    out_sell_signal: &mut [f64],
) {
    let entry_level = entry_level as f64;
    let exit_level = exit_level as f64;
    let mut sma13 = RollingSum::default();
    let mut lowest75 = MonotonicQueue::default();
    let mut highest75 = MonotonicQueue::default();
    let mut close_norm_sma = WeightedSmaState::new(20);
    let mut smooth5 = WeightedSmaState::new(5);
    let mut valid_run = 0usize;
    let mut prev_value_trend = f64::NAN;

    for i in 0..close.len() {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];

        if !is_valid_ohlc(o, h, l, c) {
            reset_core_state(
                &mut sma13,
                &mut lowest75,
                &mut highest75,
                &mut close_norm_sma,
                &mut smooth5,
                &mut valid_run,
                &mut prev_value_trend,
            );
            continue;
        }

        valid_run += 1;
        let ma13 = sma13.push(c);
        lowest75.push_min(i, l);
        highest75.push_max(i, h);
        let min_index = i.saturating_sub(RANGE75_WINDOW - 1);
        lowest75.prune(min_index);
        highest75.prune(min_index);

        if prev_value_trend.is_finite() {
            out_value_trend_lag[i] = prev_value_trend;
        }

        let mut current_value_trend = f64::NAN;
        if valid_run >= RANGE75_WINDOW {
            let range_low = lowest75.current().unwrap_or(f64::NAN);
            let range_high = highest75.current().unwrap_or(f64::NAN);
            let range = range_high - range_low;
            if range.is_finite() && range > 0.0 {
                let close_norm = (c - range_low) * 100.0 / range;
                let close_norm_avg = close_norm_sma.update(close_norm);
                let smooth = smooth5.update(close_norm_avg);
                if close_norm_avg.is_finite() && smooth.is_finite() {
                    current_value_trend = 3.0 * close_norm_avg - 2.0 * smooth;
                    out_value_trend[i] = current_value_trend;
                    out_buy_signal[i] = 0.0;
                    out_sell_signal[i] = 0.0;
                }
            } else {
                close_norm_sma.reset();
                smooth5.reset();
            }
        }

        if let Some(avg13) = ma13 {
            if avg13.is_finite() && avg13 != 0.0 {
                let deviation_index = 100.0 - (((c - avg13) / avg13) * 100.0).abs();
                out_deviation_index[i] = deviation_index;
                if current_value_trend.is_finite() && current_value_trend > deviation_index {
                    out_overbought_signal[i] = deviation_index;
                }
            }
        }

        if current_value_trend.is_finite() && prev_value_trend.is_finite() {
            if prev_value_trend <= entry_level && current_value_trend > entry_level {
                out_buy_signal[i] = 1.0;
            }
            if prev_value_trend >= exit_level && current_value_trend < exit_level {
                out_sell_signal[i] = 1.0;
            }
        }

        prev_value_trend = current_value_trend;
    }
}

pub fn cyberpunk_value_trend_analyzer(
    input: &CyberpunkValueTrendAnalyzerInput,
) -> Result<CyberpunkValueTrendAnalyzerOutput, CyberpunkValueTrendAnalyzerError> {
    cyberpunk_value_trend_analyzer_with_kernel(input, Kernel::Auto)
}

pub fn cyberpunk_value_trend_analyzer_with_kernel(
    input: &CyberpunkValueTrendAnalyzerInput,
    kernel: Kernel,
) -> Result<CyberpunkValueTrendAnalyzerOutput, CyberpunkValueTrendAnalyzerError> {
    let (open, high, low, close) = input_slices(input);
    let entry_level = input.get_entry_level();
    let exit_level = input.get_exit_level();
    validate_common(open, high, low, close, entry_level, exit_level)?;
    let _chosen = normalize_single_kernel(kernel)?;

    let mut value_trend = alloc_with_nan_prefix(close.len(), RANGE75_WINDOW - 1);
    let mut value_trend_lag = alloc_with_nan_prefix(close.len(), RANGE75_WINDOW);
    let mut deviation_index = alloc_with_nan_prefix(close.len(), SMA13_WINDOW - 1);
    let mut overbought_signal = alloc_with_nan_prefix(close.len(), RANGE75_WINDOW - 1);
    let mut buy_signal = alloc_with_nan_prefix(close.len(), RANGE75_WINDOW - 1);
    let mut sell_signal = alloc_with_nan_prefix(close.len(), RANGE75_WINDOW - 1);
    value_trend.fill(f64::NAN);
    value_trend_lag.fill(f64::NAN);
    deviation_index.fill(f64::NAN);
    overbought_signal.fill(f64::NAN);
    buy_signal.fill(f64::NAN);
    sell_signal.fill(f64::NAN);

    compute_row(
        open,
        high,
        low,
        close,
        entry_level,
        exit_level,
        &mut value_trend,
        &mut value_trend_lag,
        &mut deviation_index,
        &mut overbought_signal,
        &mut buy_signal,
        &mut sell_signal,
    );

    Ok(CyberpunkValueTrendAnalyzerOutput {
        value_trend,
        value_trend_lag,
        deviation_index,
        overbought_signal,
        buy_signal,
        sell_signal,
    })
}

pub fn cyberpunk_value_trend_analyzer_into_slice(
    out_value_trend: &mut [f64],
    out_value_trend_lag: &mut [f64],
    out_deviation_index: &mut [f64],
    out_overbought_signal: &mut [f64],
    out_buy_signal: &mut [f64],
    out_sell_signal: &mut [f64],
    input: &CyberpunkValueTrendAnalyzerInput,
    kernel: Kernel,
) -> Result<(), CyberpunkValueTrendAnalyzerError> {
    let (open, high, low, close) = input_slices(input);
    let entry_level = input.get_entry_level();
    let exit_level = input.get_exit_level();
    validate_common(open, high, low, close, entry_level, exit_level)?;
    let _chosen = normalize_single_kernel(kernel)?;

    for out in [
        &mut *out_value_trend,
        &mut *out_value_trend_lag,
        &mut *out_deviation_index,
        &mut *out_overbought_signal,
        &mut *out_buy_signal,
        &mut *out_sell_signal,
    ] {
        if out.len() != close.len() {
            return Err(CyberpunkValueTrendAnalyzerError::OutputLengthMismatch {
                expected: close.len(),
                got: out.len(),
            });
        }
        out.fill(f64::NAN);
    }

    compute_row(
        open,
        high,
        low,
        close,
        entry_level,
        exit_level,
        out_value_trend,
        out_value_trend_lag,
        out_deviation_index,
        out_overbought_signal,
        out_buy_signal,
        out_sell_signal,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn cyberpunk_value_trend_analyzer_into(
    input: &CyberpunkValueTrendAnalyzerInput,
    out_value_trend: &mut [f64],
    out_value_trend_lag: &mut [f64],
    out_deviation_index: &mut [f64],
    out_overbought_signal: &mut [f64],
    out_buy_signal: &mut [f64],
    out_sell_signal: &mut [f64],
) -> Result<(), CyberpunkValueTrendAnalyzerError> {
    cyberpunk_value_trend_analyzer_into_slice(
        out_value_trend,
        out_value_trend_lag,
        out_deviation_index,
        out_overbought_signal,
        out_buy_signal,
        out_sell_signal,
        input,
        Kernel::Auto,
    )
}

pub fn cyberpunk_value_trend_analyzer_output_into_slice(
    dst: &mut [f64],
    input: &CyberpunkValueTrendAnalyzerInput,
    kernel: Kernel,
    field: CyberpunkValueTrendAnalyzerOutputField,
) -> Result<(), CyberpunkValueTrendAnalyzerError> {
    let (open, high, low, close) = input_slices(input);
    let entry_level = input.get_entry_level();
    let exit_level = input.get_exit_level();
    validate_common(open, high, low, close, entry_level, exit_level)?;
    let _chosen = normalize_single_kernel(kernel)?;
    if dst.len() != close.len() {
        return Err(CyberpunkValueTrendAnalyzerError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }
    dst.fill(f64::NAN);
    let mut stream = CyberpunkValueTrendAnalyzerStream::try_new(input.params.clone())?;
    for i in 0..close.len() {
        if let Some(point) = stream.update(open[i], high[i], low[i], close[i]) {
            dst[i] = match field {
                CyberpunkValueTrendAnalyzerOutputField::ValueTrend => point.0,
                CyberpunkValueTrendAnalyzerOutputField::ValueTrendLag => point.1,
                CyberpunkValueTrendAnalyzerOutputField::DeviationIndex => point.2,
                CyberpunkValueTrendAnalyzerOutputField::OverboughtSignal => point.3,
                CyberpunkValueTrendAnalyzerOutputField::BuySignal => point.4,
                CyberpunkValueTrendAnalyzerOutputField::SellSignal => point.5,
            };
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct CyberpunkValueTrendAnalyzerBatchRange {
    pub entry_level: (usize, usize, usize),
    pub exit_level: (usize, usize, usize),
}

impl Default for CyberpunkValueTrendAnalyzerBatchRange {
    fn default() -> Self {
        Self {
            entry_level: (DEFAULT_ENTRY_LEVEL, DEFAULT_ENTRY_LEVEL, 0),
            exit_level: (DEFAULT_EXIT_LEVEL, DEFAULT_EXIT_LEVEL, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CyberpunkValueTrendAnalyzerBatchOutput {
    pub value_trend: Vec<f64>,
    pub value_trend_lag: Vec<f64>,
    pub deviation_index: Vec<f64>,
    pub overbought_signal: Vec<f64>,
    pub buy_signal: Vec<f64>,
    pub sell_signal: Vec<f64>,
    pub combos: Vec<CyberpunkValueTrendAnalyzerParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct CyberpunkValueTrendAnalyzerBatchBuilder {
    range: CyberpunkValueTrendAnalyzerBatchRange,
    kernel: Kernel,
}

impl Default for CyberpunkValueTrendAnalyzerBatchBuilder {
    fn default() -> Self {
        Self {
            range: CyberpunkValueTrendAnalyzerBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl CyberpunkValueTrendAnalyzerBatchBuilder {
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
    pub fn entry_level_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.entry_level = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn entry_level_static(mut self, value: usize) -> Self {
        self.range.entry_level = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn exit_level_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.exit_level = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn exit_level_static(mut self, value: usize) -> Self {
        self.range.exit_level = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<CyberpunkValueTrendAnalyzerBatchOutput, CyberpunkValueTrendAnalyzerError> {
        cyberpunk_value_trend_analyzer_batch_with_kernel(
            open,
            high,
            low,
            close,
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<CyberpunkValueTrendAnalyzerBatchOutput, CyberpunkValueTrendAnalyzerError> {
        cyberpunk_value_trend_analyzer_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_grid_checked(
    range: &CyberpunkValueTrendAnalyzerBatchRange,
) -> Result<Vec<CyberpunkValueTrendAnalyzerParams>, CyberpunkValueTrendAnalyzerError> {
    let entry_levels = axis_usize(
        range.entry_level.0,
        range.entry_level.1,
        range.entry_level.2,
    )?;
    let exit_levels = axis_usize(range.exit_level.0, range.exit_level.1, range.exit_level.2)?;
    let mut combos = Vec::new();
    for &entry_level in &entry_levels {
        for &exit_level in &exit_levels {
            validate_params_only(entry_level, exit_level)?;
            combos.push(CyberpunkValueTrendAnalyzerParams {
                entry_level: Some(entry_level),
                exit_level: Some(exit_level),
            });
        }
    }
    Ok(combos)
}

pub fn expand_grid_cyberpunk_value_trend_analyzer(
    range: &CyberpunkValueTrendAnalyzerBatchRange,
) -> Vec<CyberpunkValueTrendAnalyzerParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn cyberpunk_value_trend_analyzer_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CyberpunkValueTrendAnalyzerBatchRange,
    kernel: Kernel,
) -> Result<CyberpunkValueTrendAnalyzerBatchOutput, CyberpunkValueTrendAnalyzerError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| CyberpunkValueTrendAnalyzerError::InvalidInput {
                msg: "cyberpunk_value_trend_analyzer: rows*cols overflow in batch".to_string(),
            })?;

    let mut value_trend_mu = make_uninit_matrix(rows, cols);
    let mut value_trend_lag_mu = make_uninit_matrix(rows, cols);
    let mut deviation_index_mu = make_uninit_matrix(rows, cols);
    let mut overbought_signal_mu = make_uninit_matrix(rows, cols);
    let mut buy_signal_mu = make_uninit_matrix(rows, cols);
    let mut sell_signal_mu = make_uninit_matrix(rows, cols);

    let mut value_trend_guard = core::mem::ManuallyDrop::new(value_trend_mu);
    let mut value_trend_lag_guard = core::mem::ManuallyDrop::new(value_trend_lag_mu);
    let mut deviation_index_guard = core::mem::ManuallyDrop::new(deviation_index_mu);
    let mut overbought_signal_guard = core::mem::ManuallyDrop::new(overbought_signal_mu);
    let mut buy_signal_guard = core::mem::ManuallyDrop::new(buy_signal_mu);
    let mut sell_signal_guard = core::mem::ManuallyDrop::new(sell_signal_mu);

    let value_trend = unsafe {
        std::slice::from_raw_parts_mut(value_trend_guard.as_mut_ptr() as *mut f64, total)
    };
    let value_trend_lag = unsafe {
        std::slice::from_raw_parts_mut(value_trend_lag_guard.as_mut_ptr() as *mut f64, total)
    };
    let deviation_index = unsafe {
        std::slice::from_raw_parts_mut(deviation_index_guard.as_mut_ptr() as *mut f64, total)
    };
    let overbought_signal = unsafe {
        std::slice::from_raw_parts_mut(overbought_signal_guard.as_mut_ptr() as *mut f64, total)
    };
    let buy_signal =
        unsafe { std::slice::from_raw_parts_mut(buy_signal_guard.as_mut_ptr() as *mut f64, total) };
    let sell_signal = unsafe {
        std::slice::from_raw_parts_mut(sell_signal_guard.as_mut_ptr() as *mut f64, total)
    };

    cyberpunk_value_trend_analyzer_batch_inner_into(
        open,
        high,
        low,
        close,
        sweep,
        kernel,
        true,
        value_trend,
        value_trend_lag,
        deviation_index,
        overbought_signal,
        buy_signal,
        sell_signal,
    )?;

    let value_trend = unsafe {
        Vec::from_raw_parts(
            value_trend_guard.as_mut_ptr() as *mut f64,
            value_trend_guard.len(),
            value_trend_guard.capacity(),
        )
    };
    let value_trend_lag = unsafe {
        Vec::from_raw_parts(
            value_trend_lag_guard.as_mut_ptr() as *mut f64,
            value_trend_lag_guard.len(),
            value_trend_lag_guard.capacity(),
        )
    };
    let deviation_index = unsafe {
        Vec::from_raw_parts(
            deviation_index_guard.as_mut_ptr() as *mut f64,
            deviation_index_guard.len(),
            deviation_index_guard.capacity(),
        )
    };
    let overbought_signal = unsafe {
        Vec::from_raw_parts(
            overbought_signal_guard.as_mut_ptr() as *mut f64,
            overbought_signal_guard.len(),
            overbought_signal_guard.capacity(),
        )
    };
    let buy_signal = unsafe {
        Vec::from_raw_parts(
            buy_signal_guard.as_mut_ptr() as *mut f64,
            buy_signal_guard.len(),
            buy_signal_guard.capacity(),
        )
    };
    let sell_signal = unsafe {
        Vec::from_raw_parts(
            sell_signal_guard.as_mut_ptr() as *mut f64,
            sell_signal_guard.len(),
            sell_signal_guard.capacity(),
        )
    };

    Ok(CyberpunkValueTrendAnalyzerBatchOutput {
        value_trend,
        value_trend_lag,
        deviation_index,
        overbought_signal,
        buy_signal,
        sell_signal,
        combos,
        rows,
        cols,
    })
}

pub fn cyberpunk_value_trend_analyzer_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CyberpunkValueTrendAnalyzerBatchRange,
    kernel: Kernel,
) -> Result<CyberpunkValueTrendAnalyzerBatchOutput, CyberpunkValueTrendAnalyzerError> {
    cyberpunk_value_trend_analyzer_batch_with_kernel(open, high, low, close, sweep, kernel)
}

pub fn cyberpunk_value_trend_analyzer_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CyberpunkValueTrendAnalyzerBatchRange,
    kernel: Kernel,
) -> Result<CyberpunkValueTrendAnalyzerBatchOutput, CyberpunkValueTrendAnalyzerError> {
    cyberpunk_value_trend_analyzer_batch_with_kernel(open, high, low, close, sweep, kernel)
}

fn cyberpunk_value_trend_analyzer_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CyberpunkValueTrendAnalyzerBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_value_trend: &mut [f64],
    out_value_trend_lag: &mut [f64],
    out_deviation_index: &mut [f64],
    out_overbought_signal: &mut [f64],
    out_buy_signal: &mut [f64],
    out_sell_signal: &mut [f64],
) -> Result<Vec<CyberpunkValueTrendAnalyzerParams>, CyberpunkValueTrendAnalyzerError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| CyberpunkValueTrendAnalyzerError::InvalidInput {
                msg: "cyberpunk_value_trend_analyzer: rows*cols overflow in batch_into".to_string(),
            })?;
    for out in [
        &mut *out_value_trend,
        &mut *out_value_trend_lag,
        &mut *out_deviation_index,
        &mut *out_overbought_signal,
        &mut *out_buy_signal,
        &mut *out_sell_signal,
    ] {
        if out.len() != total {
            return Err(CyberpunkValueTrendAnalyzerError::MismatchedOutputLen {
                dst_len: out.len(),
                expected_len: total,
            });
        }
    }

    let max_entry = combos
        .iter()
        .map(|params| params.entry_level.unwrap_or(DEFAULT_ENTRY_LEVEL))
        .max()
        .unwrap_or(DEFAULT_ENTRY_LEVEL);
    let max_exit = combos
        .iter()
        .map(|params| params.exit_level.unwrap_or(DEFAULT_EXIT_LEVEL))
        .max()
        .unwrap_or(DEFAULT_EXIT_LEVEL);
    validate_common(open, high, low, close, max_entry, max_exit)?;
    let _chosen = normalize_batch_kernel(kernel)?;

    let worker = |row: usize,
                  value_trend_row: &mut [f64],
                  value_trend_lag_row: &mut [f64],
                  deviation_index_row: &mut [f64],
                  overbought_signal_row: &mut [f64],
                  buy_signal_row: &mut [f64],
                  sell_signal_row: &mut [f64]| {
        value_trend_row.fill(f64::NAN);
        value_trend_lag_row.fill(f64::NAN);
        deviation_index_row.fill(f64::NAN);
        overbought_signal_row.fill(f64::NAN);
        buy_signal_row.fill(f64::NAN);
        sell_signal_row.fill(f64::NAN);
        let params = &combos[row];
        compute_row(
            open,
            high,
            low,
            close,
            params.entry_level.unwrap_or(DEFAULT_ENTRY_LEVEL),
            params.exit_level.unwrap_or(DEFAULT_EXIT_LEVEL),
            value_trend_row,
            value_trend_lag_row,
            deviation_index_row,
            overbought_signal_row,
            buy_signal_row,
            sell_signal_row,
        );
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel && rows > 1 {
        out_value_trend
            .par_chunks_mut(cols)
            .zip(out_value_trend_lag.par_chunks_mut(cols))
            .zip(out_deviation_index.par_chunks_mut(cols))
            .zip(out_overbought_signal.par_chunks_mut(cols))
            .zip(out_buy_signal.par_chunks_mut(cols))
            .zip(out_sell_signal.par_chunks_mut(cols))
            .enumerate()
            .for_each(
                |(
                    row,
                    (
                        (
                            (
                                ((value_trend_row, value_trend_lag_row), deviation_index_row),
                                overbought_signal_row,
                            ),
                            buy_signal_row,
                        ),
                        sell_signal_row,
                    ),
                )| {
                    worker(
                        row,
                        value_trend_row,
                        value_trend_lag_row,
                        deviation_index_row,
                        overbought_signal_row,
                        buy_signal_row,
                        sell_signal_row,
                    );
                },
            );
    } else {
        for (
            row,
            (
                (
                    (
                        ((value_trend_row, value_trend_lag_row), deviation_index_row),
                        overbought_signal_row,
                    ),
                    buy_signal_row,
                ),
                sell_signal_row,
            ),
        ) in out_value_trend
            .chunks_mut(cols)
            .zip(out_value_trend_lag.chunks_mut(cols))
            .zip(out_deviation_index.chunks_mut(cols))
            .zip(out_overbought_signal.chunks_mut(cols))
            .zip(out_buy_signal.chunks_mut(cols))
            .zip(out_sell_signal.chunks_mut(cols))
            .enumerate()
        {
            worker(
                row,
                value_trend_row,
                value_trend_lag_row,
                deviation_index_row,
                overbought_signal_row,
                buy_signal_row,
                sell_signal_row,
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (
            row,
            (
                (
                    (
                        ((value_trend_row, value_trend_lag_row), deviation_index_row),
                        overbought_signal_row,
                    ),
                    buy_signal_row,
                ),
                sell_signal_row,
            ),
        ) in out_value_trend
            .chunks_mut(cols)
            .zip(out_value_trend_lag.chunks_mut(cols))
            .zip(out_deviation_index.chunks_mut(cols))
            .zip(out_overbought_signal.chunks_mut(cols))
            .zip(out_buy_signal.chunks_mut(cols))
            .zip(out_sell_signal.chunks_mut(cols))
            .enumerate()
        {
            worker(
                row,
                value_trend_row,
                value_trend_lag_row,
                deviation_index_row,
                overbought_signal_row,
                buy_signal_row,
                sell_signal_row,
            );
        }
    }

    Ok(combos)
}

#[derive(Debug, Clone)]
pub struct CyberpunkValueTrendAnalyzerStream {
    entry_level: usize,
    exit_level: usize,
    index: usize,
    valid_run: usize,
    sma13: RollingSum,
    lowest75: MonotonicQueue,
    highest75: MonotonicQueue,
    close_norm_sma: WeightedSmaState,
    smooth5: WeightedSmaState,
    prev_value_trend: f64,
}

impl CyberpunkValueTrendAnalyzerStream {
    pub fn try_new(
        params: CyberpunkValueTrendAnalyzerParams,
    ) -> Result<Self, CyberpunkValueTrendAnalyzerError> {
        let entry_level = params.entry_level.unwrap_or(DEFAULT_ENTRY_LEVEL);
        let exit_level = params.exit_level.unwrap_or(DEFAULT_EXIT_LEVEL);
        validate_params_only(entry_level, exit_level)?;
        Ok(Self {
            entry_level,
            exit_level,
            index: 0,
            valid_run: 0,
            sma13: RollingSum::default(),
            lowest75: MonotonicQueue::default(),
            highest75: MonotonicQueue::default(),
            close_norm_sma: WeightedSmaState::new(20),
            smooth5: WeightedSmaState::new(5),
            prev_value_trend: f64::NAN,
        })
    }

    fn reset(&mut self) {
        self.valid_run = 0;
        self.sma13.reset();
        self.lowest75.reset();
        self.highest75.reset();
        self.close_norm_sma.reset();
        self.smooth5.reset();
        self.prev_value_trend = f64::NAN;
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        let idx = self.index;
        self.index = self.index.saturating_add(1);

        if !is_valid_ohlc(open, high, low, close) {
            self.reset();
            return None;
        }

        self.valid_run += 1;
        let ma13 = self.sma13.push(close);
        self.lowest75.push_min(idx, low);
        self.highest75.push_max(idx, high);
        let min_index = idx.saturating_sub(RANGE75_WINDOW - 1);
        self.lowest75.prune(min_index);
        self.highest75.prune(min_index);

        let value_trend_lag = self.prev_value_trend;
        let mut value_trend = f64::NAN;
        let mut deviation_index = f64::NAN;
        let mut overbought_signal = f64::NAN;
        let mut buy_signal = if self.valid_run >= RANGE75_WINDOW {
            0.0
        } else {
            f64::NAN
        };
        let mut sell_signal = buy_signal;

        if self.valid_run >= RANGE75_WINDOW {
            let range_low = self.lowest75.current().unwrap_or(f64::NAN);
            let range_high = self.highest75.current().unwrap_or(f64::NAN);
            let range = range_high - range_low;
            if range.is_finite() && range > 0.0 {
                let close_norm = (close - range_low) * 100.0 / range;
                let close_norm_avg = self.close_norm_sma.update(close_norm);
                let smooth = self.smooth5.update(close_norm_avg);
                if close_norm_avg.is_finite() && smooth.is_finite() {
                    value_trend = 3.0 * close_norm_avg - 2.0 * smooth;
                } else {
                    buy_signal = f64::NAN;
                    sell_signal = f64::NAN;
                }
            } else {
                self.close_norm_sma.reset();
                self.smooth5.reset();
                buy_signal = f64::NAN;
                sell_signal = f64::NAN;
            }
        }

        if let Some(avg13) = ma13 {
            if avg13.is_finite() && avg13 != 0.0 {
                deviation_index = 100.0 - (((close - avg13) / avg13) * 100.0).abs();
                if value_trend.is_finite() && value_trend > deviation_index {
                    overbought_signal = deviation_index;
                }
            }
        }

        if value_trend.is_finite() && self.prev_value_trend.is_finite() {
            if self.prev_value_trend <= self.entry_level as f64
                && value_trend > self.entry_level as f64
            {
                buy_signal = 1.0;
            }
            if self.prev_value_trend >= self.exit_level as f64
                && value_trend < self.exit_level as f64
            {
                sell_signal = 1.0;
            }
        }

        self.prev_value_trend = value_trend;
        Some((
            value_trend,
            value_trend_lag,
            deviation_index,
            overbought_signal,
            buy_signal,
            sell_signal,
        ))
    }
}

impl CyberpunkValueTrendAnalyzerBuilder {
    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<CyberpunkValueTrendAnalyzerOutput, CyberpunkValueTrendAnalyzerError> {
        cyberpunk_value_trend_analyzer_with_kernel(
            &CyberpunkValueTrendAnalyzerInput::from_candles(
                candles,
                CyberpunkValueTrendAnalyzerParams {
                    entry_level: self.entry_level,
                    exit_level: self.exit_level,
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
    ) -> Result<CyberpunkValueTrendAnalyzerOutput, CyberpunkValueTrendAnalyzerError> {
        cyberpunk_value_trend_analyzer_with_kernel(
            &CyberpunkValueTrendAnalyzerInput::from_slices(
                open,
                high,
                low,
                close,
                CyberpunkValueTrendAnalyzerParams {
                    entry_level: self.entry_level,
                    exit_level: self.exit_level,
                },
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<CyberpunkValueTrendAnalyzerStream, CyberpunkValueTrendAnalyzerError> {
        CyberpunkValueTrendAnalyzerStream::try_new(CyberpunkValueTrendAnalyzerParams {
            entry_level: self.entry_level,
            exit_level: self.exit_level,
        })
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "cyberpunk_value_trend_analyzer",
    signature = (open, high, low, close, entry_level=DEFAULT_ENTRY_LEVEL, exit_level=DEFAULT_EXIT_LEVEL, kernel=None)
)]
pub fn cyberpunk_value_trend_analyzer_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    entry_level: usize,
    exit_level: usize,
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
    let kern = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| {
            cyberpunk_value_trend_analyzer_with_kernel(
                &CyberpunkValueTrendAnalyzerInput::from_slices(
                    open,
                    high,
                    low,
                    close,
                    CyberpunkValueTrendAnalyzerParams {
                        entry_level: Some(entry_level),
                        exit_level: Some(exit_level),
                    },
                ),
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.value_trend.into_pyarray(py),
        out.value_trend_lag.into_pyarray(py),
        out.deviation_index.into_pyarray(py),
        out.overbought_signal.into_pyarray(py),
        out.buy_signal.into_pyarray(py),
        out.sell_signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "cyberpunk_value_trend_analyzer_batch",
    signature = (open, high, low, close, entry_level_range=(DEFAULT_ENTRY_LEVEL, DEFAULT_ENTRY_LEVEL, 0), exit_level_range=(DEFAULT_EXIT_LEVEL, DEFAULT_EXIT_LEVEL, 0), kernel=None)
)]
pub fn cyberpunk_value_trend_analyzer_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    entry_level_range: (usize, usize, usize),
    exit_level_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let out = py
        .allow_threads(|| {
            cyberpunk_value_trend_analyzer_batch_with_kernel(
                open,
                high,
                low,
                close,
                &CyberpunkValueTrendAnalyzerBatchRange {
                    entry_level: entry_level_range,
                    exit_level: exit_level_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "value_trend",
        out.value_trend
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "value_trend_lag",
        out.value_trend_lag
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "deviation_index",
        out.deviation_index
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "overbought_signal",
        out.overbought_signal
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
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
        "entry_levels",
        out.combos
            .iter()
            .map(|params| params.entry_level.unwrap_or(DEFAULT_ENTRY_LEVEL) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "exit_levels",
        out.combos
            .iter()
            .map(|params| params.exit_level.unwrap_or(DEFAULT_EXIT_LEVEL) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "CyberpunkValueTrendAnalyzerStream")]
pub struct CyberpunkValueTrendAnalyzerStreamPy {
    stream: CyberpunkValueTrendAnalyzerStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CyberpunkValueTrendAnalyzerStreamPy {
    #[new]
    #[pyo3(signature = (entry_level=DEFAULT_ENTRY_LEVEL, exit_level=DEFAULT_EXIT_LEVEL))]
    fn new(entry_level: usize, exit_level: usize) -> PyResult<Self> {
        let stream =
            CyberpunkValueTrendAnalyzerStream::try_new(CyberpunkValueTrendAnalyzerParams {
                entry_level: Some(entry_level),
                exit_level: Some(exit_level),
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
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
pub fn register_cyberpunk_value_trend_analyzer_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cyberpunk_value_trend_analyzer_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        cyberpunk_value_trend_analyzer_batch_py,
        m
    )?)?;
    m.add_class::<CyberpunkValueTrendAnalyzerStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CyberpunkValueTrendAnalyzerBatchConfig {
    pub entry_level_range: Vec<usize>,
    pub exit_level_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = cyberpunk_value_trend_analyzer_js)]
pub fn cyberpunk_value_trend_analyzer_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    entry_level: usize,
    exit_level: usize,
) -> Result<JsValue, JsValue> {
    let out = cyberpunk_value_trend_analyzer_with_kernel(
        &CyberpunkValueTrendAnalyzerInput::from_slices(
            open,
            high,
            low,
            close,
            CyberpunkValueTrendAnalyzerParams {
                entry_level: Some(entry_level),
                exit_level: Some(exit_level),
            },
        ),
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    for (key, value) in [
        (
            "value_trend",
            serde_wasm_bindgen::to_value(&out.value_trend).unwrap(),
        ),
        (
            "value_trend_lag",
            serde_wasm_bindgen::to_value(&out.value_trend_lag).unwrap(),
        ),
        (
            "deviation_index",
            serde_wasm_bindgen::to_value(&out.deviation_index).unwrap(),
        ),
        (
            "overbought_signal",
            serde_wasm_bindgen::to_value(&out.overbought_signal).unwrap(),
        ),
        (
            "buy_signal",
            serde_wasm_bindgen::to_value(&out.buy_signal).unwrap(),
        ),
        (
            "sell_signal",
            serde_wasm_bindgen::to_value(&out.sell_signal).unwrap(),
        ),
    ] {
        js_sys::Reflect::set(&obj, &JsValue::from_str(key), &value)?;
    }
    Ok(JsValue::from(obj))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = cyberpunk_value_trend_analyzer_batch_js)]
pub fn cyberpunk_value_trend_analyzer_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: CyberpunkValueTrendAnalyzerBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if cfg.entry_level_range.len() != 3 || cfg.exit_level_range.len() != 3 {
        return Err(JsValue::from_str(
            "cyberpunk_value_trend_analyzer_batch_js: range vectors must have length 3",
        ));
    }
    let out = cyberpunk_value_trend_analyzer_batch_with_kernel(
        open,
        high,
        low,
        close,
        &CyberpunkValueTrendAnalyzerBatchRange {
            entry_level: (
                cfg.entry_level_range[0],
                cfg.entry_level_range[1],
                cfg.entry_level_range[2],
            ),
            exit_level: (
                cfg.exit_level_range[0],
                cfg.exit_level_range[1],
                cfg.exit_level_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    for (key, value) in [
        (
            "value_trend",
            serde_wasm_bindgen::to_value(&out.value_trend).unwrap(),
        ),
        (
            "value_trend_lag",
            serde_wasm_bindgen::to_value(&out.value_trend_lag).unwrap(),
        ),
        (
            "deviation_index",
            serde_wasm_bindgen::to_value(&out.deviation_index).unwrap(),
        ),
        (
            "overbought_signal",
            serde_wasm_bindgen::to_value(&out.overbought_signal).unwrap(),
        ),
        (
            "buy_signal",
            serde_wasm_bindgen::to_value(&out.buy_signal).unwrap(),
        ),
        (
            "sell_signal",
            serde_wasm_bindgen::to_value(&out.sell_signal).unwrap(),
        ),
    ] {
        js_sys::Reflect::set(&obj, &JsValue::from_str(key), &value)?;
    }
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

    let combos = out
        .combos
        .iter()
        .map(|params| {
            let item = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &item,
                &JsValue::from_str("entry_level"),
                &JsValue::from_f64(params.entry_level.unwrap_or(DEFAULT_ENTRY_LEVEL) as f64),
            );
            let _ = js_sys::Reflect::set(
                &item,
                &JsValue::from_str("exit_level"),
                &JsValue::from_f64(params.exit_level.unwrap_or(DEFAULT_EXIT_LEVEL) as f64),
            );
            item
        })
        .collect::<js_sys::Array>();
    js_sys::Reflect::set(&obj, &JsValue::from_str("combos"), &combos)?;
    Ok(JsValue::from(obj))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cyberpunk_value_trend_analyzer_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(OUTPUTS * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cyberpunk_value_trend_analyzer_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, OUTPUTS * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cyberpunk_value_trend_analyzer_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    entry_level: usize,
    exit_level: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to cyberpunk_value_trend_analyzer_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, OUTPUTS * len);
        let (value_trend, tail) = out.split_at_mut(len);
        let (value_trend_lag, tail) = tail.split_at_mut(len);
        let (deviation_index, tail) = tail.split_at_mut(len);
        let (overbought_signal, tail) = tail.split_at_mut(len);
        let (buy_signal, sell_signal) = tail.split_at_mut(len);
        cyberpunk_value_trend_analyzer_into_slice(
            value_trend,
            value_trend_lag,
            deviation_index,
            overbought_signal,
            buy_signal,
            sell_signal,
            &CyberpunkValueTrendAnalyzerInput::from_slices(
                open,
                high,
                low,
                close,
                CyberpunkValueTrendAnalyzerParams {
                    entry_level: Some(entry_level),
                    exit_level: Some(exit_level),
                },
            ),
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cyberpunk_value_trend_analyzer_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    entry_start: usize,
    entry_end: usize,
    entry_step: usize,
    exit_start: usize,
    exit_end: usize,
    exit_step: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to cyberpunk_value_trend_analyzer_batch_into",
        ));
    }

    let sweep = CyberpunkValueTrendAnalyzerBatchRange {
        entry_level: (entry_start, entry_end, entry_step),
        exit_level: (exit_start, exit_end, exit_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let split = rows.checked_mul(len).ok_or_else(|| {
        JsValue::from_str("rows*cols overflow in cyberpunk_value_trend_analyzer_batch_into")
    })?;
    let total = split.checked_mul(OUTPUTS).ok_or_else(|| {
        JsValue::from_str("outputs*rows*cols overflow in cyberpunk_value_trend_analyzer_batch_into")
    })?;

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (value_trend, tail) = out.split_at_mut(split);
        let (value_trend_lag, tail) = tail.split_at_mut(split);
        let (deviation_index, tail) = tail.split_at_mut(split);
        let (overbought_signal, tail) = tail.split_at_mut(split);
        let (buy_signal, sell_signal) = tail.split_at_mut(split);
        cyberpunk_value_trend_analyzer_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            Kernel::Auto,
            false,
            value_trend,
            value_trend_lag,
            deviation_index,
            overbought_signal,
            buy_signal,
            sell_signal,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cyberpunk_value_trend_analyzer_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    entry_level: usize,
    exit_level: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = cyberpunk_value_trend_analyzer_js(open, high, low, close, entry_level, exit_level)?;
    crate::write_wasm_object_f64_outputs(
        "cyberpunk_value_trend_analyzer_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn cyberpunk_value_trend_analyzer_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = cyberpunk_value_trend_analyzer_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "cyberpunk_value_trend_analyzer_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let trend = 100.0 + i as f64 * 0.18;
            let wave = ((i % 17) as f64 - 8.0) * 0.9 + ((i % 9) as f64 - 4.0) * 0.35;
            let o = trend + wave - if i % 2 == 0 { 0.7 } else { -0.4 };
            let c = trend + wave + ((i * 7 % 11) as f64 - 5.0) * 0.22;
            let h = o.max(c) + 1.25 + (i % 5) as f64 * 0.08;
            let l = o.min(c) - 1.15 - (i % 7) as f64 * 0.07;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
        }
        (open, high, low, close)
    }

    fn assert_vec_eq_nan(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((*a - *b).abs() <= 1e-12, "mismatch: left={a:?} right={b:?}");
        }
    }

    #[test]
    fn output_contract() {
        let (open, high, low, close) = sample_ohlc(240);
        let out = cyberpunk_value_trend_analyzer(&CyberpunkValueTrendAnalyzerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            CyberpunkValueTrendAnalyzerParams::default(),
        ))
        .unwrap();
        assert_eq!(out.value_trend.len(), close.len());
        assert_eq!(out.sell_signal.len(), close.len());
        assert!(out.value_trend.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn invalid_params() {
        let (open, high, low, close) = sample_ohlc(120);
        let err = cyberpunk_value_trend_analyzer(&CyberpunkValueTrendAnalyzerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            CyberpunkValueTrendAnalyzerParams {
                entry_level: Some(0),
                exit_level: Some(75),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            CyberpunkValueTrendAnalyzerError::InvalidEntryLevel { .. }
        ));
    }

    #[test]
    fn into_slice_matches_safe_api() {
        let (open, high, low, close) = sample_ohlc(220);
        let input = CyberpunkValueTrendAnalyzerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            CyberpunkValueTrendAnalyzerParams::default(),
        );
        let safe = cyberpunk_value_trend_analyzer(&input).unwrap();
        let mut value_trend = vec![0.0; close.len()];
        let mut value_trend_lag = vec![0.0; close.len()];
        let mut deviation_index = vec![0.0; close.len()];
        let mut overbought_signal = vec![0.0; close.len()];
        let mut buy_signal = vec![0.0; close.len()];
        let mut sell_signal = vec![0.0; close.len()];
        cyberpunk_value_trend_analyzer_into_slice(
            &mut value_trend,
            &mut value_trend_lag,
            &mut deviation_index,
            &mut overbought_signal,
            &mut buy_signal,
            &mut sell_signal,
            &input,
            Kernel::Auto,
        )
        .unwrap();
        assert_vec_eq_nan(&safe.value_trend, &value_trend);
        assert_vec_eq_nan(&safe.value_trend_lag, &value_trend_lag);
        assert_vec_eq_nan(&safe.deviation_index, &deviation_index);
        assert_vec_eq_nan(&safe.overbought_signal, &overbought_signal);
        assert_vec_eq_nan(&safe.buy_signal, &buy_signal);
        assert_vec_eq_nan(&safe.sell_signal, &sell_signal);
    }

    #[test]
    fn batch_single_matches_safe_api() {
        let (open, high, low, close) = sample_ohlc(210);
        let batch = cyberpunk_value_trend_analyzer_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &CyberpunkValueTrendAnalyzerBatchRange::default(),
            Kernel::Auto,
        )
        .unwrap();
        let safe = cyberpunk_value_trend_analyzer(&CyberpunkValueTrendAnalyzerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            CyberpunkValueTrendAnalyzerParams::default(),
        ))
        .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_vec_eq_nan(&batch.value_trend, &safe.value_trend);
        assert_vec_eq_nan(&batch.value_trend_lag, &safe.value_trend_lag);
        assert_vec_eq_nan(&batch.deviation_index, &safe.deviation_index);
        assert_vec_eq_nan(&batch.overbought_signal, &safe.overbought_signal);
        assert_vec_eq_nan(&batch.buy_signal, &safe.buy_signal);
        assert_vec_eq_nan(&batch.sell_signal, &safe.sell_signal);
    }

    #[test]
    fn stream_matches_safe_api() {
        let (open, high, low, close) = sample_ohlc(200);
        let safe = cyberpunk_value_trend_analyzer(&CyberpunkValueTrendAnalyzerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            CyberpunkValueTrendAnalyzerParams::default(),
        ))
        .unwrap();
        let mut stream = CyberpunkValueTrendAnalyzerStream::try_new(
            CyberpunkValueTrendAnalyzerParams::default(),
        )
        .unwrap();
        let mut value_trend = Vec::with_capacity(close.len());
        let mut value_trend_lag = Vec::with_capacity(close.len());
        let mut deviation_index = Vec::with_capacity(close.len());
        let mut overbought_signal = Vec::with_capacity(close.len());
        let mut buy_signal = Vec::with_capacity(close.len());
        let mut sell_signal = Vec::with_capacity(close.len());
        for (((&o, &h), &l), &c) in open
            .iter()
            .zip(high.iter())
            .zip(low.iter())
            .zip(close.iter())
        {
            let point = stream.update(o, h, l, c).unwrap();
            value_trend.push(point.0);
            value_trend_lag.push(point.1);
            deviation_index.push(point.2);
            overbought_signal.push(point.3);
            buy_signal.push(point.4);
            sell_signal.push(point.5);
        }
        assert_vec_eq_nan(&safe.value_trend, &value_trend);
        assert_vec_eq_nan(&safe.value_trend_lag, &value_trend_lag);
        assert_vec_eq_nan(&safe.deviation_index, &deviation_index);
        assert_vec_eq_nan(&safe.overbought_signal, &overbought_signal);
        assert_vec_eq_nan(&safe.buy_signal, &buy_signal);
        assert_vec_eq_nan(&safe.sell_signal, &sell_signal);
    }

    #[test]
    fn batch_multi_param_contract() {
        let (open, high, low, close) = sample_ohlc(180);
        let batch = cyberpunk_value_trend_analyzer_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &CyberpunkValueTrendAnalyzerBatchRange {
                entry_level: (25, 35, 10),
                exit_level: (70, 80, 10),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, close.len());
        assert_eq!(batch.combos.len(), 4);
        assert_eq!(batch.value_trend.len(), 4 * close.len());
    }
}
