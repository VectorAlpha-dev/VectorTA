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

use crate::indicators::ema::{ema_with_kernel, EmaData, EmaInput, EmaParams, EmaStream};
use crate::indicators::moving_averages::frama::{FramaParams, FramaStream};
use crate::indicators::moving_averages::ma::{ma_with_kernel, MaData};
use crate::indicators::moving_averages::ma_stream::{ma_stream, MaStream};
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use std::error::Error;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";
const DEFAULT_AVG_TYPE: &str = "SMA";
const DEFAULT_LENGTH: usize = 28;
const DEFAULT_MULT: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Open,
    High,
    Low,
    Close,
    Oc2,
    Hl2,
    Occ3,
    Hlc3,
    Ohlc4,
    Hlcc4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AvgKind {
    Sma,
    Ema,
    Hma,
    Dema,
    Tema,
    Rma,
    Frama,
}

#[derive(Debug, Clone)]
pub enum EmdTrendData<'a> {
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
pub struct EmdTrendOutput {
    pub direction: Vec<f64>,
    pub average: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EmdTrendParams {
    pub source: Option<String>,
    pub avg_type: Option<String>,
    pub length: Option<usize>,
    pub mult: Option<f64>,
}

impl Default for EmdTrendParams {
    fn default() -> Self {
        Self {
            source: Some(DEFAULT_SOURCE.to_string()),
            avg_type: Some(DEFAULT_AVG_TYPE.to_string()),
            length: Some(DEFAULT_LENGTH),
            mult: Some(DEFAULT_MULT),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmdTrendInput<'a> {
    pub data: EmdTrendData<'a>,
    pub params: EmdTrendParams,
}

impl<'a> EmdTrendInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: EmdTrendParams) -> Self {
        Self {
            data: EmdTrendData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: EmdTrendParams,
    ) -> Self {
        Self {
            data: EmdTrendData::Slices {
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
        Self::from_candles(candles, EmdTrendParams::default())
    }

    #[inline]
    pub fn get_source(&self) -> &str {
        self.params.source.as_deref().unwrap_or(DEFAULT_SOURCE)
    }

    #[inline]
    pub fn get_avg_type(&self) -> &str {
        self.params.avg_type.as_deref().unwrap_or(DEFAULT_AVG_TYPE)
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(DEFAULT_MULT)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EmdTrendBuilder {
    source: Option<SourceKind>,
    avg_type: Option<AvgKind>,
    length: Option<usize>,
    mult: Option<f64>,
    kernel: Kernel,
}

impl Default for EmdTrendBuilder {
    fn default() -> Self {
        Self {
            source: None,
            avg_type: None,
            length: None,
            mult: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EmdTrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &str) -> Result<Self, EmdTrendError> {
        self.source = Some(parse_source_kind(value)?);
        Ok(self)
    }

    #[inline(always)]
    pub fn avg_type(mut self, value: &str) -> Result<Self, EmdTrendError> {
        self.avg_type = Some(parse_avg_kind(value)?);
        Ok(self)
    }

    #[inline(always)]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline(always)]
    pub fn mult(mut self, value: f64) -> Self {
        self.mult = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }
}

#[derive(Debug, Error)]
pub enum EmdTrendError {
    #[error("emd_trend: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "emd_trend: Input length mismatch: open = {open_len}, high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    InputLengthMismatch {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("emd_trend: All values are NaN.")]
    AllValuesNaN,
    #[error("emd_trend: Invalid source: {value}")]
    InvalidSource { value: String },
    #[error("emd_trend: Invalid avg_type: {value}")]
    InvalidAvgType { value: String },
    #[error("emd_trend: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("emd_trend: Invalid mult: {mult}")]
    InvalidMult { mult: f64 },
    #[error("emd_trend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("emd_trend: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("emd_trend: Output length mismatch: dst = {dst_len}, expected = {expected_len}")]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("emd_trend: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("emd_trend: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("emd_trend: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[inline(always)]
fn source_kind_name(kind: SourceKind) -> &'static str {
    match kind {
        SourceKind::Open => "open",
        SourceKind::High => "high",
        SourceKind::Low => "low",
        SourceKind::Close => "close",
        SourceKind::Oc2 => "oc2",
        SourceKind::Hl2 => "hl2",
        SourceKind::Occ3 => "occ3",
        SourceKind::Hlc3 => "hlc3",
        SourceKind::Ohlc4 => "ohlc4",
        SourceKind::Hlcc4 => "hlcc4",
    }
}

#[inline(always)]
fn avg_kind_name(kind: AvgKind) -> &'static str {
    match kind {
        AvgKind::Sma => "SMA",
        AvgKind::Ema => "EMA",
        AvgKind::Hma => "HMA",
        AvgKind::Dema => "DEMA",
        AvgKind::Tema => "TEMA",
        AvgKind::Rma => "RMA",
        AvgKind::Frama => "FRAMA",
    }
}

#[inline(always)]
fn parse_source_kind(value: &str) -> Result<SourceKind, EmdTrendError> {
    if value.eq_ignore_ascii_case("open") {
        return Ok(SourceKind::Open);
    }
    if value.eq_ignore_ascii_case("high") {
        return Ok(SourceKind::High);
    }
    if value.eq_ignore_ascii_case("low") {
        return Ok(SourceKind::Low);
    }
    if value.eq_ignore_ascii_case("close") {
        return Ok(SourceKind::Close);
    }
    if value.eq_ignore_ascii_case("oc2") {
        return Ok(SourceKind::Oc2);
    }
    if value.eq_ignore_ascii_case("hl2") {
        return Ok(SourceKind::Hl2);
    }
    if value.eq_ignore_ascii_case("occ3") {
        return Ok(SourceKind::Occ3);
    }
    if value.eq_ignore_ascii_case("hlc3") {
        return Ok(SourceKind::Hlc3);
    }
    if value.eq_ignore_ascii_case("ohlc4") {
        return Ok(SourceKind::Ohlc4);
    }
    if value.eq_ignore_ascii_case("hlcc4") {
        return Ok(SourceKind::Hlcc4);
    }
    Err(EmdTrendError::InvalidSource {
        value: value.to_string(),
    })
}

#[inline(always)]
fn parse_avg_kind(value: &str) -> Result<AvgKind, EmdTrendError> {
    if value.eq_ignore_ascii_case("SMA") {
        return Ok(AvgKind::Sma);
    }
    if value.eq_ignore_ascii_case("EMA") {
        return Ok(AvgKind::Ema);
    }
    if value.eq_ignore_ascii_case("HMA") {
        return Ok(AvgKind::Hma);
    }
    if value.eq_ignore_ascii_case("DEMA") {
        return Ok(AvgKind::Dema);
    }
    if value.eq_ignore_ascii_case("TEMA") {
        return Ok(AvgKind::Tema);
    }
    if value.eq_ignore_ascii_case("RMA") {
        return Ok(AvgKind::Rma);
    }
    if value.eq_ignore_ascii_case("FRAMA") {
        return Ok(AvgKind::Frama);
    }
    Err(EmdTrendError::InvalidAvgType {
        value: value.to_string(),
    })
}

#[inline(always)]
fn internal_avg_id(kind: AvgKind) -> &'static str {
    match kind {
        AvgKind::Sma => "sma",
        AvgKind::Ema => "ema",
        AvgKind::Hma => "hma",
        AvgKind::Dema => "dema",
        AvgKind::Tema => "tema",
        AvgKind::Rma => "wilders",
        AvgKind::Frama => "frama",
    }
}

#[inline(always)]
fn source_value(kind: SourceKind, open: f64, high: f64, low: f64, close: f64) -> f64 {
    match kind {
        SourceKind::Open => open,
        SourceKind::High => high,
        SourceKind::Low => low,
        SourceKind::Close => close,
        SourceKind::Oc2 => {
            if open.is_finite() && close.is_finite() {
                0.5 * (open + close)
            } else {
                f64::NAN
            }
        }
        SourceKind::Hl2 => {
            if high.is_finite() && low.is_finite() {
                0.5 * (high + low)
            } else {
                f64::NAN
            }
        }
        SourceKind::Occ3 => {
            if open.is_finite() && close.is_finite() {
                (open + close + close) / 3.0
            } else {
                f64::NAN
            }
        }
        SourceKind::Hlc3 => {
            if high.is_finite() && low.is_finite() && close.is_finite() {
                (high + low + close) / 3.0
            } else {
                f64::NAN
            }
        }
        SourceKind::Ohlc4 => {
            if open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite() {
                (open + high + low + close) * 0.25
            } else {
                f64::NAN
            }
        }
        SourceKind::Hlcc4 => {
            if high.is_finite() && low.is_finite() && close.is_finite() {
                (high + low + close + close) * 0.25
            } else {
                f64::NAN
            }
        }
    }
}

#[inline(always)]
fn longest_valid_run(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for &value in values {
        if value.is_finite() {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}

#[inline(always)]
fn axis_usize(range: (usize, usize, usize)) -> Vec<usize> {
    let (start, end, step) = range;
    if step == 0 || start == end {
        return vec![start];
    }
    let (lo, hi) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let mut out = Vec::new();
    let mut value = lo;
    loop {
        out.push(value);
        match value.checked_add(step) {
            Some(next) if next <= hi => value = next,
            _ => break,
        }
    }
    if start > end {
        out.reverse();
    }
    out
}

#[inline(always)]
fn axis_f64(range: (f64, f64, f64)) -> Vec<f64> {
    let (start, end, step) = range;
    if step == 0.0 || (start - end).abs() <= f64::EPSILON {
        return vec![start];
    }
    let (lo, hi, reverse) = if start <= end {
        (start, end, false)
    } else {
        (end, start, true)
    };
    let mut out = Vec::new();
    let mut value = lo;
    let eps = step.abs() * 1e-9 + 1e-12;
    while value <= hi + eps {
        out.push(value);
        value += step.abs();
    }
    if reverse {
        out.reverse();
    }
    out
}

#[inline(always)]
fn validate_params_only(
    source: &str,
    avg_type: &str,
    length: usize,
    mult: f64,
) -> Result<(SourceKind, AvgKind), EmdTrendError> {
    let source_kind = parse_source_kind(source)?;
    let avg_kind = parse_avg_kind(avg_type)?;
    if length == 0 {
        return Err(EmdTrendError::InvalidLength { length });
    }
    if !mult.is_finite() || mult < 0.05 {
        return Err(EmdTrendError::InvalidMult { mult });
    }
    Ok((source_kind, avg_kind))
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a EmdTrendInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), EmdTrendError> {
    match &input.data {
        EmdTrendData::Candles { candles } => Ok((
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        EmdTrendData::Slices {
            open,
            high,
            low,
            close,
        } => Ok((open, high, low, close)),
    }
}

#[inline(always)]
fn build_source_series(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source_kind: SourceKind,
) -> Result<Vec<f64>, EmdTrendError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(EmdTrendError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(EmdTrendError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let mut out = Vec::with_capacity(close.len());
    for i in 0..close.len() {
        out.push(source_value(
            source_kind,
            open[i],
            high[i],
            low[i],
            close[i],
        ));
    }
    if longest_valid_run(&out) == 0 {
        return Err(EmdTrendError::AllValuesNaN);
    }
    Ok(out)
}

#[inline(always)]
fn borrowed_source_series<'a>(
    input: &'a EmdTrendInput<'a>,
    source_kind: SourceKind,
) -> Option<&'a [f64]> {
    match (&input.data, source_kind) {
        (EmdTrendData::Candles { candles }, SourceKind::Open) => Some(&candles.open),
        (EmdTrendData::Candles { candles }, SourceKind::High) => Some(&candles.high),
        (EmdTrendData::Candles { candles }, SourceKind::Low) => Some(&candles.low),
        (EmdTrendData::Candles { candles }, SourceKind::Close) => Some(&candles.close),
        (EmdTrendData::Candles { candles }, SourceKind::Hl2) => Some(&candles.hl2),
        (EmdTrendData::Candles { candles }, SourceKind::Hlc3) => Some(&candles.hlc3),
        (EmdTrendData::Candles { candles }, SourceKind::Ohlc4) => Some(&candles.ohlc4),
        (EmdTrendData::Candles { candles }, SourceKind::Hlcc4) => Some(&candles.hlcc4),
        (EmdTrendData::Slices { open, .. }, SourceKind::Open) => Some(open),
        (EmdTrendData::Slices { high, .. }, SourceKind::High) => Some(high),
        (EmdTrendData::Slices { low, .. }, SourceKind::Low) => Some(low),
        (EmdTrendData::Slices { close, .. }, SourceKind::Close) => Some(close),
        _ => None,
    }
}

#[inline(always)]
fn source_series<'a>(
    input: &'a EmdTrendInput<'a>,
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    source_kind: SourceKind,
) -> Result<std::borrow::Cow<'a, [f64]>, EmdTrendError> {
    if let Some(src) = borrowed_source_series(input, source_kind) {
        if src.is_empty() {
            return Err(EmdTrendError::EmptyInputData);
        }
        if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
            return Err(EmdTrendError::InputLengthMismatch {
                open_len: open.len(),
                high_len: high.len(),
                low_len: low.len(),
                close_len: close.len(),
            });
        }
        if src.len() != close.len() {
            return Err(EmdTrendError::InputLengthMismatch {
                open_len: open.len(),
                high_len: high.len(),
                low_len: low.len(),
                close_len: close.len(),
            });
        }
        if longest_valid_run(src) == 0 {
            return Err(EmdTrendError::AllValuesNaN);
        }
        Ok(std::borrow::Cow::Borrowed(src))
    } else {
        build_source_series(open, high, low, close, source_kind).map(std::borrow::Cow::Owned)
    }
}

#[inline(always)]
fn parse_number_after(segment: &str) -> Option<usize> {
    let eq = segment.find('=')?;
    let number: String = segment[eq + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    number.parse().ok()
}

#[inline(always)]
fn parse_needed_valid(msg: &str) -> Option<(usize, usize)> {
    let needed_key = "needed";
    let valid_key = "valid";
    let needed_idx = msg.find(needed_key)?;
    let valid_idx = msg.find(valid_key)?;
    let needed = parse_number_after(&msg[needed_idx + needed_key.len()..])?;
    let valid = parse_number_after(&msg[valid_idx + valid_key.len()..])?;
    Some((needed, valid))
}

#[inline(always)]
fn map_ma_error(err: Box<dyn Error>) -> EmdTrendError {
    let msg = err.to_string();
    if let Some((needed, valid)) = parse_needed_valid(&msg) {
        return EmdTrendError::NotEnoughValidData { needed, valid };
    }
    EmdTrendError::InvalidInput { msg }
}

#[inline(always)]
fn normalize_single_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => detect_best_kernel(),
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => Kernel::Scalar,
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        other => other,
    }
}

#[inline(always)]
fn normalize_batch_kernel(kernel: Kernel) -> Result<Kernel, EmdTrendError> {
    Ok(match kernel {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::ScalarBatch => Kernel::Scalar,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::Avx512Batch => Kernel::Avx512,
            other => other,
        },
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => Kernel::Scalar,
        Kernel::Scalar | Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2 | Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512 | Kernel::Avx512Batch => Kernel::Avx512,
        other => return Err(EmdTrendError::InvalidKernelForBatch(other)),
    })
}

#[inline(always)]
fn compute_average_series(
    src: &[f64],
    avg_kind: AvgKind,
    length: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, EmdTrendError> {
    ma_with_kernel(
        internal_avg_id(avg_kind),
        MaData::Slice(src),
        length,
        kernel,
    )
    .map_err(map_ma_error)
}

#[inline(always)]
fn compute_deviation_ema(
    avg: &[f64],
    src: &[f64],
    length: usize,
    kernel: Kernel,
) -> Result<Vec<f64>, EmdTrendError> {
    let mut abs_dev = alloc_with_nan_prefix(src.len(), 0);
    abs_dev.fill(f64::NAN);
    for i in 0..src.len() {
        if avg[i].is_finite() && src[i].is_finite() {
            abs_dev[i] = (src[i] - avg[i]).abs();
        }
    }
    let input = EmaInput {
        data: EmaData::Slice(&abs_dev),
        params: EmaParams {
            period: Some(length),
        },
    };
    ema_with_kernel(&input, kernel)
        .map(|out| out.values)
        .map_err(|e| EmdTrendError::InvalidInput { msg: e.to_string() })
}

fn compute_from_source_into(
    src: &[f64],
    avg_kind: AvgKind,
    length: usize,
    mult: f64,
    kernel: Kernel,
    direction_out: &mut [f64],
    average_out: &mut [f64],
    upper_out: &mut [f64],
    lower_out: &mut [f64],
) -> Result<(), EmdTrendError> {
    let len = src.len();
    if direction_out.len() != len {
        return Err(EmdTrendError::OutputLengthMismatch {
            expected: len,
            got: direction_out.len(),
        });
    }
    if average_out.len() != len {
        return Err(EmdTrendError::OutputLengthMismatch {
            expected: len,
            got: average_out.len(),
        });
    }
    if upper_out.len() != len {
        return Err(EmdTrendError::OutputLengthMismatch {
            expected: len,
            got: upper_out.len(),
        });
    }
    if lower_out.len() != len {
        return Err(EmdTrendError::OutputLengthMismatch {
            expected: len,
            got: lower_out.len(),
        });
    }

    let average = compute_average_series(src, avg_kind, length, kernel)?;
    let deviation = compute_deviation_ema(&average, src, length, kernel)?;

    average_out.copy_from_slice(&average);
    direction_out.fill(0.0);
    upper_out.fill(f64::NAN);
    lower_out.fill(f64::NAN);

    let mut direction = 0.0f64;
    for i in 0..len {
        if average[i].is_finite() && deviation[i].is_finite() {
            upper_out[i] = average[i] + deviation[i] * mult;
            lower_out[i] = average[i] - deviation[i] * mult;
        }
        if i > 0
            && src[i].is_finite()
            && src[i - 1].is_finite()
            && upper_out[i].is_finite()
            && upper_out[i - 1].is_finite()
            && src[i] > upper_out[i]
            && src[i - 1] <= upper_out[i - 1]
        {
            direction = 1.0;
        } else if i > 0
            && src[i].is_finite()
            && src[i - 1].is_finite()
            && lower_out[i].is_finite()
            && lower_out[i - 1].is_finite()
            && src[i] < lower_out[i]
            && src[i - 1] >= lower_out[i - 1]
        {
            direction = -1.0;
        }
        direction_out[i] = direction;
    }

    Ok(())
}

#[inline]
pub fn emd_trend(input: &EmdTrendInput) -> Result<EmdTrendOutput, EmdTrendError> {
    emd_trend_with_kernel(input, Kernel::Auto)
}

pub fn emd_trend_with_kernel(
    input: &EmdTrendInput,
    kernel: Kernel,
) -> Result<EmdTrendOutput, EmdTrendError> {
    let (open, high, low, close) = input_slices(input)?;
    let (source_kind, avg_kind) = validate_params_only(
        input.get_source(),
        input.get_avg_type(),
        input.get_length(),
        input.get_mult(),
    )?;
    let src = source_series(input, open, high, low, close, source_kind)?;
    let longest = longest_valid_run(&src);
    let needed = if avg_kind == AvgKind::Frama && input.get_length() % 2 == 1 {
        input.get_length() + 1
    } else {
        input.get_length()
    };
    if longest < needed {
        return Err(EmdTrendError::NotEnoughValidData {
            needed,
            valid: longest,
        });
    }

    let chosen = normalize_single_kernel(kernel);
    let mut direction = alloc_uninit_f64(src.len());
    let mut average = alloc_uninit_f64(src.len());
    let mut upper = alloc_uninit_f64(src.len());
    let mut lower = alloc_uninit_f64(src.len());

    compute_from_source_into(
        &src,
        avg_kind,
        input.get_length(),
        input.get_mult(),
        chosen,
        &mut direction,
        &mut average,
        &mut upper,
        &mut lower,
    )?;

    Ok(EmdTrendOutput {
        direction,
        average,
        upper,
        lower,
    })
}

pub fn emd_trend_into_slice(
    out_direction: &mut [f64],
    out_average: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    input: &EmdTrendInput,
    kernel: Kernel,
) -> Result<(), EmdTrendError> {
    let (open, high, low, close) = input_slices(input)?;
    let (source_kind, avg_kind) = validate_params_only(
        input.get_source(),
        input.get_avg_type(),
        input.get_length(),
        input.get_mult(),
    )?;
    let src = source_series(input, open, high, low, close, source_kind)?;
    let longest = longest_valid_run(&src);
    let needed = if avg_kind == AvgKind::Frama && input.get_length() % 2 == 1 {
        input.get_length() + 1
    } else {
        input.get_length()
    };
    if longest < needed {
        return Err(EmdTrendError::NotEnoughValidData {
            needed,
            valid: longest,
        });
    }

    let chosen = normalize_single_kernel(kernel);
    compute_from_source_into(
        &src,
        avg_kind,
        input.get_length(),
        input.get_mult(),
        chosen,
        out_direction,
        out_average,
        out_upper,
        out_lower,
    )
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn emd_trend_into(
    input: &EmdTrendInput,
    out_direction: &mut [f64],
    out_average: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<(), EmdTrendError> {
    emd_trend_into_slice(
        out_direction,
        out_average,
        out_upper,
        out_lower,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone)]
pub struct EmdTrendBatchRange {
    pub length: (usize, usize, usize),
    pub mult: (f64, f64, f64),
}

impl Default for EmdTrendBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            mult: (DEFAULT_MULT, DEFAULT_MULT, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmdTrendBatchOutput {
    pub direction: Vec<f64>,
    pub average: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub combos: Vec<EmdTrendParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct EmdTrendBatchBuilder {
    range: EmdTrendBatchRange,
    source: Option<SourceKind>,
    avg_type: Option<AvgKind>,
    kernel: Kernel,
}

impl Default for EmdTrendBatchBuilder {
    fn default() -> Self {
        Self {
            range: EmdTrendBatchRange::default(),
            source: None,
            avg_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EmdTrendBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn source(mut self, value: &str) -> Result<Self, EmdTrendError> {
        self.source = Some(parse_source_kind(value)?);
        Ok(self)
    }

    pub fn avg_type(mut self, value: &str) -> Result<Self, EmdTrendError> {
        self.avg_type = Some(parse_avg_kind(value)?);
        Ok(self)
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step);
        self
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<EmdTrendBatchOutput, EmdTrendError> {
        emd_trend_batch_with_kernel(
            open,
            high,
            low,
            close,
            &self.range,
            source_kind_name(self.source.unwrap_or(SourceKind::Close)),
            avg_kind_name(self.avg_type.unwrap_or(AvgKind::Sma)),
            self.kernel,
        )
    }

    pub fn apply_candles(self, candles: &Candles) -> Result<EmdTrendBatchOutput, EmdTrendError> {
        self.apply_slices(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )
    }
}

pub fn expand_grid_emd_trend(
    sweep: &EmdTrendBatchRange,
    source: &str,
    avg_type: &str,
) -> Result<Vec<EmdTrendParams>, EmdTrendError> {
    let (source_kind, avg_kind) =
        validate_params_only(source, avg_type, DEFAULT_LENGTH, DEFAULT_MULT)?;
    let lengths = axis_usize(sweep.length);
    let mults = axis_f64(sweep.mult);
    let mut out = Vec::with_capacity(lengths.len().saturating_mul(mults.len()));
    for &length in &lengths {
        if length == 0 {
            return Err(EmdTrendError::InvalidRange {
                start: sweep.length.0.to_string(),
                end: sweep.length.1.to_string(),
                step: sweep.length.2.to_string(),
            });
        }
        for &mult in &mults {
            if !mult.is_finite() || mult < 0.05 {
                return Err(EmdTrendError::InvalidRange {
                    start: sweep.mult.0.to_string(),
                    end: sweep.mult.1.to_string(),
                    step: sweep.mult.2.to_string(),
                });
            }
            out.push(EmdTrendParams {
                source: Some(source_kind_name(source_kind).to_string()),
                avg_type: Some(avg_kind_name(avg_kind).to_string()),
                length: Some(length),
                mult: Some(mult),
            });
        }
    }
    Ok(out)
}

pub fn emd_trend_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &EmdTrendBatchRange,
    source: &str,
    avg_type: &str,
    kernel: Kernel,
) -> Result<EmdTrendBatchOutput, EmdTrendError> {
    let combos = expand_grid_emd_trend(sweep, source, avg_type)?;
    if combos.is_empty() {
        return Err(EmdTrendError::InvalidRange {
            start: sweep.length.0.to_string(),
            end: sweep.length.1.to_string(),
            step: sweep.length.2.to_string(),
        });
    }
    let rows = combos.len();
    let cols = close.len();
    let (source_kind, avg_kind) =
        validate_params_only(source, avg_type, DEFAULT_LENGTH, DEFAULT_MULT)?;
    let src = build_source_series(open, high, low, close, source_kind)?;
    let longest = longest_valid_run(&src);
    for combo in &combos {
        let length = combo.length.unwrap_or(DEFAULT_LENGTH);
        let mult = combo.mult.unwrap_or(DEFAULT_MULT);
        validate_params_only(source, avg_type, length, mult)?;
        let needed = if avg_kind == AvgKind::Frama && length % 2 == 1 {
            length + 1
        } else {
            length
        };
        if longest < needed {
            return Err(EmdTrendError::NotEnoughValidData {
                needed,
                valid: longest,
            });
        }
    }
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| EmdTrendError::InvalidInput {
            msg: "emd_trend: rows*cols overflow in batch".to_string(),
        })?;

    let mut direction_mu = make_uninit_matrix(rows, cols);
    let mut average_mu = make_uninit_matrix(rows, cols);
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);

    let direction =
        unsafe { std::slice::from_raw_parts_mut(direction_mu.as_mut_ptr() as *mut f64, total) };
    let average =
        unsafe { std::slice::from_raw_parts_mut(average_mu.as_mut_ptr() as *mut f64, total) };
    let upper = unsafe { std::slice::from_raw_parts_mut(upper_mu.as_mut_ptr() as *mut f64, total) };
    let lower = unsafe { std::slice::from_raw_parts_mut(lower_mu.as_mut_ptr() as *mut f64, total) };

    emd_trend_batch_inner_into(
        open, high, low, close, sweep, source, avg_type, kernel, false, direction, average, upper,
        lower,
    )?;

    let direction = unsafe {
        Vec::from_raw_parts(
            direction_mu.as_mut_ptr() as *mut f64,
            direction_mu.len(),
            direction_mu.capacity(),
        )
    };
    let average = unsafe {
        Vec::from_raw_parts(
            average_mu.as_mut_ptr() as *mut f64,
            average_mu.len(),
            average_mu.capacity(),
        )
    };
    let upper = unsafe {
        Vec::from_raw_parts(
            upper_mu.as_mut_ptr() as *mut f64,
            upper_mu.len(),
            upper_mu.capacity(),
        )
    };
    let lower = unsafe {
        Vec::from_raw_parts(
            lower_mu.as_mut_ptr() as *mut f64,
            lower_mu.len(),
            lower_mu.capacity(),
        )
    };
    std::mem::forget(direction_mu);
    std::mem::forget(average_mu);
    std::mem::forget(upper_mu);
    std::mem::forget(lower_mu);

    Ok(EmdTrendBatchOutput {
        direction,
        average,
        upper,
        lower,
        combos,
        rows,
        cols,
    })
}

pub fn emd_trend_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &EmdTrendBatchRange,
    source: &str,
    avg_type: &str,
    kernel: Kernel,
) -> Result<EmdTrendBatchOutput, EmdTrendError> {
    emd_trend_batch_with_kernel(open, high, low, close, sweep, source, avg_type, kernel)
}

pub fn emd_trend_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &EmdTrendBatchRange,
    source: &str,
    avg_type: &str,
    kernel: Kernel,
) -> Result<EmdTrendBatchOutput, EmdTrendError> {
    emd_trend_batch_with_kernel(open, high, low, close, sweep, source, avg_type, kernel)
}

fn emd_trend_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &EmdTrendBatchRange,
    source: &str,
    avg_type: &str,
    kernel: Kernel,
    _parallel: bool,
    out_direction: &mut [f64],
    out_average: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<Vec<EmdTrendParams>, EmdTrendError> {
    let (source_kind, avg_kind) =
        validate_params_only(source, avg_type, DEFAULT_LENGTH, DEFAULT_MULT)?;
    let src = build_source_series(open, high, low, close, source_kind)?;
    let longest = longest_valid_run(&src);
    if longest == 0 {
        return Err(EmdTrendError::AllValuesNaN);
    }

    let combos = expand_grid_emd_trend(sweep, source, avg_type)?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| EmdTrendError::InvalidInput {
            msg: "emd_trend: rows*cols overflow in batch_into".to_string(),
        })?;

    if out_direction.len() != total {
        return Err(EmdTrendError::MismatchedOutputLen {
            dst_len: out_direction.len(),
            expected_len: total,
        });
    }
    if out_average.len() != total {
        return Err(EmdTrendError::MismatchedOutputLen {
            dst_len: out_average.len(),
            expected_len: total,
        });
    }
    if out_upper.len() != total {
        return Err(EmdTrendError::MismatchedOutputLen {
            dst_len: out_upper.len(),
            expected_len: total,
        });
    }
    if out_lower.len() != total {
        return Err(EmdTrendError::MismatchedOutputLen {
            dst_len: out_lower.len(),
            expected_len: total,
        });
    }

    let chosen = normalize_batch_kernel(kernel)?;
    for (row, combo) in combos.iter().enumerate() {
        let length = combo.length.unwrap_or(DEFAULT_LENGTH);
        let mult = combo.mult.unwrap_or(DEFAULT_MULT);
        let needed = if avg_kind == AvgKind::Frama && length % 2 == 1 {
            length + 1
        } else {
            length
        };
        if longest < needed {
            return Err(EmdTrendError::NotEnoughValidData {
                needed,
                valid: longest,
            });
        }

        let start = row * cols;
        let end = start + cols;
        compute_from_source_into(
            &src,
            avg_kind,
            length,
            mult,
            chosen,
            &mut out_direction[start..end],
            &mut out_average[start..end],
            &mut out_upper[start..end],
            &mut out_lower[start..end],
        )?;
    }

    Ok(combos)
}

pub struct EmdTrendStream {
    source_kind: SourceKind,
    avg_kind: AvgKind,
    direction: f64,
    length: usize,
    mult: f64,
    src_history: Vec<f64>,
}

impl EmdTrendStream {
    pub fn try_new(params: EmdTrendParams) -> Result<Self, EmdTrendError> {
        let source = params.source.as_deref().unwrap_or(DEFAULT_SOURCE);
        let avg_type = params.avg_type.as_deref().unwrap_or(DEFAULT_AVG_TYPE);
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let mult = params.mult.unwrap_or(DEFAULT_MULT);
        let (source_kind, avg_kind) = validate_params_only(source, avg_type, length, mult)?;

        Ok(Self {
            source_kind,
            avg_kind,
            direction: 0.0,
            length,
            mult,
            src_history: Vec::new(),
        })
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64)> {
        let src = source_value(self.source_kind, open, high, low, close);
        if !src.is_finite() {
            return None;
        }
        self.src_history.push(src);
        let len = self.src_history.len();
        let mut direction = vec![0.0; len];
        let mut average = vec![f64::NAN; len];
        let mut upper = vec![f64::NAN; len];
        let mut lower = vec![f64::NAN; len];

        match compute_from_source_into(
            &self.src_history,
            self.avg_kind,
            self.length,
            self.mult,
            Kernel::Scalar,
            &mut direction,
            &mut average,
            &mut upper,
            &mut lower,
        ) {
            Ok(()) => {
                self.direction = direction[len - 1];
                Some((
                    self.direction,
                    average[len - 1],
                    upper[len - 1],
                    lower[len - 1],
                ))
            }
            Err(EmdTrendError::NotEnoughValidData { .. }) => {
                Some((self.direction, f64::NAN, f64::NAN, f64::NAN))
            }
            Err(_) => Some((self.direction, f64::NAN, f64::NAN, f64::NAN)),
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "EmdTrendStream")]
pub struct EmdTrendStreamPy {
    inner: EmdTrendStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EmdTrendStreamPy {
    #[new]
    fn new(
        source: Option<&str>,
        avg_type: Option<&str>,
        length: usize,
        mult: f64,
    ) -> PyResult<Self> {
        let inner = EmdTrendStream::try_new(EmdTrendParams {
            source: source.map(str::to_string),
            avg_type: avg_type.map(str::to_string),
            length: Some(length),
            mult: Some(mult),
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
    ) -> Option<(f64, f64, f64, f64)> {
        self.inner.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "emd_trend",
    signature = (open, high, low, close, source="close", avg_type="SMA", length=28, mult=1.0, kernel=None)
)]
pub fn emd_trend_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    source: &str,
    avg_type: &str,
    length: usize,
    mult: f64,
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
    let input = EmdTrendInput::from_slices(
        open,
        high,
        low,
        close,
        EmdTrendParams {
            source: Some(source.to_string()),
            avg_type: Some(avg_type.to_string()),
            length: Some(length),
            mult: Some(mult),
        },
    );
    let kern = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| emd_trend_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.direction.into_pyarray(py),
        out.average.into_pyarray(py),
        out.upper.into_pyarray(py),
        out.lower.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "emd_trend_batch",
    signature = (open, high, low, close, length_range=(28, 28, 0), mult_range=(1.0, 1.0, 0.0), source="close", avg_type="SMA", kernel=None)
)]
pub fn emd_trend_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    source: &str,
    avg_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let batch = py
        .allow_threads(|| {
            emd_trend_batch_with_kernel(
                open,
                high,
                low,
                close,
                &EmdTrendBatchRange {
                    length: length_range,
                    mult: mult_range,
                },
                source,
                avg_type,
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    let rows = batch.rows;
    let cols = batch.cols;
    dict.set_item(
        "direction",
        PyArray1::from_vec(py, batch.direction).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "average",
        PyArray1::from_vec(py, batch.average).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "upper",
        PyArray1::from_vec(py, batch.upper).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "lower",
        PyArray1::from_vec(py, batch.lower).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "lengths",
        batch
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "mults",
        batch
            .combos
            .iter()
            .map(|params| params.mult.unwrap_or(DEFAULT_MULT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_emd_trend_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(emd_trend_py, m)?)?;
    m.add_function(wrap_pyfunction!(emd_trend_batch_py, m)?)?;
    m.add_class::<EmdTrendStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EmdTrendBatchConfig {
    pub length_range: Vec<usize>,
    pub mult_range: Vec<f64>,
    pub source: Option<String>,
    pub avg_type: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = emd_trend_js)]
pub fn emd_trend_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &str,
    avg_type: &str,
    length: usize,
    mult: f64,
) -> Result<JsValue, JsValue> {
    let out = emd_trend_with_kernel(
        &EmdTrendInput::from_slices(
            open,
            high,
            low,
            close,
            EmdTrendParams {
                source: Some(source.to_string()),
                avg_type: Some(avg_type.to_string()),
                length: Some(length),
                mult: Some(mult),
            },
        ),
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("direction"),
        &serde_wasm_bindgen::to_value(&out.direction).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("average"),
        &serde_wasm_bindgen::to_value(&out.average).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("upper"),
        &serde_wasm_bindgen::to_value(&out.upper).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("lower"),
        &serde_wasm_bindgen::to_value(&out.lower).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = emd_trend_batch_js)]
pub fn emd_trend_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EmdTrendBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.mult_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }
    let source = config.source.as_deref().unwrap_or(DEFAULT_SOURCE);
    let avg_type = config.avg_type.as_deref().unwrap_or(DEFAULT_AVG_TYPE);
    let out = emd_trend_batch_with_kernel(
        open,
        high,
        low,
        close,
        &EmdTrendBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
            mult: (
                config.mult_range[0],
                config.mult_range[1],
                config.mult_range[2],
            ),
        },
        source,
        avg_type,
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("direction"),
        &serde_wasm_bindgen::to_value(&out.direction).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("average"),
        &serde_wasm_bindgen::to_value(&out.average).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("upper"),
        &serde_wasm_bindgen::to_value(&out.upper).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("lower"),
        &serde_wasm_bindgen::to_value(&out.lower).unwrap(),
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
pub fn emd_trend_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(4 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emd_trend_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 4 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emd_trend_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    source: &str,
    avg_type: &str,
    length: usize,
    mult: f64,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer passed to emd_trend_into"));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 4 * len);
        let (direction, tail) = out.split_at_mut(len);
        let (average, tail) = tail.split_at_mut(len);
        let (upper, lower) = tail.split_at_mut(len);
        emd_trend_into_slice(
            direction,
            average,
            upper,
            lower,
            &EmdTrendInput::from_slices(
                open,
                high,
                low,
                close,
                EmdTrendParams {
                    source: Some(source.to_string()),
                    avg_type: Some(avg_type.to_string()),
                    length: Some(length),
                    mult: Some(mult),
                },
            ),
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emd_trend_batch_into(
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
    avg_type: &str,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to emd_trend_batch_into",
        ));
    }

    let sweep = EmdTrendBatchRange {
        length: (length_start, length_end, length_step),
        mult: (mult_start, mult_end, mult_step),
    };
    let combos = expand_grid_emd_trend(&sweep, source, avg_type)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let split = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow in emd_trend_batch_into"))?;
    let total = split
        .checked_mul(4)
        .ok_or_else(|| JsValue::from_str("4*rows*cols overflow in emd_trend_batch_into"))?;

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (direction, tail) = out.split_at_mut(split);
        let (average, tail) = tail.split_at_mut(split);
        let (upper, lower) = tail.split_at_mut(split);
        emd_trend_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            source,
            avg_type,
            Kernel::Auto,
            false,
            direction,
            average,
            upper,
            lower,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emd_trend_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &str,
    avg_type: &str,
    length: usize,
    mult: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = emd_trend_js(open, high, low, close, source, avg_type, length, mult)?;
    crate::write_wasm_object_f64_outputs("emd_trend_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn emd_trend_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = emd_trend_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs("emd_trend_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, ParamKV,
        ParamValue,
    };

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let x = i as f64;
            let base = 100.0 + x * 0.04 + (x * 0.11).sin() * 1.8;
            let o = base + (x * 0.17).cos() * 0.6;
            let c = base + (x * 0.09).sin() * 0.9;
            let h = o.max(c) + 0.8 + (x * 0.05).cos().abs() * 0.4;
            let l = o.min(c) - 0.8 - (x * 0.07).sin().abs() * 0.3;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
        }
        (open, high, low, close)
    }

    fn assert_vec_eq_nan(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&l, &r)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if l.is_nan() && r.is_nan() {
                continue;
            }
            assert!(
                (l - r).abs() <= 1e-10,
                "mismatch at {idx}: lhs={l}, rhs={r}"
            );
        }
    }

    #[test]
    fn emd_trend_output_contract() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(256);
        let out = emd_trend_with_kernel(
            &EmdTrendInput::from_slices(&open, &high, &low, &close, EmdTrendParams::default()),
            Kernel::Scalar,
        )?;
        assert_eq!(out.direction.len(), close.len());
        assert_eq!(out.average.len(), close.len());
        assert_eq!(out.upper.len(), close.len());
        assert_eq!(out.lower.len(), close.len());
        assert!(out.average.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn emd_trend_exact_small_case() -> Result<(), Box<dyn Error>> {
        let open = vec![1.0, 1.0, 1.0, 10.0, 10.0];
        let high = vec![1.0, 1.0, 1.0, 10.0, 10.0];
        let low = vec![1.0, 1.0, 1.0, 10.0, 10.0];
        let close = vec![1.0, 1.0, 1.0, 10.0, 10.0];
        let out = emd_trend_with_kernel(
            &EmdTrendInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                EmdTrendParams {
                    source: Some("close".to_string()),
                    avg_type: Some("SMA".to_string()),
                    length: Some(2),
                    mult: Some(1.0),
                },
            ),
            Kernel::Scalar,
        )?;
        assert_vec_eq_nan(&out.direction, &[0.0, 0.0, 0.0, 1.0, 1.0]);
        assert_vec_eq_nan(&out.average, &[f64::NAN, 1.0, 1.0, 5.5, 10.0]);
        assert_vec_eq_nan(&out.upper, &[f64::NAN, 1.0, 1.0, 8.5, 11.0]);
        assert_vec_eq_nan(&out.lower, &[f64::NAN, 1.0, 1.0, 2.5, 9.0]);
        Ok(())
    }

    #[test]
    fn emd_trend_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(200);
        let input =
            EmdTrendInput::from_slices(&open, &high, &low, &close, EmdTrendParams::default());
        let base = emd_trend(&input)?;
        let mut direction = vec![0.0; close.len()];
        let mut average = vec![f64::NAN; close.len()];
        let mut upper = vec![f64::NAN; close.len()];
        let mut lower = vec![f64::NAN; close.len()];
        emd_trend_into(&input, &mut direction, &mut average, &mut upper, &mut lower)?;
        assert_vec_eq_nan(&direction, &base.direction);
        assert_vec_eq_nan(&average, &base.average);
        assert_vec_eq_nan(&upper, &base.upper);
        assert_vec_eq_nan(&lower, &base.lower);
        Ok(())
    }

    #[test]
    fn emd_trend_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(180);
        let single = emd_trend_with_kernel(
            &EmdTrendInput::from_slices(&open, &high, &low, &close, EmdTrendParams::default()),
            Kernel::Scalar,
        )?;
        let batch = emd_trend_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &EmdTrendBatchRange::default(),
            "close",
            "SMA",
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_vec_eq_nan(&batch.direction[..close.len()], &single.direction);
        assert_vec_eq_nan(&batch.average[..close.len()], &single.average);
        assert_vec_eq_nan(&batch.upper[..close.len()], &single.upper);
        assert_vec_eq_nan(&batch.lower[..close.len()], &single.lower);
        Ok(())
    }

    #[test]
    fn emd_trend_stream_contract() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(220);
        let mut stream = EmdTrendStream::try_new(EmdTrendParams {
            source: Some("close".to_string()),
            avg_type: Some("SMA".to_string()),
            length: Some(28),
            mult: Some(1.0),
        })?;
        let mut points = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            points.push(stream.update(open[i], high[i], low[i], close[i]).unwrap());
        }
        assert_eq!(points.len(), close.len());
        assert!(points.iter().any(|(_, avg, _, _)| avg.is_finite()));
        Ok(())
    }

    #[test]
    fn emd_trend_rejects_invalid_params() {
        let (open, high, low, close) = sample_ohlc(64);
        let err = emd_trend(&EmdTrendInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            EmdTrendParams {
                source: Some("bad".to_string()),
                avg_type: Some("SMA".to_string()),
                length: Some(28),
                mult: Some(1.0),
            },
        ))
        .unwrap_err();
        assert!(matches!(err, EmdTrendError::InvalidSource { .. }));
    }

    #[test]
    fn emd_trend_dispatch_compute_returns_average() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(192);
        let params = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("close"),
            },
            ParamKV {
                key: "avg_type",
                value: ParamValue::EnumString("SMA"),
            },
            ParamKV {
                key: "length",
                value: ParamValue::Int(28),
            },
            ParamKV {
                key: "mult",
                value: ParamValue::Float(1.0),
            },
        ];
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "emd_trend",
            output_id: Some("average"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &params,
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(out.output_id, "average");
        match out.series {
            IndicatorSeries::F64(values) => assert_eq!(values.len(), close.len()),
            other => panic!("expected f64 series, got {:?}", other),
        }
        Ok(())
    }
}
