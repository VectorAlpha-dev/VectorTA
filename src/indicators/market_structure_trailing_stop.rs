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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

const RESET_ON_CHOCH: &str = "CHoCH";
const RESET_ON_ALL: &str = "All";

#[derive(Debug, Clone)]
pub enum MarketStructureTrailingStopData<'a> {
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
pub struct MarketStructureTrailingStopOutput {
    pub trailing_stop: Vec<f64>,
    pub state: Vec<f64>,
    pub structure: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MarketStructureTrailingStopParams {
    pub length: Option<usize>,
    pub increment_factor: Option<f64>,
    pub reset_on: Option<String>,
}

impl Default for MarketStructureTrailingStopParams {
    fn default() -> Self {
        Self {
            length: Some(14),
            increment_factor: Some(100.0),
            reset_on: Some(RESET_ON_CHOCH.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketStructureTrailingStopInput<'a> {
    pub data: MarketStructureTrailingStopData<'a>,
    pub params: MarketStructureTrailingStopParams,
}

impl<'a> MarketStructureTrailingStopInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: MarketStructureTrailingStopParams) -> Self {
        Self {
            data: MarketStructureTrailingStopData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: MarketStructureTrailingStopParams,
    ) -> Self {
        Self {
            data: MarketStructureTrailingStopData::Slices {
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
        Self::from_candles(candles, MarketStructureTrailingStopParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }

    #[inline]
    pub fn get_increment_factor(&self) -> f64 {
        self.params.increment_factor.unwrap_or(100.0)
    }

    #[inline]
    pub fn get_reset_on(&self) -> &str {
        self.params.reset_on.as_deref().unwrap_or(RESET_ON_CHOCH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MarketStructureTrailingStopBuilder {
    length: Option<usize>,
    increment_factor: Option<f64>,
    reset_on: Option<&'static str>,
    kernel: Kernel,
}

impl Default for MarketStructureTrailingStopBuilder {
    fn default() -> Self {
        Self {
            length: None,
            increment_factor: None,
            reset_on: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MarketStructureTrailingStopBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline(always)]
    pub fn increment_factor(mut self, value: f64) -> Self {
        self.increment_factor = Some(value);
        self
    }

    #[inline(always)]
    pub fn reset_on(mut self, value: &str) -> Result<Self, MarketStructureTrailingStopError> {
        self.reset_on = Some(canonical_reset_on(value)?);
        Ok(self)
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
    ) -> Result<MarketStructureTrailingStopOutput, MarketStructureTrailingStopError> {
        market_structure_trailing_stop_with_kernel(
            &MarketStructureTrailingStopInput::from_candles(
                candles,
                MarketStructureTrailingStopParams {
                    length: self.length,
                    increment_factor: self.increment_factor,
                    reset_on: self.reset_on.map(str::to_string),
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
    ) -> Result<MarketStructureTrailingStopOutput, MarketStructureTrailingStopError> {
        market_structure_trailing_stop_with_kernel(
            &MarketStructureTrailingStopInput::from_slices(
                open,
                high,
                low,
                close,
                MarketStructureTrailingStopParams {
                    length: self.length,
                    increment_factor: self.increment_factor,
                    reset_on: self.reset_on.map(str::to_string),
                },
            ),
            self.kernel,
        )
    }
}

#[derive(Debug, Error)]
pub enum MarketStructureTrailingStopError {
    #[error("market_structure_trailing_stop: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "market_structure_trailing_stop: Input length mismatch: open = {open_len}, high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    InputLengthMismatch {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("market_structure_trailing_stop: All values are NaN.")]
    AllValuesNaN,
    #[error("market_structure_trailing_stop: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("market_structure_trailing_stop: Invalid increment_factor: {increment_factor}")]
    InvalidIncrementFactor { increment_factor: f64 },
    #[error("market_structure_trailing_stop: Invalid reset_on: {value}")]
    InvalidResetOn { value: String },
    #[error(
        "market_structure_trailing_stop: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "market_structure_trailing_stop: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "market_structure_trailing_stop: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error(
        "market_structure_trailing_stop: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("market_structure_trailing_stop: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("market_structure_trailing_stop: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResetOnMode {
    Choch,
    All,
}

#[derive(Clone, Copy, Debug)]
struct ValidationStats {
    reset_mode: ResetOnMode,
    first_valid: usize,
    longest_valid: usize,
}

#[inline(always)]
fn is_valid_ohlc(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline(always)]
fn longest_valid_run(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for (((&o, &h), &l), &c) in open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
    {
        if is_valid_ohlc(o, h, l, c) {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn valid_run_stats(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Option<(usize, usize)> {
    let mut best = 0usize;
    let mut cur = 0usize;
    let mut first_valid = usize::MAX;
    for i in 0..close.len() {
        if is_valid_ohlc(open[i], high[i], low[i], close[i]) {
            if first_valid == usize::MAX {
                first_valid = i;
            }
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    if first_valid == usize::MAX {
        None
    } else {
        Some((first_valid, best))
    }
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a MarketStructureTrailingStopInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), MarketStructureTrailingStopError> {
    match &input.data {
        MarketStructureTrailingStopData::Candles { candles } => Ok((
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        MarketStructureTrailingStopData::Slices {
            open,
            high,
            low,
            close,
        } => Ok((open, high, low, close)),
    }
}

#[inline(always)]
fn canonical_reset_on(value: &str) -> Result<&'static str, MarketStructureTrailingStopError> {
    if value.eq_ignore_ascii_case(RESET_ON_CHOCH) {
        return Ok(RESET_ON_CHOCH);
    }
    if value.eq_ignore_ascii_case(RESET_ON_ALL) {
        return Ok(RESET_ON_ALL);
    }
    Err(MarketStructureTrailingStopError::InvalidResetOn {
        value: value.to_string(),
    })
}

#[inline(always)]
fn parse_reset_on(value: &str) -> Result<ResetOnMode, MarketStructureTrailingStopError> {
    match canonical_reset_on(value)? {
        RESET_ON_CHOCH => Ok(ResetOnMode::Choch),
        RESET_ON_ALL => Ok(ResetOnMode::All),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn validate_common(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    increment_factor: f64,
    reset_on: &str,
) -> Result<ResetOnMode, MarketStructureTrailingStopError> {
    Ok(
        validate_common_stats(open, high, low, close, length, increment_factor, reset_on)?
            .reset_mode,
    )
}

#[inline(always)]
fn validate_common_stats(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    increment_factor: f64,
    reset_on: &str,
) -> Result<ValidationStats, MarketStructureTrailingStopError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(MarketStructureTrailingStopError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(MarketStructureTrailingStopError::InputLengthMismatch {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    if length == 0 {
        return Err(MarketStructureTrailingStopError::InvalidLength { length });
    }
    if !increment_factor.is_finite() || increment_factor < 0.0 {
        return Err(MarketStructureTrailingStopError::InvalidIncrementFactor { increment_factor });
    }
    let reset_mode = parse_reset_on(reset_on)?;
    let (first_valid, longest) = valid_run_stats(open, high, low, close)
        .ok_or(MarketStructureTrailingStopError::AllValuesNaN)?;
    let needed = length
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| MarketStructureTrailingStopError::InvalidInput {
            msg: "market_structure_trailing_stop: length overflow".to_string(),
        })?;
    if longest < needed {
        return Err(MarketStructureTrailingStopError::NotEnoughValidData {
            needed,
            valid: longest,
        });
    }
    Ok(ValidationStats {
        reset_mode,
        first_valid,
        longest_valid: longest,
    })
}

#[inline(always)]
fn is_pivot_high(high: &[f64], center: usize, length: usize) -> bool {
    unsafe {
        let ptr = high.as_ptr();
        let pivot = *ptr.add(center);
        let mut idx = center - length;
        while idx < center {
            if *ptr.add(idx) > pivot {
                return false;
            }
            idx += 1;
        }
        idx = center + 1;
        let end = center + length;
        while idx <= end {
            if *ptr.add(idx) >= pivot {
                return false;
            }
            idx += 1;
        }
    }
    true
}

#[inline(always)]
fn is_pivot_low(low: &[f64], center: usize, length: usize) -> bool {
    unsafe {
        let ptr = low.as_ptr();
        let pivot = *ptr.add(center);
        let mut idx = center - length;
        while idx < center {
            if *ptr.add(idx) < pivot {
                return false;
            }
            idx += 1;
        }
        idx = center + 1;
        let end = center + length;
        while idx <= end {
            if *ptr.add(idx) <= pivot {
                return false;
            }
            idx += 1;
        }
    }
    true
}

fn compute_run(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    increment_factor: f64,
    reset_mode: ResetOnMode,
    trailing_stop: &mut [f64],
    state: &mut [f64],
    structure: &mut [f64],
) {
    let n = close.len();
    if n < (2 * length + 1) {
        return;
    }

    let incr = increment_factor / 100.0;
    let mut ph_y = f64::NAN;
    let mut ph_x = 0usize;
    let mut pl_y = f64::NAN;
    let mut pl_x = 0usize;
    let mut ph_cross = false;
    let mut pl_cross = false;
    let mut top = f64::NAN;
    let mut btm = f64::NAN;
    let mut max_close = f64::NAN;
    let mut min_close = f64::NAN;
    let mut ts = f64::NAN;
    let mut os: i8 = 0;

    for idx in 0..n {
        let mut ms: i8 = 0;

        if idx >= 2 * length {
            let center = idx - length;
            if is_pivot_high(high, center, length) {
                ph_y = high[center];
                ph_x = center;
                ph_cross = false;
            }
            if is_pivot_low(low, center, length) {
                pl_y = low[center];
                pl_x = center;
                pl_cross = false;
            }
        }

        let c = close[idx];

        if ph_y.is_finite() && !ph_cross && c > ph_y {
            ms = match reset_mode {
                ResetOnMode::Choch if os == -1 => 1,
                ResetOnMode::All => 1,
                _ => 0,
            };
            ph_cross = true;
            os = 1;
            btm = low[idx];
            let mut scan = idx;
            while scan > ph_x {
                btm = btm.min(low[scan]);
                scan -= 1;
            }
        }

        if pl_y.is_finite() && !pl_cross && c < pl_y {
            ms = match reset_mode {
                ResetOnMode::Choch if os == 1 => -1,
                ResetOnMode::All => -1,
                _ => 0,
            };
            pl_cross = true;
            os = -1;
            top = high[idx];
            let mut scan = idx;
            while scan > pl_x {
                top = top.max(high[scan]);
                scan -= 1;
            }
        }

        let prev_max = max_close;
        let prev_min = min_close;

        if ms == 1 {
            max_close = c;
        } else if ms == -1 {
            min_close = c;
        } else {
            if max_close.is_finite() && c > max_close {
                max_close = c;
            }
            if min_close.is_finite() && c < min_close {
                min_close = c;
            }
        }

        ts = if ms == 1 {
            btm
        } else if ms == -1 {
            top
        } else if os == 1 {
            if ts.is_finite() && max_close.is_finite() && prev_max.is_finite() {
                ts + (max_close - prev_max) * incr
            } else {
                f64::NAN
            }
        } else if ts.is_finite() && min_close.is_finite() && prev_min.is_finite() {
            ts + (min_close - prev_min) * incr
        } else {
            f64::NAN
        };

        trailing_stop[idx] = ts;
        state[idx] = os as f64;
        structure[idx] = ms as f64;
    }
}

fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    increment_factor: f64,
    reset_mode: ResetOnMode,
    trailing_stop: &mut [f64],
    state: &mut [f64],
    structure: &mut [f64],
) {
    let mut idx = 0usize;
    while idx < close.len() {
        while idx < close.len() && !is_valid_ohlc(open[idx], high[idx], low[idx], close[idx]) {
            idx += 1;
        }
        let start = idx;
        while idx < close.len() && is_valid_ohlc(open[idx], high[idx], low[idx], close[idx]) {
            idx += 1;
        }
        let end = idx;
        if end > start {
            compute_run(
                &high[start..end],
                &low[start..end],
                &close[start..end],
                length,
                increment_factor,
                reset_mode,
                &mut trailing_stop[start..end],
                &mut state[start..end],
                &mut structure[start..end],
            );
        }
    }
}

#[inline]
pub fn market_structure_trailing_stop(
    input: &MarketStructureTrailingStopInput,
) -> Result<MarketStructureTrailingStopOutput, MarketStructureTrailingStopError> {
    market_structure_trailing_stop_with_kernel(input, Kernel::Auto)
}

pub fn market_structure_trailing_stop_with_kernel(
    input: &MarketStructureTrailingStopInput,
    kernel: Kernel,
) -> Result<MarketStructureTrailingStopOutput, MarketStructureTrailingStopError> {
    let (open, high, low, close) = input_slices(input)?;
    let length = input.get_length();
    let increment_factor = input.get_increment_factor();
    let stats = validate_common_stats(
        open,
        high,
        low,
        close,
        length,
        increment_factor,
        input.get_reset_on(),
    )?;
    let reset_mode = stats.reset_mode;
    let clean_tail = stats.longest_valid == close.len() - stats.first_valid;

    let prefix = if clean_tail {
        stats.first_valid
    } else {
        close.len()
    };
    let mut trailing_stop = alloc_with_nan_prefix(close.len(), prefix);
    let mut state = alloc_with_nan_prefix(close.len(), prefix);
    let mut structure = alloc_with_nan_prefix(close.len(), prefix);

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    if clean_tail {
        let start = stats.first_valid;
        compute_run(
            &high[start..],
            &low[start..],
            &close[start..],
            length,
            increment_factor,
            reset_mode,
            &mut trailing_stop[start..],
            &mut state[start..],
            &mut structure[start..],
        );
    } else {
        compute_row(
            open,
            high,
            low,
            close,
            length,
            increment_factor,
            reset_mode,
            &mut trailing_stop,
            &mut state,
            &mut structure,
        );
    }

    Ok(MarketStructureTrailingStopOutput {
        trailing_stop,
        state,
        structure,
    })
}

pub fn market_structure_trailing_stop_into_slice(
    out_trailing_stop: &mut [f64],
    out_state: &mut [f64],
    out_structure: &mut [f64],
    input: &MarketStructureTrailingStopInput,
    kernel: Kernel,
) -> Result<(), MarketStructureTrailingStopError> {
    let (open, high, low, close) = input_slices(input)?;
    let length = input.get_length();
    let increment_factor = input.get_increment_factor();
    let stats = validate_common_stats(
        open,
        high,
        low,
        close,
        length,
        increment_factor,
        input.get_reset_on(),
    )?;
    let reset_mode = stats.reset_mode;
    let clean_tail = stats.longest_valid == close.len() - stats.first_valid;

    if out_trailing_stop.len() != close.len() {
        return Err(MarketStructureTrailingStopError::OutputLengthMismatch {
            expected: close.len(),
            got: out_trailing_stop.len(),
        });
    }
    if out_state.len() != close.len() {
        return Err(MarketStructureTrailingStopError::OutputLengthMismatch {
            expected: close.len(),
            got: out_state.len(),
        });
    }
    if out_structure.len() != close.len() {
        return Err(MarketStructureTrailingStopError::OutputLengthMismatch {
            expected: close.len(),
            got: out_structure.len(),
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    if clean_tail {
        let start = stats.first_valid;
        for dst in [
            &mut *out_trailing_stop,
            &mut *out_state,
            &mut *out_structure,
        ] {
            for value in &mut dst[..start] {
                *value = f64::NAN;
            }
        }
        compute_run(
            &high[start..],
            &low[start..],
            &close[start..],
            length,
            increment_factor,
            reset_mode,
            &mut out_trailing_stop[start..],
            &mut out_state[start..],
            &mut out_structure[start..],
        );
    } else {
        out_trailing_stop.fill(f64::NAN);
        out_state.fill(f64::NAN);
        out_structure.fill(f64::NAN);
        compute_row(
            open,
            high,
            low,
            close,
            length,
            increment_factor,
            reset_mode,
            out_trailing_stop,
            out_state,
            out_structure,
        );
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn market_structure_trailing_stop_into(
    input: &MarketStructureTrailingStopInput,
    out_trailing_stop: &mut [f64],
    out_state: &mut [f64],
    out_structure: &mut [f64],
) -> Result<(), MarketStructureTrailingStopError> {
    market_structure_trailing_stop_into_slice(
        out_trailing_stop,
        out_state,
        out_structure,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, Copy)]
pub struct MarketStructureTrailingStopBatchRange {
    pub length: (usize, usize, usize),
    pub increment_factor: (f64, f64, f64),
}

impl Default for MarketStructureTrailingStopBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 14, 0),
            increment_factor: (100.0, 100.0, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketStructureTrailingStopBatchOutput {
    pub trailing_stop: Vec<f64>,
    pub state: Vec<f64>,
    pub structure: Vec<f64>,
    pub combos: Vec<MarketStructureTrailingStopParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MarketStructureTrailingStopBatchBuilder {
    range: MarketStructureTrailingStopBatchRange,
    reset_on: Option<&'static str>,
    kernel: Kernel,
}

impl Default for MarketStructureTrailingStopBatchBuilder {
    fn default() -> Self {
        Self {
            range: MarketStructureTrailingStopBatchRange::default(),
            reset_on: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MarketStructureTrailingStopBatchBuilder {
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
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn increment_factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.increment_factor = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn increment_factor_static(mut self, value: f64) -> Self {
        self.range.increment_factor = (value, value, 0.0);
        self
    }

    #[inline(always)]
    pub fn reset_on(mut self, value: &str) -> Result<Self, MarketStructureTrailingStopError> {
        self.reset_on = Some(canonical_reset_on(value)?);
        Ok(self)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<MarketStructureTrailingStopBatchOutput, MarketStructureTrailingStopError> {
        market_structure_trailing_stop_batch_with_kernel(
            open,
            high,
            low,
            close,
            &self.range,
            self.reset_on.unwrap_or(RESET_ON_CHOCH),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<MarketStructureTrailingStopBatchOutput, MarketStructureTrailingStopError> {
        market_structure_trailing_stop_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.reset_on.unwrap_or(RESET_ON_CHOCH),
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
) -> Result<Vec<usize>, MarketStructureTrailingStopError> {
    if start == 0 || end == 0 {
        return Err(MarketStructureTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(MarketStructureTrailingStopError::InvalidRange {
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
            return Err(MarketStructureTrailingStopError::InvalidRange {
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
fn expand_f64_range(
    field: &'static str,
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, MarketStructureTrailingStopError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(MarketStructureTrailingStopError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(MarketStructureTrailingStopError::InvalidRange {
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
            return Err(MarketStructureTrailingStopError::InvalidRange {
                start: field.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current = if next > end { end } else { next };
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_checked(
    range: &MarketStructureTrailingStopBatchRange,
    reset_on: &str,
) -> Result<Vec<MarketStructureTrailingStopParams>, MarketStructureTrailingStopError> {
    let lengths = expand_usize_range("length", range.length.0, range.length.1, range.length.2)?;
    let increment_factors = expand_f64_range(
        "increment_factor",
        range.increment_factor.0,
        range.increment_factor.1,
        range.increment_factor.2,
    )?;
    let reset_on = canonical_reset_on(reset_on)?;

    let mut out = Vec::new();
    for &length in &lengths {
        for &increment_factor in &increment_factors {
            out.push(MarketStructureTrailingStopParams {
                length: Some(length),
                increment_factor: Some(increment_factor),
                reset_on: Some(reset_on.to_string()),
            });
        }
    }
    Ok(out)
}

pub fn expand_grid_market_structure_trailing_stop(
    range: &MarketStructureTrailingStopBatchRange,
    reset_on: &str,
) -> Vec<MarketStructureTrailingStopParams> {
    expand_grid_checked(range, reset_on).unwrap_or_default()
}

pub fn market_structure_trailing_stop_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MarketStructureTrailingStopBatchRange,
    reset_on: &str,
    kernel: Kernel,
) -> Result<MarketStructureTrailingStopBatchOutput, MarketStructureTrailingStopError> {
    market_structure_trailing_stop_batch_inner(
        open, high, low, close, sweep, reset_on, kernel, true,
    )
}

pub fn market_structure_trailing_stop_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MarketStructureTrailingStopBatchRange,
    reset_on: &str,
    kernel: Kernel,
) -> Result<MarketStructureTrailingStopBatchOutput, MarketStructureTrailingStopError> {
    market_structure_trailing_stop_batch_inner(
        open, high, low, close, sweep, reset_on, kernel, false,
    )
}

pub fn market_structure_trailing_stop_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MarketStructureTrailingStopBatchRange,
    reset_on: &str,
    kernel: Kernel,
) -> Result<MarketStructureTrailingStopBatchOutput, MarketStructureTrailingStopError> {
    market_structure_trailing_stop_batch_inner(
        open, high, low, close, sweep, reset_on, kernel, true,
    )
}

fn market_structure_trailing_stop_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MarketStructureTrailingStopBatchRange,
    reset_on: &str,
    kernel: Kernel,
    parallel: bool,
) -> Result<MarketStructureTrailingStopBatchOutput, MarketStructureTrailingStopError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => {
            return Err(MarketStructureTrailingStopError::InvalidKernelForBatch(
                other,
            ))
        }
    }

    let combos = expand_grid_checked(sweep, reset_on)?;
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(14))
        .max()
        .unwrap_or(0);
    let max_increment = combos
        .iter()
        .map(|params| params.increment_factor.unwrap_or(100.0))
        .fold(0.0_f64, f64::max);
    validate_common(open, high, low, close, max_length, max_increment, reset_on)?;

    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| MarketStructureTrailingStopError::InvalidInput {
                msg: "market_structure_trailing_stop: rows*cols overflow in batch".to_string(),
            })?;

    let mut trailing_stop = vec![f64::NAN; total];
    let mut state = vec![f64::NAN; total];
    let mut structure = vec![f64::NAN; total];
    market_structure_trailing_stop_batch_inner_into(
        open,
        high,
        low,
        close,
        sweep,
        reset_on,
        kernel,
        parallel,
        &mut trailing_stop,
        &mut state,
        &mut structure,
    )?;

    Ok(MarketStructureTrailingStopBatchOutput {
        trailing_stop,
        state,
        structure,
        combos,
        rows,
        cols,
    })
}

fn market_structure_trailing_stop_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MarketStructureTrailingStopBatchRange,
    reset_on: &str,
    kernel: Kernel,
    parallel: bool,
    out_trailing_stop: &mut [f64],
    out_state: &mut [f64],
    out_structure: &mut [f64],
) -> Result<Vec<MarketStructureTrailingStopParams>, MarketStructureTrailingStopError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => {
            return Err(MarketStructureTrailingStopError::InvalidKernelForBatch(
                other,
            ))
        }
    }

    let combos = expand_grid_checked(sweep, reset_on)?;
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(14))
        .max()
        .unwrap_or(0);
    let max_increment = combos
        .iter()
        .map(|params| params.increment_factor.unwrap_or(100.0))
        .fold(0.0_f64, f64::max);
    validate_common(open, high, low, close, max_length, max_increment, reset_on)?;

    let cols = close.len();
    let total = combos.len().checked_mul(cols).ok_or_else(|| {
        MarketStructureTrailingStopError::InvalidInput {
            msg: "market_structure_trailing_stop: rows*cols overflow in batch_into".to_string(),
        }
    })?;
    if out_trailing_stop.len() != total {
        return Err(MarketStructureTrailingStopError::MismatchedOutputLen {
            dst_len: out_trailing_stop.len(),
            expected_len: total,
        });
    }
    if out_state.len() != total {
        return Err(MarketStructureTrailingStopError::MismatchedOutputLen {
            dst_len: out_state.len(),
            expected_len: total,
        });
    }
    if out_structure.len() != total {
        return Err(MarketStructureTrailingStopError::MismatchedOutputLen {
            dst_len: out_structure.len(),
            expected_len: total,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize,
                  trailing_stop_row: &mut [f64],
                  state_row: &mut [f64],
                  structure_row: &mut [f64]| {
        trailing_stop_row.fill(f64::NAN);
        state_row.fill(f64::NAN);
        structure_row.fill(f64::NAN);
        let params = &combos[row];
        let reset_mode = parse_reset_on(params.reset_on.as_deref().unwrap_or(RESET_ON_CHOCH))
            .expect("combo reset_on already validated");
        compute_row(
            open,
            high,
            low,
            close,
            params.length.unwrap_or(14),
            params.increment_factor.unwrap_or(100.0),
            reset_mode,
            trailing_stop_row,
            state_row,
            structure_row,
        );
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel && combos.len() > 1 {
        out_trailing_stop
            .par_chunks_mut(cols)
            .zip(out_state.par_chunks_mut(cols))
            .zip(out_structure.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((trailing_stop_row, state_row), structure_row))| {
                worker(row, trailing_stop_row, state_row, structure_row);
            });
    } else {
        for (row, ((trailing_stop_row, state_row), structure_row)) in out_trailing_stop
            .chunks_mut(cols)
            .zip(out_state.chunks_mut(cols))
            .zip(out_structure.chunks_mut(cols))
            .enumerate()
        {
            worker(row, trailing_stop_row, state_row, structure_row);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = parallel;
        for (row, ((trailing_stop_row, state_row), structure_row)) in out_trailing_stop
            .chunks_mut(cols)
            .zip(out_state.chunks_mut(cols))
            .zip(out_structure.chunks_mut(cols))
            .enumerate()
        {
            worker(row, trailing_stop_row, state_row, structure_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "market_structure_trailing_stop",
    signature = (open, high, low, close, length=14, increment_factor=100.0, reset_on=RESET_ON_CHOCH, kernel=None)
)]
pub fn market_structure_trailing_stop_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    increment_factor: f64,
    reset_on: &str,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = MarketStructureTrailingStopInput::from_slices(
        open,
        high,
        low,
        close,
        MarketStructureTrailingStopParams {
            length: Some(length),
            increment_factor: Some(increment_factor),
            reset_on: Some(reset_on.to_string()),
        },
    );
    let out = py
        .allow_threads(|| market_structure_trailing_stop_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.trailing_stop.into_pyarray(py),
        out.state.into_pyarray(py),
        out.structure.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "market_structure_trailing_stop_batch",
    signature = (open, high, low, close, length_range=(14, 14, 0), increment_factor_range=(100.0, 100.0, 0.0), reset_on=RESET_ON_CHOCH, kernel=None)
)]
pub fn market_structure_trailing_stop_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    increment_factor_range: (f64, f64, f64),
    reset_on: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let output = py
        .allow_threads(|| {
            market_structure_trailing_stop_batch_with_kernel(
                open,
                high,
                low,
                close,
                &MarketStructureTrailingStopBatchRange {
                    length: length_range,
                    increment_factor: increment_factor_range,
                },
                reset_on,
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "trailing_stop",
        output
            .trailing_stop
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
        "structure",
        output
            .structure
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "increment_factors",
        output
            .combos
            .iter()
            .map(|params| params.increment_factor.unwrap_or(100.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "reset_ons",
        output
            .combos
            .iter()
            .map(|params| {
                params
                    .reset_on
                    .clone()
                    .unwrap_or_else(|| RESET_ON_CHOCH.to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_market_structure_trailing_stop_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(market_structure_trailing_stop_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        market_structure_trailing_stop_batch_py,
        m
    )?)?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStructureTrailingStopBatchConfig {
    pub length_range: Vec<usize>,
    pub increment_factor_range: Vec<f64>,
    pub reset_on: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = market_structure_trailing_stop_js)]
pub fn market_structure_trailing_stop_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    increment_factor: f64,
    reset_on: &str,
) -> Result<JsValue, JsValue> {
    let input = MarketStructureTrailingStopInput::from_slices(
        open,
        high,
        low,
        close,
        MarketStructureTrailingStopParams {
            length: Some(length),
            increment_factor: Some(increment_factor),
            reset_on: Some(reset_on.to_string()),
        },
    );
    let out = market_structure_trailing_stop_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("trailing_stop"),
        &serde_wasm_bindgen::to_value(&out.trailing_stop).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("state"),
        &serde_wasm_bindgen::to_value(&out.state).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("structure"),
        &serde_wasm_bindgen::to_value(&out.structure).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = market_structure_trailing_stop_batch_js)]
pub fn market_structure_trailing_stop_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: MarketStructureTrailingStopBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.increment_factor_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = market_structure_trailing_stop_batch_with_kernel(
        open,
        high,
        low,
        close,
        &MarketStructureTrailingStopBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
            increment_factor: (
                config.increment_factor_range[0],
                config.increment_factor_range[1],
                config.increment_factor_range[2],
            ),
        },
        config.reset_on.as_deref().unwrap_or(RESET_ON_CHOCH),
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("trailing_stop"),
        &serde_wasm_bindgen::to_value(&out.trailing_stop).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("state"),
        &serde_wasm_bindgen::to_value(&out.state).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("structure"),
        &serde_wasm_bindgen::to_value(&out.structure).unwrap(),
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
pub fn market_structure_trailing_stop_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(3 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_structure_trailing_stop_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 3 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_structure_trailing_stop_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    increment_factor: f64,
    reset_on: &str,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to market_structure_trailing_stop_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 3 * len);
        let (dst_trailing_stop, tail) = out.split_at_mut(len);
        let (dst_state, dst_structure) = tail.split_at_mut(len);
        let input = MarketStructureTrailingStopInput::from_slices(
            open,
            high,
            low,
            close,
            MarketStructureTrailingStopParams {
                length: Some(length),
                increment_factor: Some(increment_factor),
                reset_on: Some(reset_on.to_string()),
            },
        );
        market_structure_trailing_stop_into_slice(
            dst_trailing_stop,
            dst_state,
            dst_structure,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_structure_trailing_stop_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    increment_factor_start: f64,
    increment_factor_end: f64,
    increment_factor_step: f64,
    reset_on: &str,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to market_structure_trailing_stop_batch_into",
        ));
    }

    let sweep = MarketStructureTrailingStopBatchRange {
        length: (length_start, length_end, length_step),
        increment_factor: (
            increment_factor_start,
            increment_factor_end,
            increment_factor_step,
        ),
    };
    let combos =
        expand_grid_checked(&sweep, reset_on).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let split = rows.checked_mul(len).ok_or_else(|| {
        JsValue::from_str("rows*cols overflow in market_structure_trailing_stop_batch_into")
    })?;
    let total = split.checked_mul(3).ok_or_else(|| {
        JsValue::from_str("3*rows*cols overflow in market_structure_trailing_stop_batch_into")
    })?;

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (dst_trailing_stop, tail) = out.split_at_mut(split);
        let (dst_state, dst_structure) = tail.split_at_mut(split);
        market_structure_trailing_stop_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            reset_on,
            Kernel::Auto,
            false,
            dst_trailing_stop,
            dst_state,
            dst_structure,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_structure_trailing_stop_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    increment_factor: f64,
    reset_on: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = market_structure_trailing_stop_js(
        open,
        high,
        low,
        close,
        length,
        increment_factor,
        reset_on,
    )?;
    crate::write_wasm_object_f64_outputs(
        "market_structure_trailing_stop_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_structure_trailing_stop_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = market_structure_trailing_stop_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "market_structure_trailing_stop_batch_output_into_js",
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

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + (x * 0.26).sin() * 8.0 + (x * 0.08).cos() * 3.5 + x * 0.03
            })
            .collect();
        let open: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c + ((i as f64) * 0.41).cos() * 0.85)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.max(c) + 1.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.min(c) - 1.2)
            .collect();
        (open, high, low, close)
    }

    fn assert_vec_eq_nan(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (l, r) in lhs.iter().zip(rhs.iter()) {
            if l.is_nan() && r.is_nan() {
                continue;
            }
            assert_eq!(l, r);
        }
    }

    #[test]
    fn market_structure_trailing_stop_output_contract() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(320);
        let input = MarketStructureTrailingStopInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            MarketStructureTrailingStopParams {
                length: Some(5),
                increment_factor: Some(100.0),
                reset_on: Some(RESET_ON_ALL.to_string()),
            },
        );
        let out = market_structure_trailing_stop_with_kernel(&input, Kernel::Scalar)?;

        assert_eq!(out.trailing_stop.len(), close.len());
        assert_eq!(out.state.len(), close.len());
        assert_eq!(out.structure.len(), close.len());
        assert!(out.trailing_stop.iter().any(|v| v.is_finite()));
        assert!(out
            .state
            .iter()
            .all(|&v| v.is_nan() || v == -1.0 || v == 0.0 || v == 1.0));
        assert!(out
            .structure
            .iter()
            .all(|&v| v.is_nan() || v == -1.0 || v == 0.0 || v == 1.0));
        Ok(())
    }

    #[test]
    fn market_structure_trailing_stop_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(256);
        let input = MarketStructureTrailingStopInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            MarketStructureTrailingStopParams {
                length: Some(6),
                increment_factor: Some(80.0),
                reset_on: Some(RESET_ON_ALL.to_string()),
            },
        );
        let out = market_structure_trailing_stop_with_kernel(&input, Kernel::Scalar)?;
        let mut trailing_stop = vec![f64::NAN; close.len()];
        let mut state = vec![f64::NAN; close.len()];
        let mut structure = vec![f64::NAN; close.len()];
        market_structure_trailing_stop_into_slice(
            &mut trailing_stop,
            &mut state,
            &mut structure,
            &input,
            Kernel::Scalar,
        )?;
        assert_vec_eq_nan(&trailing_stop, &out.trailing_stop);
        assert_vec_eq_nan(&state, &out.state);
        assert_vec_eq_nan(&structure, &out.structure);
        Ok(())
    }

    #[test]
    fn market_structure_trailing_stop_reset_modes_differ() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(360);
        let all = market_structure_trailing_stop_with_kernel(
            &MarketStructureTrailingStopInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                MarketStructureTrailingStopParams {
                    length: Some(5),
                    increment_factor: Some(100.0),
                    reset_on: Some(RESET_ON_ALL.to_string()),
                },
            ),
            Kernel::Scalar,
        )?;
        let choch = market_structure_trailing_stop_with_kernel(
            &MarketStructureTrailingStopInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                MarketStructureTrailingStopParams {
                    length: Some(5),
                    increment_factor: Some(100.0),
                    reset_on: Some(RESET_ON_CHOCH.to_string()),
                },
            ),
            Kernel::Scalar,
        )?;

        assert_ne!(all.trailing_stop, choch.trailing_stop);
        let all_non_zero = all
            .structure
            .iter()
            .filter(|&&v| v != 0.0 && v.is_finite())
            .count();
        let choch_non_zero = choch
            .structure
            .iter()
            .filter(|&&v| v != 0.0 && v.is_finite())
            .count();
        assert!(all_non_zero >= choch_non_zero);
        Ok(())
    }

    #[test]
    fn market_structure_trailing_stop_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(240);
        let single = market_structure_trailing_stop_with_kernel(
            &MarketStructureTrailingStopInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                MarketStructureTrailingStopParams {
                    length: Some(5),
                    increment_factor: Some(120.0),
                    reset_on: Some(RESET_ON_ALL.to_string()),
                },
            ),
            Kernel::Scalar,
        )?;
        let batch = market_structure_trailing_stop_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &MarketStructureTrailingStopBatchRange {
                length: (5, 5, 0),
                increment_factor: (120.0, 120.0, 0.0),
            },
            RESET_ON_ALL,
            Kernel::Scalar,
        )?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_vec_eq_nan(&batch.trailing_stop, &single.trailing_stop);
        assert_vec_eq_nan(&batch.state, &single.state);
        assert_vec_eq_nan(&batch.structure, &single.structure);
        Ok(())
    }

    #[test]
    fn market_structure_trailing_stop_rejects_invalid_params() {
        let (open, high, low, close) = sample_ohlc(64);
        let err = market_structure_trailing_stop_with_kernel(
            &MarketStructureTrailingStopInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                MarketStructureTrailingStopParams {
                    length: Some(0),
                    increment_factor: Some(100.0),
                    reset_on: Some(RESET_ON_CHOCH.to_string()),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MarketStructureTrailingStopError::InvalidLength { .. }
        ));

        let err = market_structure_trailing_stop_with_kernel(
            &MarketStructureTrailingStopInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                MarketStructureTrailingStopParams {
                    length: Some(5),
                    increment_factor: Some(-1.0),
                    reset_on: Some(RESET_ON_CHOCH.to_string()),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MarketStructureTrailingStopError::InvalidIncrementFactor { .. }
        ));

        let err = market_structure_trailing_stop_with_kernel(
            &MarketStructureTrailingStopInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                MarketStructureTrailingStopParams {
                    length: Some(5),
                    increment_factor: Some(100.0),
                    reset_on: Some("bad".to_string()),
                },
            ),
            Kernel::Scalar,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MarketStructureTrailingStopError::InvalidResetOn { .. }
        ));
    }

    #[test]
    fn market_structure_trailing_stop_dispatch_compute_returns_expected_outputs(
    ) -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(220);
        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "increment_factor",
                value: ParamValue::Float(100.0),
            },
            ParamKV {
                key: "reset_on",
                value: ParamValue::EnumString(RESET_ON_ALL),
            },
        ];

        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "market_structure_trailing_stop",
            output_id: Some("trailing_stop"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &params,
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(out.output_id, "trailing_stop");
        match out.series {
            IndicatorSeries::F64(values) => assert_eq!(values.len(), close.len()),
            other => panic!("expected f64 series, got {:?}", other),
        }

        let state_out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "market_structure_trailing_stop",
            output_id: Some("state"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            params: &params,
            kernel: Kernel::Scalar,
        })?;
        assert_eq!(state_out.output_id, "state");
        Ok(())
    }
}
