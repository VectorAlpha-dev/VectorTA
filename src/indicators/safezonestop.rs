//! # SafeZoneStop
//!
//! The SafeZoneStop indicator attempts to place stop-loss levels based on
//! directional movement and volatility, using MINUS_DM or PLUS_DM logic under the hood.
//! Parity with alma.rs in terms of performance, features, and API structure. SIMD variants
//! are stubbed to the scalar implementation as per requirements.
//!
//! ## Parameters
//! - **period**: The time period for calculating DM (Wilder's smoothing). Defaults to 22.
//! - **mult**: Multiplier for the DM measure. Defaults to 2.5.
//! - **max_lookback**: Window for final max/min. Defaults to 3.
//! - **direction**: "long" or "short".
//!
//! ## Errors
//! - **AllValuesNaN**: safezonestop: All input data values are `NaN`.
//! - **InvalidPeriod**: safezonestop: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: safezonestop: Not enough valid data points for the requested `period`.
//! - **MismatchedLengths**: safezonestop: Input slices have different lengths.
//! - **InvalidDirection**: safezonestop: Direction must be "long" or "short".
//!
//! ## Returns
//! - **`Ok(SafeZoneStopOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(SafeZoneStopError)`** otherwise.
//!
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for SafeZoneStopInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SafeZoneStopData::Candles { candles, .. } => source_type(candles, "close"),
            SafeZoneStopData::Slices { low, .. } => low,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SafeZoneStopData<'a> {
    Candles {
        candles: &'a Candles,
        direction: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        direction: &'a str,
    },
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub max_lookback: Option<usize>,
}

impl Default for SafeZoneStopParams {
    fn default() -> Self {
        Self {
            period: Some(22),
            mult: Some(2.5),
            max_lookback: Some(3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopInput<'a> {
    pub data: SafeZoneStopData<'a>,
    pub params: SafeZoneStopParams,
}

impl<'a> SafeZoneStopInput<'a> {
    #[inline]
    pub fn from_candles(
        c: &'a Candles,
        direction: &'a str,
        p: SafeZoneStopParams,
    ) -> Self {
        Self {
            data: SafeZoneStopData::Candles { candles: c, direction },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        direction: &'a str,
        p: SafeZoneStopParams,
    ) -> Self {
        Self {
            data: SafeZoneStopData::Slices { high, low, direction },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "long", SafeZoneStopParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(22)
    }
    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(2.5)
    }
    #[inline]
    pub fn get_max_lookback(&self) -> usize {
        self.params.max_lookback.unwrap_or(3)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SafeZoneStopBuilder {
    period: Option<usize>,
    mult: Option<f64>,
    max_lookback: Option<usize>,
    direction: Option<&'static str>,
    kernel: Kernel,
}

impl Default for SafeZoneStopBuilder {
    fn default() -> Self {
        Self {
            period: None,
            mult: None,
            max_lookback: None,
            direction: Some("long"),
            kernel: Kernel::Auto,
        }
    }
}

impl SafeZoneStopBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self { self.period = Some(n); self }
    #[inline(always)]
    pub fn mult(mut self, x: f64) -> Self { self.mult = Some(x); self }
    #[inline(always)]
    pub fn max_lookback(mut self, n: usize) -> Self { self.max_lookback = Some(n); self }
    #[inline(always)]
    pub fn direction(mut self, d: &'static str) -> Self { self.direction = Some(d); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<SafeZoneStopOutput, SafeZoneStopError> {
        let p = SafeZoneStopParams {
            period: self.period,
            mult: self.mult,
            max_lookback: self.max_lookback,
        };
        let i = SafeZoneStopInput::from_candles(
            c,
            self.direction.unwrap_or("long"),
            p,
        );
        safezonestop_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<SafeZoneStopOutput, SafeZoneStopError> {
        let p = SafeZoneStopParams {
            period: self.period,
            mult: self.mult,
            max_lookback: self.max_lookback,
        };
        let i = SafeZoneStopInput::from_slices(
            high,
            low,
            self.direction.unwrap_or("long"),
            p,
        );
        safezonestop_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum SafeZoneStopError {
    #[error("safezonestop: All values are NaN.")]
    AllValuesNaN,
    #[error("safezonestop: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("safezonestop: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("safezonestop: Mismatched lengths")]
    MismatchedLengths,
    #[error("safezonestop: Invalid direction. Must be 'long' or 'short'.")]
    InvalidDirection,
}

#[inline]
pub fn safezonestop(input: &SafeZoneStopInput) -> Result<SafeZoneStopOutput, SafeZoneStopError> {
    safezonestop_with_kernel(input, Kernel::Auto)
}

pub fn safezonestop_with_kernel(
    input: &SafeZoneStopInput,
    kernel: Kernel,
) -> Result<SafeZoneStopOutput, SafeZoneStopError> {
    let (high, low, direction) = match &input.data {
        SafeZoneStopData::Candles { candles, direction } => {
            let h = source_type(candles, "high");
            let l = source_type(candles, "low");
            (h, l, *direction)
        }
        SafeZoneStopData::Slices { high, low, direction } => (*high, *low, *direction),
    };

    if high.len() != low.len() {
        return Err(SafeZoneStopError::MismatchedLengths);
    }

    let period = input.get_period();
    let mult = input.get_mult();
    let max_lookback = input.get_max_lookback();
    let len = high.len();

    if period == 0 || period > len {
        return Err(SafeZoneStopError::InvalidPeriod { period, data_len: len });
    }

    let has_any_non_nan = high.iter().any(|&v| !v.is_nan()) || low.iter().any(|&v| !v.is_nan());
    if !has_any_non_nan {
        return Err(SafeZoneStopError::AllValuesNaN);
    }

    if direction != "long" && direction != "short" {
        return Err(SafeZoneStopError::InvalidDirection);
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                safezonestop_scalar(high, low, period, mult, max_lookback, direction, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                safezonestop_avx2(high, low, period, mult, max_lookback, direction, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                safezonestop_avx512(high, low, period, mult, max_lookback, direction, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(SafeZoneStopOutput { values: out })
}

#[inline(always)]
pub unsafe fn safezonestop_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    let len = high.len();

    let mut last_low = vec![f64::NAN; len];
    let mut last_high = vec![f64::NAN; len];
    for i in 1..len {
        last_low[i] = low[i - 1];
        last_high[i] = high[i - 1];
    }

    let mut minus_dm = vec![0.0; len];
    for i in 1..len {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        if down_move > up_move && down_move > 0.0 {
            minus_dm[i] = down_move;
        }
    }
    let mut minus_dm_smooth = vec![f64::NAN; len];
    if period < len {
        let mut sum = 0.0;
        for i in 1..=period {
            sum += minus_dm[i];
        }
        minus_dm_smooth[period] = sum;
        for i in (period + 1)..len {
            minus_dm_smooth[i] = minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / (period as f64)) + minus_dm[i];
        }
    }

    let mut plus_dm = vec![0.0; len];
    for i in 1..len {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        if up_move > down_move && up_move > 0.0 {
            plus_dm[i] = up_move;
        }
    }
    let mut plus_dm_smooth = vec![f64::NAN; len];
    if period < len {
        let mut sum = 0.0;
        for i in 1..=period {
            sum += plus_dm[i];
        }
        plus_dm_smooth[period] = sum;
        for i in (period + 1)..len {
            plus_dm_smooth[i] = plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / (period as f64)) + plus_dm[i];
        }
    }

    let mut intermediate = vec![f64::NAN; len];
    if direction == "long" {
        for i in 0..len {
            if !minus_dm_smooth[i].is_nan() && !last_low[i].is_nan() {
                intermediate[i] = last_low[i] - mult * minus_dm_smooth[i];
            }
        }
    } else {
        for i in 0..len {
            if !plus_dm_smooth[i].is_nan() && !last_high[i].is_nan() {
                intermediate[i] = last_high[i] + mult * plus_dm_smooth[i];
            }
        }
    }

    for i in 0..len {
        if i + 1 < max_lookback {
            continue;
        }
        let start_idx = i + 1 - max_lookback;
        if direction == "long" {
            let mut mx = f64::NAN;
            for j in start_idx..=i {
                let val = intermediate[j];
                if val.is_nan() {
                    continue;
                }
                if mx.is_nan() || val > mx {
                    mx = val;
                }
            }
            out[i] = mx;
        } else {
            let mut mn = f64::NAN;
            for j in start_idx..=i {
                let val = intermediate[j];
                if val.is_nan() {
                    continue;
                }
                if mn.is_nan() || val < mn {
                    mn = val;
                }
            }
            out[i] = mn;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    if period <= 32 {
        safezonestop_avx512_short(high, low, period, mult, max_lookback, direction, out);
    } else {
        safezonestop_avx512_long(high, low, period, mult, max_lookback, direction, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopStream {
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: String,
    buffer_high: Vec<f64>,
    buffer_low: Vec<f64>,
    idx: usize,
    filled: bool,
    last_result: f64,
}

impl SafeZoneStopStream {
    pub fn try_new(params: SafeZoneStopParams, direction: &str) -> Result<Self, SafeZoneStopError> {
        let period = params.period.unwrap_or(22);
        let mult = params.mult.unwrap_or(2.5);
        let max_lookback = params.max_lookback.unwrap_or(3);
        if period == 0 {
            return Err(SafeZoneStopError::InvalidPeriod { period, data_len: 0 });
        }
        if direction != "long" && direction != "short" {
            return Err(SafeZoneStopError::InvalidDirection);
        }
        Ok(Self {
            period,
            mult,
            max_lookback,
            direction: direction.to_string(),
            buffer_high: vec![f64::NAN; period.max(max_lookback)],
            buffer_low: vec![f64::NAN; period.max(max_lookback)],
            idx: 0,
            filled: false,
            last_result: f64::NAN,
        })
    }
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        let n = self.buffer_high.len();
        self.buffer_high[self.idx] = high;
        self.buffer_low[self.idx] = low;
        self.idx = (self.idx + 1) % n;
        if !self.filled && self.idx == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        let slice_high = self.buffer_high.iter().cycle().skip(self.idx).take(n).cloned().collect::<Vec<_>>();
        let slice_low = self.buffer_low.iter().cycle().skip(self.idx).take(n).cloned().collect::<Vec<_>>();
        let mut out = vec![f64::NAN; n];
        unsafe {
            safezonestop_scalar(
                &slice_high,
                &slice_low,
                self.period,
                self.mult,
                self.max_lookback,
                &self.direction,
                &mut out,
            );
        }
        self.last_result = *out.last().unwrap_or(&f64::NAN);
        Some(self.last_result)
    }
}

#[derive(Clone, Debug)]
pub struct SafeZoneStopBatchRange {
    pub period: (usize, usize, usize),
    pub mult: (f64, f64, f64),
    pub max_lookback: (usize, usize, usize),
}

impl Default for SafeZoneStopBatchRange {
    fn default() -> Self {
        Self {
            period: (22, 22, 0),
            mult: (2.5, 2.5, 0.0),
            max_lookback: (3, 3, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SafeZoneStopBatchBuilder {
    range: SafeZoneStopBatchRange,
    direction: &'static str,
    kernel: Kernel,
}

impl SafeZoneStopBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step); self
    }
    #[inline]
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step); self
    }
    #[inline]
    pub fn max_lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.max_lookback = (start, end, step); self
    }
    #[inline]
    pub fn direction(mut self, d: &'static str) -> Self { self.direction = d; self }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
        safezonestop_batch_with_kernel(high, low, &self.range, self.direction, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        direction: &'static str,
        k: Kernel,
    ) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
        SafeZoneStopBatchBuilder::new().kernel(k).direction(direction).apply_slices(high, low)
    }
}

pub fn safezonestop_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &SafeZoneStopBatchRange,
    direction: &str,
    k: Kernel,
) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(SafeZoneStopError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    safezonestop_batch_par_slice(high, low, sweep, direction, simd)
}

#[derive(Clone, Debug)]
pub struct SafeZoneStopBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SafeZoneStopParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SafeZoneStopBatchOutput {
    pub fn row_for_params(&self, p: &SafeZoneStopParams) -> Option<usize> {
        self.combos.iter().position(|c|
            c.period.unwrap_or(22) == p.period.unwrap_or(22)
                && (c.mult.unwrap_or(2.5) - p.mult.unwrap_or(2.5)).abs() < 1e-12
                && c.max_lookback.unwrap_or(3) == p.max_lookback.unwrap_or(3)
        )
    }
    pub fn values_for(&self, p: &SafeZoneStopParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SafeZoneStopBatchRange) -> Vec<SafeZoneStopParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        let mut v = Vec::new(); let mut x = start;
        while x <= end + 1e-12 { v.push(x); x += step; }
        v
    }
    let periods = axis_usize(r.period);
    let mults = axis_f64(r.mult);
    let lookbacks = axis_usize(r.max_lookback);
    let mut out = Vec::with_capacity(periods.len() * mults.len() * lookbacks.len());
    for &p in &periods {
        for &m in &mults {
            for &l in &lookbacks {
                out.push(SafeZoneStopParams {
                    period: Some(p), mult: Some(m), max_lookback: Some(l)
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn safezonestop_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &SafeZoneStopBatchRange,
    direction: &str,
    kern: Kernel,
) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
    safezonestop_batch_inner(high, low, sweep, direction, kern, false)
}

#[inline(always)]
pub fn safezonestop_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &SafeZoneStopBatchRange,
    direction: &str,
    kern: Kernel,
) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
    safezonestop_batch_inner(high, low, sweep, direction, kern, true)
}

#[inline(always)]
fn safezonestop_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &SafeZoneStopBatchRange,
    direction: &str,
    kern: Kernel,
    parallel: bool,
) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SafeZoneStopError::InvalidPeriod { period: 0, data_len: 0 });
    }
    if high.len() != low.len() {
        return Err(SafeZoneStopError::MismatchedLengths);
    }
    let len = high.len();
    let first = high.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(SafeZoneStopError::NotEnoughValidData { needed: max_p, valid: len - first });
    }
    let rows = combos.len();
    let cols = len;
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let mult = combos[row].mult.unwrap();
        let max_lookback = combos[row].max_lookback.unwrap();
        match kern {
            Kernel::Scalar => safezonestop_row_scalar(high, low, period, mult, max_lookback, direction, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => safezonestop_row_avx2(high, low, period, mult, max_lookback, direction, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => safezonestop_row_avx512(high, low, period, mult, max_lookback, direction, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(SafeZoneStopBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
pub unsafe fn safezonestop_row_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_row_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_row_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    if period <= 32 {
        safezonestop_row_avx512_short(high, low, period, mult, max_lookback, direction, out)
    } else {
        safezonestop_row_avx512_long(high, low, period, mult, max_lookback, direction, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_row_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_row_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    mult: f64,
    max_lookback: usize,
    direction: &str,
    out: &mut [f64],
) {
    safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_safezonestop_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SafeZoneStopParams {
            period: Some(14), mult: None, max_lookback: None,
        };
        let input = SafeZoneStopInput::from_candles(&candles, "short", params);
        let output = safezonestop_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_safezonestop_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SafeZoneStopParams {
            period: Some(22), mult: Some(2.5), max_lookback: Some(3),
        };
        let input = SafeZoneStopInput::from_candles(&candles, "long", params);
        let output = safezonestop_with_kernel(&input, kernel)?;
        let expected = [
            45331.180007991,
            45712.94455308232,
            46019.94707339676,
            46461.767660969635,
            46461.767660969635,
        ];
        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            let diff = (val - expected[i]).abs();
            assert!(
                diff < 1e-4,
                "[{}] SafeZoneStop {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected[i]
            );
        }
        Ok(())
    }

    fn check_safezonestop_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SafeZoneStopInput::with_default_candles(&candles);
        let output = safezonestop_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_safezonestop_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = SafeZoneStopParams { period: Some(0), mult: Some(2.5), max_lookback: Some(3) };
        let input = SafeZoneStopInput::from_slices(&high, &low, "long", params);
        let res = safezonestop_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SafeZoneStop should fail with zero period", test_name);
        Ok(())
    }

    fn check_safezonestop_mismatched_lengths(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0];
        let params = SafeZoneStopParams::default();
        let input = SafeZoneStopInput::from_slices(&high, &low, "long", params);
        let res = safezonestop_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SafeZoneStop should fail with mismatched lengths", test_name);
        Ok(())
    }

    fn check_safezonestop_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SafeZoneStopInput::with_default_candles(&candles);
        let res = safezonestop_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    macro_rules! generate_all_safezonestop_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }
    generate_all_safezonestop_tests!(
        check_safezonestop_partial_params,
        check_safezonestop_accuracy,
        check_safezonestop_default_candles,
        check_safezonestop_zero_period,
        check_safezonestop_mismatched_lengths,
        check_safezonestop_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let high = source_type(&c, "high");
        let low = source_type(&c, "low");

        let output = SafeZoneStopBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(high, low)?;

        let def = SafeZoneStopParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            45331.180007991,
            45712.94455308232,
            46019.94707339676,
            46461.767660969635,
            46461.767660969635,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-4,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
