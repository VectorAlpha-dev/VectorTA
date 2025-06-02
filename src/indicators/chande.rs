//! # Chande Exits (Chandelier Exits)
//!
//! Volatility-based trailing exit using ATR and rolling max/min, with builder, batch, and AVX/parallel support.
//! API/feature/test coverage parity with alma.rs.
//!
//! ## Parameters
//! - **period**: Window size for both ATR and rolling max/min (default: 22).
//! - **mult**: ATR multiplier (default: 3.0).
//! - **direction**: "long" or "short" (default: "long").
//!
//! ## Errors
//! - **AllValuesNaN**: chande: All input values are NaN.
//! - **InvalidPeriod**: chande: period is zero or exceeds length.
//! - **NotEnoughValidData**: chande: Not enough valid data for period.
//! - **InvalidDirection**: chande: direction must be "long" or "short".
//!
//! ## Returns
//! - `Ok(ChandeOutput)` on success, `Err(ChandeError)` on error.
//!

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ChandeData<'a> {
    Candles { candles: &'a Candles },
}

#[derive(Debug, Clone)]
pub struct ChandeOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ChandeParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub direction: Option<String>,
}

impl Default for ChandeParams {
    fn default() -> Self {
        Self {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChandeInput<'a> {
    pub data: ChandeData<'a>,
    pub params: ChandeParams,
}

impl<'a> ChandeInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: ChandeParams) -> Self {
        Self { data: ChandeData::Candles { candles: c }, params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, ChandeParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(22)
    }
    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(3.0)
    }
    #[inline]
    pub fn get_direction(&self) -> &str {
        self.params.direction.as_deref().unwrap_or("long")
    }
}

#[derive(Clone, Debug)]
pub struct ChandeBuilder {
    period: Option<usize>,
    mult: Option<f64>,
    direction: Option<String>,
    kernel: Kernel,
}

impl Default for ChandeBuilder {
    fn default() -> Self {
        Self { period: None, mult: None, direction: None, kernel: Kernel::Auto }
    }
}
impl ChandeBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self { self.period = Some(n); self }
    #[inline(always)]
    pub fn mult(mut self, m: f64) -> Self { self.mult = Some(m); self }
    #[inline(always)]
    pub fn direction<S: Into<String>>(mut self, d: S) -> Self { self.direction = Some(d.into()); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<ChandeOutput, ChandeError> {
        let p = ChandeParams { period: self.period, mult: self.mult, direction: self.direction };
        let i = ChandeInput::from_candles(c, p);
        chande_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<ChandeStream, ChandeError> {
        let p = ChandeParams { period: self.period, mult: self.mult, direction: self.direction };
        ChandeStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ChandeError {
    #[error("chande: All values are NaN.")]
    AllValuesNaN,
    #[error("chande: Invalid period: period = {period}, data_len = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("chande: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("chande: Invalid direction: {direction}")]
    InvalidDirection { direction: String },
}

#[inline]
pub fn chande(input: &ChandeInput) -> Result<ChandeOutput, ChandeError> {
    chande_with_kernel(input, Kernel::Auto)
}

pub fn chande_with_kernel(input: &ChandeInput, kernel: Kernel) -> Result<ChandeOutput, ChandeError> {
    let ChandeData::Candles { candles } = &input.data;
    let high = source_type(candles, "high");
    let low = source_type(candles, "low");
    let close = source_type(candles, "close");
    let len = high.len();

    let first = close.iter().position(|&x| !x.is_nan()).ok_or(ChandeError::AllValuesNaN)?;
    let period = input.get_period();
    let mult = input.get_mult();
    let dir = input.get_direction().to_lowercase();
    if dir != "long" && dir != "short" {
        return Err(ChandeError::InvalidDirection { direction: dir });
    }
    if period == 0 || period > len {
        return Err(ChandeError::InvalidPeriod { period, data_len: len });
    }
    if len - first < period {
        return Err(ChandeError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => chande_scalar(high, low, close, period, mult, &dir, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => chande_avx2(high, low, close, period, mult, &dir, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => chande_avx512(high, low, close, period, mult, &dir, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(ChandeOutput { values: out })
}

#[inline]
pub fn chande_scalar(
    high: &[f64], low: &[f64], close: &[f64], period: usize, mult: f64, dir: &str, first: usize, out: &mut [f64]
) {
    let len = high.len();
    let alpha = 1.0 / period as f64;
    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;
    let mut atr = vec![f64::NAN; len];
    for i in first..len {
        let tr = if i == first {
            high[i] - low[i]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        if i < first + period {
            sum_tr += tr;
            if i == first + period - 1 {
                rma = sum_tr / period as f64;
                atr[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr[i] = rma;
        }
        if i >= first + period - 1 {
            let a = atr[i];
            if !a.is_nan() {
                let start = i + 1 - period;
                if dir == "long" {
                    let mut m = f64::MIN;
                    for j in start..=i {
                        if high[j] > m { m = high[j]; }
                    }
                    out[i] = m - a * mult;
                } else {
                    let mut m = f64::MAX;
                    for j in start..=i {
                        if low[j] < m { m = low[j]; }
                    }
                    out[i] = m + a * mult;
                }
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn chande_avx2(
    high: &[f64], low: &[f64], close: &[f64], period: usize, mult: f64, dir: &str, first: usize, out: &mut [f64]
) {
    chande_scalar(high, low, close, period, mult, dir, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn chande_avx512(
    high: &[f64], low: &[f64], close: &[f64], period: usize, mult: f64, dir: &str, first: usize, out: &mut [f64]
) {
    if period <= 32 {
        unsafe { chande_avx512_short(high, low, close, period, mult, dir, first, out) }
    } else {
        unsafe { chande_avx512_long(high, low, close, period, mult, dir, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chande_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], period: usize, mult: f64, dir: &str, first: usize, out: &mut [f64]
) {
    chande_scalar(high, low, close, period, mult, dir, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chande_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], period: usize, mult: f64, dir: &str, first: usize, out: &mut [f64]
) {
    chande_scalar(high, low, close, period, mult, dir, first, out)
}

#[derive(Debug, Clone)]
pub struct ChandeStream {
    period: usize,
    mult: f64,
    direction: String,
    high_buf: Vec<f64>,
    low_buf: Vec<f64>,
    close_prev: f64,
    atr: f64,
    buffer_filled: usize,
    filled: bool,
}

impl ChandeStream {
    pub fn try_new(params: ChandeParams) -> Result<Self, ChandeError> {
        let period = params.period.unwrap_or(22);
        let mult = params.mult.unwrap_or(3.0);
        let direction = params.direction.unwrap_or_else(|| "long".into());
        if period == 0 {
            return Err(ChandeError::InvalidPeriod { period, data_len: 0 });
        }
        if direction != "long" && direction != "short" {
            return Err(ChandeError::InvalidDirection { direction });
        }
        Ok(Self {
            period,
            mult,
            direction,
            high_buf: Vec::with_capacity(period),
            low_buf: Vec::with_capacity(period),
            close_prev: f64::NAN,
            atr: 0.0,
            buffer_filled: 0,
            filled: false,
        })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        if self.buffer_filled < self.period {
            self.high_buf.push(high);
            self.low_buf.push(low);
            if self.buffer_filled == 0 {
                self.atr += high - low;
            } else {
                let hl = high - low;
                let hc = (high - self.close_prev).abs();
                let lc = (low - self.close_prev).abs();
                self.atr += hl.max(hc).max(lc);
            }
            self.buffer_filled += 1;
            self.close_prev = close;
            if self.buffer_filled == self.period {
                self.atr /= self.period as f64;
                self.filled = true;
            }
            return None;
        }
        let hl = high - low;
        let hc = (high - self.close_prev).abs();
        let lc = (low - self.close_prev).abs();
        let tr = hl.max(hc).max(lc);
        let alpha = 1.0 / self.period as f64;
        self.atr += alpha * (tr - self.atr);
        self.high_buf.push(high);
        self.low_buf.push(low);
        if self.high_buf.len() > self.period { self.high_buf.remove(0); }
        if self.low_buf.len() > self.period { self.low_buf.remove(0); }
        self.close_prev = close;
        if self.filled {
            if self.direction == "long" {
                let m = self.high_buf.iter().cloned().fold(f64::MIN, f64::max);
                Some(m - self.atr * self.mult)
            } else {
                let m = self.low_buf.iter().cloned().fold(f64::MAX, f64::min);
                Some(m + self.atr * self.mult)
            }
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChandeBatchRange {
    pub period: (usize, usize, usize),
    pub mult: (f64, f64, f64),
}

impl Default for ChandeBatchRange {
    fn default() -> Self {
        Self {
            period: (22, 22, 0),
            mult: (3.0, 3.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChandeBatchBuilder {
    range: ChandeBatchRange,
    direction: String,
    kernel: Kernel,
}

impl ChandeBatchBuilder {
    pub fn new() -> Self {
        Self { range: ChandeBatchRange::default(), direction: "long".into(), kernel: Kernel::Auto }
    }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn direction<S: Into<String>>(mut self, d: S) -> Self { self.direction = d.into(); self }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step); self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0); self
    }
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step); self
    }
    pub fn mult_static(mut self, m: f64) -> Self {
        self.range.mult = (m, m, 0.0); self
    }

    pub fn apply_candles(self, c: &Candles) -> Result<ChandeBatchOutput, ChandeError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        chande_batch_with_kernel(high, low, close, &self.range, &self.direction, self.kernel)
    }
}

pub fn chande_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ChandeBatchRange,
    direction: &str,
    k: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(ChandeError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    chande_batch_par_slice(high, low, close, sweep, direction, simd)
}

#[derive(Clone, Debug)]
pub struct ChandeBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ChandeParams>,
    pub rows: usize,
    pub cols: usize,
}
impl ChandeBatchOutput {
    pub fn row_for_params(&self, p: &ChandeParams) -> Option<usize> {
        self.combos.iter().position(|c|
            c.period.unwrap_or(22) == p.period.unwrap_or(22) &&
            (c.mult.unwrap_or(3.0) - p.mult.unwrap_or(3.0)).abs() < 1e-12 &&
            c.direction.as_deref().unwrap_or("long") == p.direction.as_deref().unwrap_or("long")
        )
    }
    pub fn values_for(&self, p: &ChandeParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ChandeBatchRange, dir: &str) -> Vec<ChandeParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let periods = axis_usize(r.period);
    let mults = axis_f64(r.mult);
    let mut out = Vec::with_capacity(periods.len() * mults.len());
    for &p in &periods {
        for &m in &mults {
            out.push(ChandeParams {
                period: Some(p),
                mult: Some(m),
                direction: Some(dir.to_string()),
            });
        }
    }
    out
}

#[inline(always)]
pub fn chande_batch_slice(
    high: &[f64], low: &[f64], close: &[f64], sweep: &ChandeBatchRange, dir: &str, kern: Kernel
) -> Result<ChandeBatchOutput, ChandeError> {
    chande_batch_inner(high, low, close, sweep, dir, kern, false)
}

#[inline(always)]
pub fn chande_batch_par_slice(
    high: &[f64], low: &[f64], close: &[f64], sweep: &ChandeBatchRange, dir: &str, kern: Kernel
) -> Result<ChandeBatchOutput, ChandeError> {
    chande_batch_inner(high, low, close, sweep, dir, kern, true)
}

#[inline(always)]
fn chande_batch_inner(
    high: &[f64], low: &[f64], close: &[f64], sweep: &ChandeBatchRange, dir: &str, kern: Kernel, parallel: bool
) -> Result<ChandeBatchOutput, ChandeError> {
    let combos = expand_grid(sweep, dir);
    if combos.is_empty() {
        return Err(ChandeError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = close.iter().position(|&x| !x.is_nan()).ok_or(ChandeError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(ChandeError::NotEnoughValidData { needed: max_p, valid: high.len() - first });
    }
    let rows = combos.len();
    let cols = high.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let mult = combos[row].mult.unwrap();
        let direction = combos[row].direction.as_deref().unwrap();
        match kern {
            Kernel::Scalar => chande_row_scalar(high, low, close, first, period, mult, direction, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => chande_row_avx2(high, low, close, first, period, mult, direction, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => chande_row_avx512(high, low, close, first, period, mult, direction, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        values.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() { do_row(row, slice); }
    }
    Ok(ChandeBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
unsafe fn chande_row_scalar(
    high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, mult: f64, dir: &str, out: &mut [f64]
) {
    chande_scalar(high, low, close, period, mult, dir, first, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx2(
    high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, mult: f64, dir: &str, out: &mut [f64]
) {
    chande_row_scalar(high, low, close, first, period, mult, dir, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512(
    high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, mult: f64, dir: &str, out: &mut [f64]
) {
    if period <= 32 {
        chande_row_avx512_short(high, low, close, first, period, mult, dir, out)
    } else {
        chande_row_avx512_long(high, low, close, first, period, mult, dir, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, mult: f64, dir: &str, out: &mut [f64]
) {
    chande_row_scalar(high, low, close, first, period, mult, dir, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, mult: f64, dir: &str, out: &mut [f64]
) {
    chande_row_scalar(high, low, close, first, period, mult, dir, out)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_chande_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = ChandeParams {
            period: None,
            mult: None,
            direction: None,
        };
        let input = ChandeInput::from_candles(&candles, default_params);
        let output = chande_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_chande_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = &candles.close;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);
        let chande_result = chande_with_kernel(&input, kernel)?;

        assert_eq!(chande_result.values.len(), close_prices.len());

        let expected_last_five = [
            59444.14115983658,
            58576.49837984401,
            58649.1120898511,
            58724.56154031242,
            58713.39965211639,
        ];

        assert!(chande_result.values.len() >= 5);
        let start_idx = chande_result.values.len() - 5;
        let actual_last_five = &chande_result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "[{}] Chande Exits mismatch at index {}: expected {}, got {}",
                test_name, i, exp, val
            );
        }

        let period = 22;
        for i in 0..(period - 1) {
            assert!(
                chande_result.values[i].is_nan(),
                "Expected leading NaN at index {}",
                i
            );
        }

        let default_input = ChandeInput::with_default_candles(&candles);
        let default_output = chande_with_kernel(&default_input, kernel)?;
        assert_eq!(default_output.values.len(), close_prices.len());
        Ok(())
    }

    fn check_chande_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(0),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let res = chande_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Chande should fail with zero period", test_name);
        Ok(())
    }

    fn check_chande_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(99999),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let res = chande_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Chande should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_chande_bad_direction(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("bad".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);

        let res = chande_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Chande should fail with bad direction", test_name);
        Ok(())
    }

    fn check_chande_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params);
        let result = chande_with_kernel(&input, kernel)?;

        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(
                    !result.values[i].is_nan(),
                    "[{}] Unexpected NaN at index {}", test_name, i
                );
            }
        }
        Ok(())
    }

    fn check_chande_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ChandeParams {
            period: Some(22),
            mult: Some(3.0),
            direction: Some("long".into()),
        };
        let input = ChandeInput::from_candles(&candles, params.clone());
        let batch_output = chande_with_kernel(&input, kernel)?.values;

        let mut stream = ChandeStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for ((&h, &l), &c) in candles.high.iter().zip(&candles.low).zip(&candles.close) {
            match stream.update(h, l, c) {
                Some(chande_val) => stream_values.push(chande_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Chande streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_chande_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

    generate_all_chande_tests!(
        check_chande_partial_params,
        check_chande_accuracy,
        check_chande_zero_period,
        check_chande_period_exceeds_length,
        check_chande_bad_direction,
        check_chande_nan_handling,
        check_chande_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = ChandeBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        let def = ChandeParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            59444.14115983658,
            58576.49837984401,
            58649.1120898511,
            58724.56154031242,
            58713.39965211639,
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
