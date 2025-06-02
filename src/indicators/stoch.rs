//! # Stochastic Oscillator (Stoch)
//!
//! A momentum indicator comparing a particular closing price to a range of prices over a certain period.
//! Supports scalar, avx2, and avx512 stubs, batch/grid sweeping, and streaming API.
//!
//! ## Parameters
//! - **fastk_period**: Highest high/lowest low window size. Default: 14
//! - **slowk_period**: MA period for %K smoothing. Default: 3
//! - **slowk_ma_type**: MA type for %K smoothing. Default: "sma"
//! - **slowd_period**: MA period for %D. Default: 3
//! - **slowd_ma_type**: MA type for %D. Default: "sma"
//!
//! ## Errors
//! - **EmptyData**: stoch: Input slices are empty.
//! - **MismatchedLength**: stoch: Input slices have different lengths.
//! - **InvalidPeriod**: stoch: Period is zero or exceeds data length.
//! - **NotEnoughValidData**: stoch: Not enough valid (non-NaN) data points.
//! - **AllValuesNaN**: stoch: All input values are NaN.
//!
//! ## Returns
//! - **Ok(StochOutput)** on success. `k` and `d` fields have same length as input.
//! - **Err(StochError)** otherwise.
//!

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// === Input/Output Structs ===

#[derive(Debug, Clone)]
pub enum StochData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64], close: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct StochOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StochParams {
    pub fastk_period: Option<usize>,
    pub slowk_period: Option<usize>,
    pub slowk_ma_type: Option<String>,
    pub slowd_period: Option<usize>,
    pub slowd_ma_type: Option<String>,
}

impl Default for StochParams {
    fn default() -> Self {
        Self {
            fastk_period: Some(14),
            slowk_period: Some(3),
            slowk_ma_type: Some("sma".to_string()),
            slowd_period: Some(3),
            slowd_ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochInput<'a> {
    pub data: StochData<'a>,
    pub params: StochParams,
}

impl<'a> StochInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: StochParams) -> Self {
        Self { data: StochData::Candles { candles: c }, params: p }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], p: StochParams) -> Self {
        Self { data: StochData::Slices { high, low, close }, params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, StochParams::default())
    }
    #[inline]
    pub fn get_fastk_period(&self) -> usize {
        self.params.fastk_period.unwrap_or(14)
    }
    #[inline]
    pub fn get_slowk_period(&self) -> usize {
        self.params.slowk_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_slowk_ma_type(&self) -> String {
        self.params.slowk_ma_type.clone().unwrap_or_else(|| "sma".to_string())
    }
    #[inline]
    pub fn get_slowd_period(&self) -> usize {
        self.params.slowd_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_slowd_ma_type(&self) -> String {
        self.params.slowd_ma_type.clone().unwrap_or_else(|| "sma".to_string())
    }
}

// === Builder ===

#[derive(Copy, Clone, Debug)]
pub struct StochBuilder {
    fastk_period: Option<usize>,
    slowk_period: Option<usize>,
    slowk_ma_type: Option<&'static str>,
    slowd_period: Option<usize>,
    slowd_ma_type: Option<&'static str>,
    kernel: Kernel,
}

impl Default for StochBuilder {
    fn default() -> Self {
        Self {
            fastk_period: None,
            slowk_period: None,
            slowk_ma_type: None,
            slowd_period: None,
            slowd_ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StochBuilder {
    #[inline(always)] pub fn new() -> Self { Self::default() }
    #[inline(always)] pub fn fastk_period(mut self, n: usize) -> Self { self.fastk_period = Some(n); self }
    #[inline(always)] pub fn slowk_period(mut self, n: usize) -> Self { self.slowk_period = Some(n); self }
    #[inline(always)] pub fn slowk_ma_type(mut self, t: &'static str) -> Self { self.slowk_ma_type = Some(t); self }
    #[inline(always)] pub fn slowd_period(mut self, n: usize) -> Self { self.slowd_period = Some(n); self }
    #[inline(always)] pub fn slowd_ma_type(mut self, t: &'static str) -> Self { self.slowd_ma_type = Some(t); self }
    #[inline(always)] pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<StochOutput, StochError> {
        let p = StochParams {
            fastk_period: self.fastk_period,
            slowk_period: self.slowk_period,
            slowk_ma_type: self.slowk_ma_type.map(|s| s.to_string()),
            slowd_period: self.slowd_period,
            slowd_ma_type: self.slowd_ma_type.map(|s| s.to_string()),
        };
        let i = StochInput::from_candles(c, p);
        stoch_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<StochOutput, StochError> {
        let p = StochParams {
            fastk_period: self.fastk_period,
            slowk_period: self.slowk_period,
            slowk_ma_type: self.slowk_ma_type.map(|s| s.to_string()),
            slowd_period: self.slowd_period,
            slowd_ma_type: self.slowd_ma_type.map(|s| s.to_string()),
        };
        let i = StochInput::from_slices(high, low, close, p);
        stoch_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<StochStream, StochError> {
        let p = StochParams {
            fastk_period: self.fastk_period,
            slowk_period: self.slowk_period,
            slowk_ma_type: self.slowk_ma_type.map(|s| s.to_string()),
            slowd_period: self.slowd_period,
            slowd_ma_type: self.slowd_ma_type.map(|s| s.to_string()),
        };
        StochStream::try_new(p)
    }
}

// === Errors ===

#[derive(Debug, Error)]
pub enum StochError {
    #[error("stoch: Empty data provided.")]
    EmptyData,
    #[error("stoch: Mismatched length of input data (high, low, close).")]
    MismatchedLength,
    #[error("stoch: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("stoch: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stoch: All values are NaN.")]
    AllValuesNaN,
    #[error("stoch: {0}")]
    Other(String),
}

// === Indicator Functions (API Parity) ===

#[inline]
pub fn stoch(input: &StochInput) -> Result<StochOutput, StochError> {
    stoch_with_kernel(input, Kernel::Auto)
}

pub fn stoch_with_kernel(input: &StochInput, kernel: Kernel) -> Result<StochOutput, StochError> {
    let (high, low, close) = match &input.data {
        StochData::Candles { candles } => {
            let high = candles.select_candle_field("high").map_err(|e| StochError::Other(e.to_string()))?;
            let low = candles.select_candle_field("low").map_err(|e| StochError::Other(e.to_string()))?;
            let close = candles.select_candle_field("close").map_err(|e| StochError::Other(e.to_string()))?;
            (high, low, close)
        }
        StochData::Slices { high, low, close } => (*high, *low, *close),
    };

    let data_len = high.len();
    if data_len == 0 || low.is_empty() || close.is_empty() {
        return Err(StochError::EmptyData);
    }
    if data_len != low.len() || data_len != close.len() {
        return Err(StochError::MismatchedLength);
    }

    let fastk_period = input.get_fastk_period();
    let slowk_period = input.get_slowk_period();
    let slowd_period = input.get_slowd_period();

    if fastk_period == 0 || fastk_period > data_len {
        return Err(StochError::InvalidPeriod { period: fastk_period, data_len });
    }
    if slowk_period == 0 || slowk_period > data_len {
        return Err(StochError::InvalidPeriod { period: slowk_period, data_len });
    }
    if slowd_period == 0 || slowd_period > data_len {
        return Err(StochError::InvalidPeriod { period: slowd_period, data_len });
    }

    let first_valid_idx = high.iter().zip(low.iter()).zip(close.iter()).position(|((h,l),c)| !h.is_nan() && !l.is_nan() && !c.is_nan()).ok_or(StochError::AllValuesNaN)?;

    if (data_len - first_valid_idx) < fastk_period {
        return Err(StochError::NotEnoughValidData { needed: fastk_period, valid: data_len - first_valid_idx });
    }

    let mut hh = vec![f64::NAN; data_len];
    let mut ll = vec![f64::NAN; data_len];

    let max_vals = max_rolling(&high[first_valid_idx..], fastk_period).map_err(|e| StochError::Other(e.to_string()))?;
    let min_vals = min_rolling(&low[first_valid_idx..], fastk_period).map_err(|e| StochError::Other(e.to_string()))?;

    for (i, &val) in max_vals.iter().enumerate() { hh[i + first_valid_idx] = val; }
    for (i, &val) in min_vals.iter().enumerate() { ll[i + first_valid_idx] = val; }

    let mut k_raw = vec![f64::NAN; data_len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                stoch_scalar(high, low, close, &hh, &ll, fastk_period, first_valid_idx, &mut k_raw)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                stoch_avx2(high, low, close, &hh, &ll, fastk_period, first_valid_idx, &mut k_raw)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                stoch_avx512(high, low, close, &hh, &ll, fastk_period, first_valid_idx, &mut k_raw)
            }
            _ => unreachable!(),
        }
    }

    let slowk_ma_type = input.get_slowk_ma_type();
    let slowd_ma_type = input.get_slowd_ma_type();
    let k_vec = ma(&slowk_ma_type, MaData::Slice(&k_raw), slowk_period).map_err(|e| StochError::Other(e.to_string()))?;
    let d_vec = ma(&slowd_ma_type, MaData::Slice(&k_vec), slowd_period).map_err(|e| StochError::Other(e.to_string()))?;
    Ok(StochOutput { k: k_vec, d: d_vec })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if fastk_period <= 32 {
        unsafe { stoch_avx512_short(high, low, close, hh, ll, fastk_period, first_valid, out) }
    } else {
        unsafe { stoch_avx512_long(high, low, close, hh, ll, fastk_period, first_valid, out) }
    }
}

#[inline]
pub fn stoch_scalar(
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    for i in (first_val + fastk_period - 1)..close.len() {
        let denom = hh[i] - ll[i];
        if denom.abs() < f64::EPSILON {
            out[i] = 50.0;
        } else {
            out[i] = 100.0 * (close[i] - ll[i]) / denom;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    stoch_scalar(high, low, close, hh, ll, fastk_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    stoch_scalar(high, low, close, hh, ll, fastk_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    stoch_scalar(high, low, close, hh, ll, fastk_period, first_valid, out)
}

// === Batch API ===

#[derive(Clone, Debug)]
pub struct StochBatchRange {
    pub fastk_period: (usize, usize, usize),
    pub slowk_period: (usize, usize, usize),
    pub slowk_ma_type: (String, String, f64), // Step as dummy, static only
    pub slowd_period: (usize, usize, usize),
    pub slowd_ma_type: (String, String, f64), // Step as dummy, static only
}

impl Default for StochBatchRange {
    fn default() -> Self {
        Self {
            fastk_period: (14, 14, 0),
            slowk_period: (3, 3, 0),
            slowk_ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
            slowd_period: (3, 3, 0),
            slowd_ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StochBatchBuilder {
    range: StochBatchRange,
    kernel: Kernel,
}

impl StochBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn fastk_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fastk_period = (start, end, step); self
    }
    pub fn fastk_period_static(mut self, p: usize) -> Self {
        self.range.fastk_period = (p, p, 0); self
    }
    pub fn slowk_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slowk_period = (start, end, step); self
    }
    pub fn slowk_period_static(mut self, p: usize) -> Self {
        self.range.slowk_period = (p, p, 0); self
    }
    pub fn slowk_ma_type_static(mut self, t: &str) -> Self {
        self.range.slowk_ma_type = (t.to_string(), t.to_string(), 0.0); self
    }
    pub fn slowd_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slowd_period = (start, end, step); self
    }
    pub fn slowd_period_static(mut self, p: usize) -> Self {
        self.range.slowd_period = (p, p, 0); self
    }
    pub fn slowd_ma_type_static(mut self, t: &str) -> Self {
        self.range.slowd_ma_type = (t.to_string(), t.to_string(), 0.0); self
    }

    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<StochBatchOutput, StochError> {
        stoch_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<StochBatchOutput, StochError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
}

pub fn stoch_batch_with_kernel(
    high: &[f64], low: &[f64], close: &[f64],
    sweep: &StochBatchRange, k: Kernel
) -> Result<StochBatchOutput, StochError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(StochError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    stoch_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct StochBatchOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub combos: Vec<StochParams>,
    pub rows: usize,
    pub cols: usize,
}
impl StochBatchOutput {
    pub fn row_for_params(&self, p: &StochParams) -> Option<usize> {
        self.combos.iter().position(|c|
            c.fastk_period == p.fastk_period &&
            c.slowk_period == p.slowk_period &&
            c.slowk_ma_type == p.slowk_ma_type &&
            c.slowd_period == p.slowd_period &&
            c.slowd_ma_type == p.slowd_ma_type
        )
    }
    pub fn values_for(&self, p: &StochParams) -> Option<(&[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (&self.k[start..start + self.cols], &self.d[start..start + self.cols])
        })
    }
}

#[inline(always)]
fn expand_grid(r: &StochBatchRange) -> Vec<StochParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_str((start, end, _): (String, String, f64)) -> Vec<String> {
        if start == end { vec![start] } else { vec![start, end] }
    }
    let fastk_periods = axis_usize(r.fastk_period);
    let slowk_periods = axis_usize(r.slowk_period);
    let slowk_types = axis_str(r.slowk_ma_type.clone());
    let slowd_periods = axis_usize(r.slowd_period);
    let slowd_types = axis_str(r.slowd_ma_type.clone());
    let mut out = Vec::with_capacity(fastk_periods.len() * slowk_periods.len() * slowk_types.len() * slowd_periods.len() * slowd_types.len());
    for &fkp in &fastk_periods {
        for &skp in &slowk_periods {
            for skt in &slowk_types {
                for &sdp in &slowd_periods {
                    for sdt in &slowd_types {
                        out.push(StochParams {
                            fastk_period: Some(fkp),
                            slowk_period: Some(skp),
                            slowk_ma_type: Some(skt.clone()),
                            slowd_period: Some(sdp),
                            slowd_ma_type: Some(sdt.clone()),
                        });
                    }
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn stoch_batch_slice(
    high: &[f64], low: &[f64], close: &[f64],
    sweep: &StochBatchRange, kern: Kernel,
) -> Result<StochBatchOutput, StochError> {
    stoch_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn stoch_batch_par_slice(
    high: &[f64], low: &[f64], close: &[f64],
    sweep: &StochBatchRange, kern: Kernel,
) -> Result<StochBatchOutput, StochError> {
    stoch_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn stoch_batch_inner(
    high: &[f64], low: &[f64], close: &[f64],
    sweep: &StochBatchRange, kern: Kernel, parallel: bool,
) -> Result<StochBatchOutput, StochError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(StochError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = high.iter().zip(low.iter()).zip(close.iter()).position(|((h,l),c)| !h.is_nan() && !l.is_nan() && !c.is_nan()).ok_or(StochError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.fastk_period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(StochError::NotEnoughValidData { needed: max_p, valid: high.len() - first });
    }
    let rows = combos.len();
    let cols = high.len();
    let mut k_mat = vec![f64::NAN; rows * cols];
    let mut d_mat = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, outk: &mut [f64], outd: &mut [f64]| unsafe {
        let prm = &combos[row];
        let mut hh = vec![f64::NAN; cols];
        let mut ll = vec![f64::NAN; cols];
        let max_vals = max_rolling(&high[first..], prm.fastk_period.unwrap()).map_err(|e| StochError::Other(e.to_string())).unwrap();
        let min_vals = min_rolling(&low[first..], prm.fastk_period.unwrap()).map_err(|e| StochError::Other(e.to_string())).unwrap();
        for (i, &val) in max_vals.iter().enumerate() { hh[i + first] = val; }
        for (i, &val) in min_vals.iter().enumerate() { ll[i + first] = val; }
        let mut kraw = vec![f64::NAN; cols];
        match kern {
            Kernel::Scalar => stoch_row_scalar(high, low, close, &hh, &ll, prm.fastk_period.unwrap(), first, &mut kraw),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => stoch_row_avx2(high, low, close, &hh, &ll, prm.fastk_period.unwrap(), first, &mut kraw),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => stoch_row_avx512(high, low, close, &hh, &ll, prm.fastk_period.unwrap(), first, &mut kraw),
            _ => unreachable!(),
        }
        let k = ma(&prm.slowk_ma_type.clone().unwrap(), MaData::Slice(&kraw), prm.slowk_period.unwrap()).unwrap();
        let d = ma(&prm.slowd_ma_type.clone().unwrap(), MaData::Slice(&k), prm.slowd_period.unwrap()).unwrap();
        outk.copy_from_slice(&k);
        outd.copy_from_slice(&d);
    };
    if parallel {
        k_mat.par_chunks_mut(cols)
            .zip(d_mat.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (k, d))| do_row(row, k, d));
    } else {
        for (row, (k, d)) in k_mat.chunks_mut(cols).zip(d_mat.chunks_mut(cols)).enumerate() {
            do_row(row, k, d);
        }
    }
    Ok(StochBatchOutput { k: k_mat, d: d_mat, combos, rows, cols })
}

#[inline(always)]
unsafe fn stoch_row_scalar(
    _high: &[f64], _low: &[f64], close: &[f64], hh: &[f64], ll: &[f64],
    fastk_period: usize, first: usize, out: &mut [f64]
) {
    for i in (first + fastk_period - 1)..close.len() {
        let denom = hh[i] - ll[i];
        if denom.abs() < f64::EPSILON {
            out[i] = 50.0;
        } else {
            out[i] = 100.0 * (close[i] - ll[i]) / denom;
        }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx2(
    high: &[f64], low: &[f64], close: &[f64], hh: &[f64], ll: &[f64],
    fastk_period: usize, first: usize, out: &mut [f64]
) {
    stoch_row_scalar(high, low, close, hh, ll, fastk_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx512(
    high: &[f64], low: &[f64], close: &[f64], hh: &[f64], ll: &[f64],
    fastk_period: usize, first: usize, out: &mut [f64]
) {
    if fastk_period <= 32 {
        stoch_row_avx512_short(high, low, close, hh, ll, fastk_period, first, out)
    } else {
        stoch_row_avx512_long(high, low, close, hh, ll, fastk_period, first, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], hh: &[f64], ll: &[f64],
    fastk_period: usize, first: usize, out: &mut [f64]
) {
    stoch_row_scalar(high, low, close, hh, ll, fastk_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], hh: &[f64], ll: &[f64],
    fastk_period: usize, first: usize, out: &mut [f64]
) {
    stoch_row_scalar(high, low, close, hh, ll, fastk_period, first, out)
}

// === Streaming ===

#[derive(Debug, Clone)]
pub struct StochStream {
    fastk_period: usize,
    slowk_period: usize,
    slowk_ma_type: String,
    slowd_period: usize,
    slowd_ma_type: String,
    high_buf: Vec<f64>,
    low_buf: Vec<f64>,
    close_buf: Vec<f64>,
    k_stream: Option<Vec<f64>>,
    d_stream: Option<Vec<f64>>,
    head: usize,
    filled: bool,
}

impl StochStream {
    pub fn try_new(params: StochParams) -> Result<Self, StochError> {
        let fastk_period = params.fastk_period.unwrap_or(14);
        let slowk_period = params.slowk_period.unwrap_or(3);
        let slowd_period = params.slowd_period.unwrap_or(3);
        if fastk_period == 0 || slowk_period == 0 || slowd_period == 0 {
            return Err(StochError::InvalidPeriod { period: 0, data_len: 0 });
        }
        Ok(Self {
            fastk_period,
            slowk_period,
            slowk_ma_type: params.slowk_ma_type.unwrap_or_else(|| "sma".to_string()),
            slowd_period,
            slowd_ma_type: params.slowd_ma_type.unwrap_or_else(|| "sma".to_string()),
            high_buf: vec![f64::NAN; fastk_period],
            low_buf: vec![f64::NAN; fastk_period],
            close_buf: vec![f64::NAN; fastk_period],
            k_stream: None,
            d_stream: None,
            head: 0,
            filled: false,
        })
    }
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.high_buf[self.head] = high;
        self.low_buf[self.head] = low;
        self.close_buf[self.head] = close;
        self.head = (self.head + 1) % self.fastk_period;
        if !self.filled && self.head == 0 { self.filled = true; }
        if !self.filled { return None; }
        let start = if self.head == 0 { 0 } else { self.head };
        let mut highs = vec![];
        let mut lows = vec![];
        let mut closes = vec![];
        for i in 0..self.fastk_period {
            let idx = (start + i) % self.fastk_period;
            highs.push(self.high_buf[idx]);
            lows.push(self.low_buf[idx]);
            closes.push(self.close_buf[idx]);
        }
        let max_h = highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_l = lows.iter().cloned().fold(f64::INFINITY, f64::min);
        let last_close = closes[self.fastk_period - 1];
        let k_val = if (max_h - min_l).abs() < f64::EPSILON { 50.0 } else { 100.0 * (last_close - min_l) / (max_h - min_l) };
        let mut k_vec = self.k_stream.take().unwrap_or_else(|| vec![f64::NAN; self.slowk_period]);
        k_vec.remove(0); k_vec.push(k_val);
        self.k_stream = Some(k_vec.clone());
        let slowk = ma(&self.slowk_ma_type, MaData::Slice(&k_vec), self.slowk_period).unwrap();
        let k_last = *slowk.last().unwrap_or(&f64::NAN);
        let mut d_vec = self.d_stream.take().unwrap_or_else(|| vec![f64::NAN; self.slowd_period]);
        d_vec.remove(0); d_vec.push(k_last);
        self.d_stream = Some(d_vec.clone());
        let slowd = ma(&self.slowd_ma_type, MaData::Slice(&d_vec), self.slowd_period).unwrap();
        let d_last = *slowd.last().unwrap_or(&f64::NAN);
        Some((k_last, d_last))
    }
}

// === Tests ===

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_stoch_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = StochParams::default();
        let input = StochInput::from_candles(&candles, default_params);
        let output = stoch_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        Ok(())
    }
    fn check_stoch_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StochInput::from_candles(&candles, StochParams::default());
        let result = stoch_with_kernel(&input, kernel)?;
        assert_eq!(result.k.len(), candles.close.len());
        assert_eq!(result.d.len(), candles.close.len());
        let last_five_k = [
            42.51122827572717,
            40.13864479593807,
            37.853934778363374,
            37.337021714266086,
            36.26053890551548,
        ];
        let last_five_d = [
            41.36561869426493,
            41.7691857059163,
            40.16793595000925,
            38.44320042952222,
            37.15049846604803,
        ];
        let k_slice = &result.k[result.k.len() - 5..];
        let d_slice = &result.d[result.d.len() - 5..];
        for i in 0..5 {
            assert!((k_slice[i] - last_five_k[i]).abs() < 1e-6, "Mismatch in K at {}: got {}, expected {}", i, k_slice[i], last_five_k[i]);
            assert!((d_slice[i] - last_five_d[i]).abs() < 1e-6, "Mismatch in D at {}: got {}, expected {}", i, d_slice[i], last_five_d[i]);
        }
        Ok(())
    }
    fn check_stoch_default_candles(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StochInput::with_default_candles(&candles);
        let output = stoch_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        Ok(())
    }
    fn check_stoch_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 9.5, 10.5];
        let close = [9.5, 10.6, 11.5];
        let params = StochParams { fastk_period: Some(0), ..Default::default() };
        let input = StochInput::from_slices(&high, &low, &close, params);
        let result = stoch_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }
    fn check_stoch_period_exceeds_length(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 9.5, 10.5];
        let close = [9.5, 10.6, 11.5];
        let params = StochParams { fastk_period: Some(10), ..Default::default() };
        let input = StochInput::from_slices(&high, &low, &close, params);
        let result = stoch_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }
    fn check_stoch_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = StochParams::default();
        let input = StochInput::from_slices(&nan_data, &nan_data, &nan_data, params);
        let result = stoch_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }
    macro_rules! generate_all_stoch_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }
    generate_all_stoch_tests!(
        check_stoch_partial_params,
        check_stoch_accuracy,
        check_stoch_default_candles,
        check_stoch_zero_period,
        check_stoch_period_exceeds_length,
        check_stoch_all_nan
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
    skip_if_unsupported!(kernel, test);

    let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let c = read_candles_from_csv(file)?;

    let output = StochBatchBuilder::new()
        .kernel(kernel)
        .apply_candles(&c)?;

    let def = StochParams::default();
    let (row_k, row_d) = output.values_for(&def).expect("default row missing");

    assert_eq!(row_k.len(), c.close.len());
    assert_eq!(row_d.len(), c.close.len());

    let expected_k = [
        42.51122827572717,
        40.13864479593807,
        37.853934778363374,
        37.337021714266086,
        36.26053890551548,
    ];
    let expected_d = [
        41.36561869426493,
        41.7691857059163,
        40.16793595000925,
        38.44320042952222,
        37.15049846604803,
    ];
    let start = row_k.len() - 5;
    for (i, &v) in row_k[start..].iter().enumerate() {
        assert!(
            (v - expected_k[i]).abs() < 1e-6,
            "[{test}] default-row K mismatch at idx {i}: {v} vs {expected_k:?}"
        );
    }
    for (i, &v) in row_d[start..].iter().enumerate() {
        assert!(
            (v - expected_d[i]).abs() < 1e-6,
            "[{test}] default-row D mismatch at idx {i}: {v} vs {expected_d:?}"
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
