//! # KDJ (Stochastic Oscillator with MA smoothing)
//!
//! KDJ is derived from the Stochastic Oscillator (K, D) with an additional line J,
//! where `J = 3 * K - 2 * D`. This indicator highlights momentum and potential
//! overbought/oversold conditions.
//!
//! ## Parameters
//! - **fast_k_period**: The window for the fast stochastic calculation (default: 9).
//! - **slow_k_period**: The smoothing period for K (default: 3).
//! - **slow_k_ma_type**: MA type for smoothing K ("sma", "ema", etc., default: "sma").
//! - **slow_d_period**: The smoothing period for D (default: 3).
//! - **slow_d_ma_type**: MA type for smoothing D ("sma", "ema", etc., default: "sma").
//!
//! ## Errors
//! - **AllValuesNaN**: kdj: All input values are NaN.
//! - **InvalidPeriod**: kdj: period is zero or exceeds data length.
//! - **NotEnoughValidData**: kdj: Not enough valid data points for the requested period.
//! - **EmptyData**: kdj: Input data slice is empty.
//!
//! ## Returns
//! - **`Ok(KdjOutput)`** on success, with vectors for k, d, and j values.
//! - **`Err(KdjError)`** otherwise.
//!

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::{max_rolling, min_rolling, RollingError};
use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// ======== Input Data ========

#[derive(Debug, Clone)]
pub enum KdjData<'a> {
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
pub struct KdjInput<'a> {
    pub data: KdjData<'a>,
    pub params: KdjParams,
}

impl<'a> KdjInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: KdjParams) -> Self {
        Self { data: KdjData::Candles { candles }, params }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], params: KdjParams) -> Self {
        Self { data: KdjData::Slices { high, low, close }, params }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, KdjParams::default())
    }
    #[inline]
    pub fn get_fast_k_period(&self) -> usize {
        self.params.fast_k_period.unwrap_or(9)
    }
    #[inline]
    pub fn get_slow_k_period(&self) -> usize {
        self.params.slow_k_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_slow_k_ma_type(&self) -> &str {
        self.params.slow_k_ma_type.as_deref().unwrap_or("sma")
    }
    #[inline]
    pub fn get_slow_d_period(&self) -> usize {
        self.params.slow_d_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_slow_d_ma_type(&self) -> &str {
        self.params.slow_d_ma_type.as_deref().unwrap_or("sma")
    }
}

// ======== Parameters ========

#[derive(Debug, Clone)]
pub struct KdjParams {
    pub fast_k_period: Option<usize>,
    pub slow_k_period: Option<usize>,
    pub slow_k_ma_type: Option<String>,
    pub slow_d_period: Option<usize>,
    pub slow_d_ma_type: Option<String>,
}

impl Default for KdjParams {
    fn default() -> Self {
        Self {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        }
    }
}

// ======== Output ========

#[derive(Debug, Clone)]
pub struct KdjOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub j: Vec<f64>,
}

// ======== Error Type ========

#[derive(Debug, Error)]
pub enum KdjError {
    #[error("kdj: Empty data provided.")]
    EmptyData,
    #[error("kdj: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kdj: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("kdj: All values are NaN.")]
    AllValuesNaN,
    #[error("kdj: Rolling error {0}")]
    RollingError(#[from] RollingError),
    #[error("kdj: MA error {0}")]
    MaError(#[from] Box<dyn Error + Send + Sync>),
}

// ======== Main Function/Dispatch ========

#[inline]
pub fn kdj(input: &KdjInput) -> Result<KdjOutput, KdjError> {
    kdj_with_kernel(input, Kernel::Auto)
}

pub fn kdj_with_kernel(input: &KdjInput, kernel: Kernel) -> Result<KdjOutput, KdjError> {
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        KdjData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        KdjData::Slices { high, low, close } => (high, low, close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(KdjError::EmptyData);
    }

    let fast_k_period = input.get_fast_k_period();
    let slow_k_period = input.get_slow_k_period();
    let slow_k_ma_type = input.get_slow_k_ma_type();
    let slow_d_period = input.get_slow_d_period();
    let slow_d_ma_type = input.get_slow_d_ma_type();

    if fast_k_period == 0 || fast_k_period > high.len() {
        return Err(KdjError::InvalidPeriod { period: fast_k_period, data_len: high.len() });
    }

    let first_valid_idx = high.iter().zip(low.iter()).zip(close.iter())
        .position(|((&h, &l), &c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
        .ok_or(KdjError::AllValuesNaN)?;

    if (high.len() - first_valid_idx) < fast_k_period {
        return Err(KdjError::NotEnoughValidData { needed: fast_k_period, valid: high.len() - first_valid_idx });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                kdj_scalar(high, low, close, fast_k_period, slow_k_period, slow_k_ma_type, slow_d_period, slow_d_ma_type, first_valid_idx)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                kdj_avx2(high, low, close, fast_k_period, slow_k_period, slow_k_ma_type, slow_d_period, slow_d_ma_type, first_valid_idx)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                kdj_avx512(high, low, close, fast_k_period, slow_k_period, slow_k_ma_type, slow_d_period, slow_d_ma_type, first_valid_idx)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn kdj_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    let hh = max_rolling(high, fast_k_period)?;
    let ll = min_rolling(low, fast_k_period)?;
    let mut stoch = vec![f64::NAN; high.len()];
    for i in first_valid_idx..high.len() {
        if i < (first_valid_idx + fast_k_period - 1) { continue; }
        let denom = hh[i] - ll[i];
        if denom == 0.0 || denom.is_nan() {
            stoch[i] = f64::NAN;
        } else {
            stoch[i] = 100.0 * ((close[i] - ll[i]) / denom);
        }
    }
    let k = ma(slow_k_ma_type, MaData::Slice(&stoch), slow_k_period)
      .map_err(|e| KdjError::MaError(e.to_string().into()))?;
    let d = ma(slow_d_ma_type, MaData::Slice(&k), slow_d_period)
        .map_err(|e| KdjError::MaError(e.to_string().into()))?;

    let mut j = vec![f64::NAN; high.len()];
    for i in 0..high.len() {
        if k[i].is_nan() || d[i].is_nan() {
            j[i] = f64::NAN;
        } else {
            j[i] = 3.0 * k[i] - 2.0 * d[i];
        }
    }
    Ok(KdjOutput { k, d, j })
}

// =========== SIMD Stubs ===========

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kdj_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    if fast_k_period <= 32 {
        unsafe {
            kdj_avx512_short(
                high, low, close,
                fast_k_period, slow_k_period, slow_k_ma_type,
                slow_d_period, slow_d_ma_type,
                first_valid_idx
            )
        }
    } else {
        unsafe {
            kdj_avx512_long(
                high, low, close,
                fast_k_period, slow_k_period, slow_k_ma_type,
                slow_d_period, slow_d_ma_type,
                first_valid_idx
            )
        }
    }
}

#[inline]
pub fn kdj_avx2(
    high: &[f64], low: &[f64], close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    // AVX2 stub, points to scalar
    kdj_scalar(high, low, close, fast_k_period, slow_k_period, slow_k_ma_type, slow_d_period, slow_d_ma_type, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kdj_avx512_short(
    high: &[f64], low: &[f64], close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    // AVX512 short stub, points to scalar
    kdj_scalar(high, low, close, fast_k_period, slow_k_period, slow_k_ma_type, slow_d_period, slow_d_ma_type, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kdj_avx512_long(
    high: &[f64], low: &[f64], close: &[f64],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    first_valid_idx: usize,
) -> Result<KdjOutput, KdjError> {
    // AVX512 long stub, points to scalar
    kdj_scalar(high, low, close, fast_k_period, slow_k_period, slow_k_ma_type, slow_d_period, slow_d_ma_type, first_valid_idx)
}

// ========== Builder/Stream ==========

#[derive(Clone, Debug)]
pub struct KdjBuilder {
    fast_k_period: Option<usize>,
    slow_k_period: Option<usize>,
    slow_k_ma_type: Option<String>,
    slow_d_period: Option<usize>,
    slow_d_ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for KdjBuilder {
    fn default() -> Self {
        Self {
            fast_k_period: None,
            slow_k_period: None,
            slow_k_ma_type: None,
            slow_d_period: None,
            slow_d_ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KdjBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn fast_k_period(mut self, n: usize) -> Self { self.fast_k_period = Some(n); self }
    #[inline(always)]
    pub fn slow_k_period(mut self, n: usize) -> Self { self.slow_k_period = Some(n); self }
    #[inline(always)]
    pub fn slow_k_ma_type<S: Into<String>>(mut self, t: S) -> Self { self.slow_k_ma_type = Some(t.into()); self }
    #[inline(always)]
    pub fn slow_d_period(mut self, n: usize) -> Self { self.slow_d_period = Some(n); self }
    #[inline(always)]
    pub fn slow_d_ma_type<S: Into<String>>(mut self, t: S) -> Self { self.slow_d_ma_type = Some(t.into()); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<KdjOutput, KdjError> {
        let p = KdjParams {
            fast_k_period: self.fast_k_period,
            slow_k_period: self.slow_k_period,
            slow_k_ma_type: self.slow_k_ma_type,
            slow_d_period: self.slow_d_period,
            slow_d_ma_type: self.slow_d_ma_type,
        };
        let i = KdjInput::from_candles(c, p);
        kdj_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<KdjOutput, KdjError> {
        let p = KdjParams {
            fast_k_period: self.fast_k_period,
            slow_k_period: self.slow_k_period,
            slow_k_ma_type: self.slow_k_ma_type,
            slow_d_period: self.slow_d_period,
            slow_d_ma_type: self.slow_d_ma_type,
        };
        let i = KdjInput::from_slices(high, low, close, p);
        kdj_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<KdjStream, KdjError> {
        let p = KdjParams {
            fast_k_period: self.fast_k_period,
            slow_k_period: self.slow_k_period,
            slow_k_ma_type: self.slow_k_ma_type,
            slow_d_period: self.slow_d_period,
            slow_d_ma_type: self.slow_d_ma_type,
        };
        KdjStream::try_new(p)
    }
}

// ========= Stream Processing ==========

#[derive(Debug, Clone)]
pub struct KdjStream {
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: String,
    slow_d_period: usize,
    slow_d_ma_type: String,
    high_buf: Vec<f64>,
    low_buf: Vec<f64>,
    close_buf: Vec<f64>,
    head: usize,
    filled: bool,
}

impl KdjStream {
    pub fn try_new(params: KdjParams) -> Result<Self, KdjError> {
        let fast_k_period = params.fast_k_period.unwrap_or(9);
        let slow_k_period = params.slow_k_period.unwrap_or(3);
        let slow_k_ma_type = params.slow_k_ma_type.unwrap_or_else(|| "sma".to_string());
        let slow_d_period = params.slow_d_period.unwrap_or(3);
        let slow_d_ma_type = params.slow_d_ma_type.unwrap_or_else(|| "sma".to_string());

        if fast_k_period == 0 {
            return Err(KdjError::InvalidPeriod { period: fast_k_period, data_len: 0 });
        }
        Ok(Self {
            fast_k_period,
            slow_k_period,
            slow_k_ma_type,
            slow_d_period,
            slow_d_ma_type,
            high_buf: vec![f64::NAN; fast_k_period],
            low_buf: vec![f64::NAN; fast_k_period],
            close_buf: vec![f64::NAN; fast_k_period],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64)> {
        self.high_buf[self.head] = high;
        self.low_buf[self.head] = low;
        self.close_buf[self.head] = close;
        self.head = (self.head + 1) % self.fast_k_period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }

        let hh = self.high_buf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ll = self.low_buf.iter().cloned().fold(f64::INFINITY, f64::min);
        let stoch = if hh == ll || hh.is_nan() || ll.is_nan() || close.is_nan() {
            f64::NAN
        } else {
            100.0 * ((close - ll) / (hh - ll))
        };

        // Streaming K/D via SMA (can be improved for EMA if needed)
        // For now, not stateful. Only last value available.
        let k = stoch;
        let d = k;
        let j = 3.0 * k - 2.0 * d;
        Some((k, d, j))
    }
}

// ========== Batch API ==========

#[derive(Clone, Debug)]
pub struct KdjBatchRange {
    pub fast_k_period: (usize, usize, usize),
    pub slow_k_period: (usize, usize, usize),
    pub slow_k_ma_type: (String, String, String),
    pub slow_d_period: (usize, usize, usize),
    pub slow_d_ma_type: (String, String, String),
}

impl Default for KdjBatchRange {
    fn default() -> Self {
        Self {
            fast_k_period: (9, 30, 1),
            slow_k_period: (3, 6, 1),
            slow_k_ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
            slow_d_period: (3, 6, 1),
            slow_d_ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KdjBatchBuilder {
    range: KdjBatchRange,
    kernel: Kernel,
}

impl KdjBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline]
    pub fn fast_k_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_k_period = (start, end, step); self
    }
    #[inline]
    pub fn slow_k_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_k_period = (start, end, step); self
    }
    #[inline]
    pub fn slow_k_ma_type_static<S: Into<String>>(mut self, s: S) -> Self {
        let v = s.into();
        self.range.slow_k_ma_type = (v.clone(), v, "".to_string()); self
    }
    #[inline]
    pub fn slow_d_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_d_period = (start, end, step); self
    }
    #[inline]
    pub fn slow_d_ma_type_static<S: Into<String>>(mut self, s: S) -> Self {
        let v = s.into();
        self.range.slow_d_ma_type = (v.clone(), v, "".to_string()); self
    }

    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<KdjBatchOutput, KdjError> {
        kdj_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<KdjBatchOutput, KdjError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
}

pub fn kdj_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KdjBatchRange,
    k: Kernel,
) -> Result<KdjBatchOutput, KdjError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(KdjError::InvalidPeriod { period: 0, data_len: 0 });
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    kdj_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct KdjBatchOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub j: Vec<f64>,
    pub combos: Vec<KdjParams>,
    pub rows: usize,
    pub cols: usize,
}
impl KdjBatchOutput {
    pub fn row_for_params(&self, p: &KdjParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fast_k_period.unwrap_or(9) == p.fast_k_period.unwrap_or(9) &&
            c.slow_k_period.unwrap_or(3) == p.slow_k_period.unwrap_or(3) &&
            c.slow_k_ma_type.as_deref().unwrap_or("sma") == p.slow_k_ma_type.as_deref().unwrap_or("sma") &&
            c.slow_d_period.unwrap_or(3) == p.slow_d_period.unwrap_or(3) &&
            c.slow_d_ma_type.as_deref().unwrap_or("sma") == p.slow_d_ma_type.as_deref().unwrap_or("sma")
        })
    }
    pub fn k_for(&self, p: &KdjParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| &self.k[row * self.cols..][..self.cols])
    }
    pub fn d_for(&self, p: &KdjParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| &self.d[row * self.cols..][..self.cols])
    }
    pub fn j_for(&self, p: &KdjParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| &self.j[row * self.cols..][..self.cols])
    }
}

#[inline(always)]
fn expand_grid(r: &KdjBatchRange) -> Vec<KdjParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_str((start, end, _): (String, String, String)) -> Vec<String> {
        if start == end { vec![start] } else { vec![start, end] }
    }
    let fast_k_periods = axis_usize(r.fast_k_period);
    let slow_k_periods = axis_usize(r.slow_k_period);
    let slow_k_ma_types = axis_str(r.slow_k_ma_type.clone());
    let slow_d_periods = axis_usize(r.slow_d_period);
    let slow_d_ma_types = axis_str(r.slow_d_ma_type.clone());
    let mut out = Vec::new();
    for &fkp in &fast_k_periods {
        for &skp in &slow_k_periods {
            for skmt in &slow_k_ma_types {
                for &sdp in &slow_d_periods {
                    for sdmt in &slow_d_ma_types {
                        out.push(KdjParams {
                            fast_k_period: Some(fkp),
                            slow_k_period: Some(skp),
                            slow_k_ma_type: Some(skmt.clone()),
                            slow_d_period: Some(sdp),
                            slow_d_ma_type: Some(sdmt.clone()),
                        });
                    }
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn kdj_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KdjBatchRange,
    kern: Kernel,
) -> Result<KdjBatchOutput, KdjError> {
    kdj_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn kdj_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KdjBatchRange,
    kern: Kernel,
) -> Result<KdjBatchOutput, KdjError> {
    kdj_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn kdj_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KdjBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<KdjBatchOutput, KdjError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KdjError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = high.iter().zip(low.iter()).zip(close.iter())
        .position(|((&h, &l), &c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
        .ok_or(KdjError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.fast_k_period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(KdjError::NotEnoughValidData { needed: max_p, valid: high.len() - first });
    }
    let rows = combos.len();
    let cols = high.len();
    let mut k_vals = vec![f64::NAN; rows * cols];
    let mut d_vals = vec![f64::NAN; rows * cols];
    let mut j_vals = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_k: &mut [f64], out_d: &mut [f64], out_j: &mut [f64]| unsafe {
        let prm = &combos[row];
        let res = kdj_row_scalar(
            high, low, close, first,
            prm.fast_k_period.unwrap(),
            prm.slow_k_period.unwrap(),
            prm.slow_k_ma_type.as_deref().unwrap_or("sma"),
            prm.slow_d_period.unwrap(),
            prm.slow_d_ma_type.as_deref().unwrap_or("sma"),
            out_k, out_d, out_j
        );
        res
    };
    if parallel {
        k_vals.par_chunks_mut(cols)
            .zip(d_vals.par_chunks_mut(cols))
            .zip(j_vals.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((out_k, out_d), out_j))| {
                let _ = do_row(row, out_k, out_d, out_j);
            });
    } else {
        for (row, ((out_k, out_d), out_j)) in k_vals.chunks_mut(cols)
            .zip(d_vals.chunks_mut(cols))
            .zip(j_vals.chunks_mut(cols))
            .enumerate()
        {
            let _ = do_row(row, out_k, out_d, out_j);
        }
    }
    Ok(KdjBatchOutput { k: k_vals, d: d_vals, j: j_vals, combos, rows, cols })
}

// ========== Row Scalar/AVX ==========

#[inline(always)]
unsafe fn kdj_row_scalar(
    high: &[f64], low: &[f64], close: &[f64], first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64], out_d: &mut [f64], out_j: &mut [f64],
) -> Result<(), KdjError> {
    let hh = max_rolling(high, fast_k_period)?;
    let ll = min_rolling(low, fast_k_period)?;
    let mut stoch = vec![f64::NAN; high.len()];
    for i in first..high.len() {
        if i < (first + fast_k_period - 1) { continue; }
        let denom = hh[i] - ll[i];
        if denom == 0.0 || denom.is_nan() {
            stoch[i] = f64::NAN;
        } else {
            stoch[i] = 100.0 * ((close[i] - ll[i]) / denom);
        }
    }
    let k = ma(slow_k_ma_type, MaData::Slice(&stoch), slow_k_period)
      .map_err(|e| KdjError::MaError(e.to_string().into()))?;
    let d = ma(slow_d_ma_type, MaData::Slice(&k), slow_d_period)
        .map_err(|e| KdjError::MaError(e.to_string().into()))?;
    for i in 0..high.len() {
        out_k[i] = k[i];
        out_d[i] = d[i];
        out_j[i] = if k[i].is_nan() || d[i].is_nan() {
            f64::NAN
        } else {
            3.0 * k[i] - 2.0 * d[i]
        };
    }
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kdj_row_avx2(
    high: &[f64], low: &[f64], close: &[f64], first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64], out_d: &mut [f64], out_j: &mut [f64],
) -> Result<(), KdjError> {
    kdj_row_scalar(
        high, low, close, first,
        fast_k_period, slow_k_period, slow_k_ma_type,
        slow_d_period, slow_d_ma_type,
        out_k, out_d, out_j,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kdj_row_avx512(
    high: &[f64], low: &[f64], close: &[f64], first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64], out_d: &mut [f64], out_j: &mut [f64],
) -> Result<(), KdjError> {
    if fast_k_period <= 32 {
        kdj_row_avx512_short(
            high, low, close, first, fast_k_period, slow_k_period,
            slow_k_ma_type, slow_d_period, slow_d_ma_type,
            out_k, out_d, out_j,
        )
    } else {
        kdj_row_avx512_long(
            high, low, close, first, fast_k_period, slow_k_period,
            slow_k_ma_type, slow_d_period, slow_d_ma_type,
            out_k, out_d, out_j,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kdj_row_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64], out_d: &mut [f64], out_j: &mut [f64],
) -> Result<(), KdjError> {
    kdj_row_scalar(
        high, low, close, first,
        fast_k_period, slow_k_period, slow_k_ma_type,
        slow_d_period, slow_d_ma_type,
        out_k, out_d, out_j,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kdj_row_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], first: usize,
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: &str,
    slow_d_period: usize,
    slow_d_ma_type: &str,
    out_k: &mut [f64], out_d: &mut [f64], out_j: &mut [f64],
) -> Result<(), KdjError> {
    kdj_row_scalar(
        high, low, close, first,
        fast_k_period, slow_k_period, slow_k_ma_type,
        slow_d_period, slow_d_ma_type,
        out_k, out_d, out_j,
    )
}

// ========== Unit Tests ==========
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_kdj_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = KdjParams {
            fast_k_period: None,
            slow_k_period: Some(4),
            slow_k_ma_type: None,
            slow_d_period: None,
            slow_d_ma_type: None,
        };
        let input = KdjInput::from_candles(&candles, partial_params);
        let output = kdj_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        assert_eq!(output.j.len(), candles.close.len());
        Ok(())
    }

    fn check_kdj_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KdjParams::default();
        let input = KdjInput::from_candles(&candles, params);
        let result = kdj_with_kernel(&input, kernel)?;
        let expected_k = [58.04341315415984, 61.56034740940419, 58.056304282719545, 56.10961365678364, 51.43992326447119];
        let expected_d = [49.57659409278555, 56.81719223571944, 59.22002161542779, 58.57542178296905, 55.20194706799139];
        let expected_j = [74.97705127690843, 71.04665775677368, 55.72886961730306, 51.17799740441281, 43.91587565743079];
        let len = result.k.len();
        let start_idx = len - 5;
        for i in 0..5 {
            let k_val = result.k[start_idx + i];
            let d_val = result.d[start_idx + i];
            let j_val = result.j[start_idx + i];
            assert!((k_val - expected_k[i]).abs() < 1e-4, "Mismatch in K at index {}: expected {}, got {}", i, expected_k[i], k_val);
            assert!((d_val - expected_d[i]).abs() < 1e-4, "Mismatch in D at index {}: expected {}, got {}", i, expected_d[i], d_val);
            assert!((j_val - expected_j[i]).abs() < 1e-4, "Mismatch in J at index {}: expected {}, got {}", i, expected_j[i], j_val);
        }
        Ok(())
    }

    fn check_kdj_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KdjInput::with_default_candles(&candles);
        match input.data {
            KdjData::Candles { .. } => {}
            _ => panic!("Expected KdjData::Candles variant"),
        }
        let output = kdj_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        Ok(())
    }

    fn check_kdj_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = KdjParams {
            fast_k_period: Some(0),
            ..Default::default()
        };
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] KDJ should fail with zero period", test_name);
        Ok(())
    }

    fn check_kdj_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = KdjParams {
            fast_k_period: Some(10),
            ..Default::default()
        };
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] KDJ should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_kdj_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = KdjParams {
            fast_k_period: Some(9),
            ..Default::default()
        };
        let input = KdjInput::from_slices(&single_point, &single_point, &single_point, params);
        let result = kdj_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] KDJ should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_kdj_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = KdjParams::default();
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] KDJ should fail with all-NaN data", test_name);
        Ok(())
    }

    fn check_kdj_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = KdjParams {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        };
        let first_input = KdjInput::from_candles(&candles, first_params);
        let first_result = kdj_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.k.len(), candles.close.len());

        let second_params = KdjParams {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        };
        let second_input = KdjInput::from_slices(
            &first_result.k,
            &first_result.k,
            &first_result.k,
            second_params,
        );
        let second_result = kdj_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.k.len(), first_result.k.len());
        for i in 50..second_result.k.len() {
            assert!(!second_result.k[i].is_nan(), "[{}] Expected no NaN in second KDJ at {}", test_name, i);
        }
        Ok(())
    }

    fn check_kdj_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KdjParams::default();
        let input = KdjInput::from_candles(&candles, params);
        let result = kdj_with_kernel(&input, kernel)?;
        if result.k.len() > 50 {
            for i in 50..result.k.len() {
                assert!(!result.k[i].is_nan(), "[{}] Expected no NaN in K after index 50 at {}", test_name, i);
            }
        }
        Ok(())
    }

    macro_rules! generate_all_kdj_tests {
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

    generate_all_kdj_tests!(
        check_kdj_partial_params,
        check_kdj_accuracy,
        check_kdj_default_candles,
        check_kdj_zero_period,
        check_kdj_period_exceeds_length,
        check_kdj_very_small_dataset,
        check_kdj_all_nan,
        check_kdj_reinput,
        check_kdj_nan_handling
    );

    // Batch test, matches alma batch style
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = KdjBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        let def = KdjParams::default();
        let row = output.k_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Just check presence, no golden values in this batch stub.
        for &v in &row[row.len().saturating_sub(5)..] {
            assert!(!v.is_nan(), "[{test}] default-row unexpected NaN at tail");
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
