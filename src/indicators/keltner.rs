//! # Keltner Channels
//!
//! A volatility-based envelope indicator. The middle band is a moving average (MA) of a user-specified source,
//! and the upper and lower bands are derived by adding or subtracting a multiple of an internally computed Average True Range (ATR).
//!
//! ## Parameters
//! - **period**: Lookback length for both the moving average and ATR (default: 20).
//! - **multiplier**: ATR multiplier for upper/lower bands (default: 2.0).
//! - **ma_type**: MA type ("ema", "sma", etc.; default: "ema").
//!
//! ## Errors
//! - **KeltnerEmptyData**: keltner: Input data is empty.
//! - **KeltnerInvalidPeriod**: keltner: `period` is zero or exceeds data length.
//! - **KeltnerNotEnoughValidData**: keltner: Not enough valid data after first valid index.
//! - **KeltnerAllValuesNaN**: keltner: All values are NaN.
//! - **KeltnerMaError**: keltner: MA error.
//!
//! ## Returns
//! - **Ok(KeltnerOutput)**: Contains upper_band, middle_band, lower_band.
//! - **Err(KeltnerError)**

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;
use std::convert::AsRef;

// Input & data types

#[derive(Debug, Clone)]
pub enum KeltnerData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), // high, low, close, source
}

#[derive(Debug, Clone)]
pub struct KeltnerOutput {
    pub upper_band: Vec<f64>,
    pub middle_band: Vec<f64>,
    pub lower_band: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KeltnerParams {
    pub period: Option<usize>,
    pub multiplier: Option<f64>,
    pub ma_type: Option<String>,
}

impl Default for KeltnerParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            multiplier: Some(2.0),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeltnerInput<'a> {
    pub data: KeltnerData<'a>,
    pub params: KeltnerParams,
}

impl<'a> KeltnerInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: KeltnerParams) -> Self {
        Self {
            data: KeltnerData::Candles { candles, source },
            params,
        }
    }
    #[inline]
    pub fn from_slice(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        source: &'a [f64],
        params: KeltnerParams,
    ) -> Self {
        Self {
            data: KeltnerData::Slice(high, low, close, source),
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", KeltnerParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_multiplier(&self) -> f64 {
        self.params.multiplier.unwrap_or(2.0)
    }
    #[inline]
    pub fn get_ma_type(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or("ema")
    }
}

#[derive(Clone, Debug)]
pub struct KeltnerBuilder {
    period: Option<usize>,
    multiplier: Option<f64>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for KeltnerBuilder {
    fn default() -> Self {
        Self {
            period: None,
            multiplier: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KeltnerBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self { self.period = Some(n); self }
    #[inline(always)]
    pub fn multiplier(mut self, x: f64) -> Self { self.multiplier = Some(x); self }
    #[inline(always)]
    pub fn ma_type(mut self, mt: &str) -> Self { self.ma_type = Some(mt.to_lowercase()); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<KeltnerOutput, KeltnerError> {
        let p = KeltnerParams {
            period: self.period,
            multiplier: self.multiplier,
            ma_type: self.ma_type,
        };
        let i = KeltnerInput::from_candles(c, "close", p);
        keltner_with_kernel(&i, self.kernel)
    }
}

// Error handling

#[derive(Debug, Error)]
pub enum KeltnerError {
    #[error("keltner: empty data provided.")]
    KeltnerEmptyData,
    #[error("keltner: invalid period: period = {period}, data length = {data_len}")]
    KeltnerInvalidPeriod { period: usize, data_len: usize },
    #[error("keltner: not enough valid data: needed = {needed}, valid = {valid}")]
    KeltnerNotEnoughValidData { needed: usize, valid: usize },
    #[error("keltner: all values are NaN.")]
    KeltnerAllValuesNaN,
    #[error("keltner: MA error: {0}")]
    KeltnerMaError(String),
}

// Core indicator API

#[inline]
pub fn keltner(input: &KeltnerInput) -> Result<KeltnerOutput, KeltnerError> {
    keltner_with_kernel(input, Kernel::Auto)
}

pub fn keltner_with_kernel(input: &KeltnerInput, kernel: Kernel) -> Result<KeltnerOutput, KeltnerError> {
    let (high, low, close, source_slice): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        KeltnerData::Candles { candles, source } => {
            (
                candles.select_candle_field("high").map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?,
                candles.select_candle_field("low").map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?,
                candles.select_candle_field("close").map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?,
                source_type(candles, source)
            )
        }
        KeltnerData::Slice(h, l, c, s) => (*h, *l, *c, *s),
    };
    let period = input.get_period();
    let len = close.len();
    if len == 0 {
        return Err(KeltnerError::KeltnerEmptyData);
    }
    if period == 0 || period > len {
        return Err(KeltnerError::KeltnerInvalidPeriod { period, data_len: len });
    }
    let first = close.iter().position(|x| !x.is_nan()).ok_or(KeltnerError::KeltnerAllValuesNaN)?;

    if (len - first) < period {
        return Err(KeltnerError::KeltnerNotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut upper_band = vec![f64::NAN; len];
    let mut middle_band = vec![f64::NAN; len];
    let mut lower_band = vec![f64::NAN; len];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                keltner_scalar(high, low, close, source_slice, period, input.get_multiplier(), input.get_ma_type(), first, &mut upper_band, &mut middle_band, &mut lower_band)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                keltner_avx2(high, low, close, source_slice, period, input.get_multiplier(), input.get_ma_type(), first, &mut upper_band, &mut middle_band, &mut lower_band)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                keltner_avx512(high, low, close, source_slice, period, input.get_multiplier(), input.get_ma_type(), first, &mut upper_band, &mut middle_band, &mut lower_band)
            }
            _ => unreachable!(),
        }
    }
    Ok(KeltnerOutput { upper_band, middle_band, lower_band })
}

// Scalar calculation (core logic)

#[inline]
pub fn keltner_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    period: usize,
    multiplier: f64,
    ma_type: &str,
    first: usize,
    upper: &mut [f64],
    middle: &mut [f64],
    lower: &mut [f64],
) {
    let len = close.len();
    let mut atr = vec![f64::NAN; len];
    let alpha = 1.0 / (period as f64);
    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;
    for i in 0..len {
        let tr = if i == 0 {
            high[0] - low[0]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        if i < period {
            sum_tr += tr;
            if i == period - 1 {
                rma = sum_tr / (period as f64);
                atr[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr[i] = rma;
        }
    }
    let ma_values = crate::indicators::moving_averages::ma::ma(ma_type, crate::indicators::moving_averages::ma::MaData::Slice(source), period).unwrap_or_else(|_| vec![f64::NAN; len]);
    for i in (first + period - 1)..len {
        let ma_v = ma_values[i];
        let atr_v = atr[i];
        if ma_v.is_nan() || atr_v.is_nan() { continue; }
        middle[i] = ma_v;
        upper[i] = ma_v + multiplier * atr_v;
        lower[i] = ma_v - multiplier * atr_v;
    }
}

// AVX2 stub

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn keltner_avx2(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    period: usize, multiplier: f64, ma_type: &str, first: usize,
    upper: &mut [f64], middle: &mut [f64], lower: &mut [f64]
) {
    keltner_scalar(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower)
}

// AVX512 stub

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn keltner_avx512(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    period: usize, multiplier: f64, ma_type: &str, first: usize,
    upper: &mut [f64], middle: &mut [f64], lower: &mut [f64]
) {
    if period <= 32 {
        unsafe { keltner_avx512_short(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower) }
    } else {
        unsafe { keltner_avx512_long(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn keltner_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    period: usize, multiplier: f64, ma_type: &str, first: usize,
    upper: &mut [f64], middle: &mut [f64], lower: &mut [f64]
) {
    keltner_scalar(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn keltner_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    period: usize, multiplier: f64, ma_type: &str, first: usize,
    upper: &mut [f64], middle: &mut [f64], lower: &mut [f64]
) {
    keltner_scalar(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower)
}

// Row/batch/parallel support

#[inline(always)]
pub fn keltner_row_scalar(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    period: usize, multiplier: f64, ma_type: &str, first: usize,
    upper: &mut [f64], middle: &mut [f64], lower: &mut [f64]
) {
    keltner_scalar(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn keltner_row_avx2(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    period: usize, multiplier: f64, ma_type: &str, first: usize,
    upper: &mut [f64], middle: &mut [f64], lower: &mut [f64]
) {
    keltner_avx2(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn keltner_row_avx512(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    period: usize, multiplier: f64, ma_type: &str, first: usize,
    upper: &mut [f64], middle: &mut [f64], lower: &mut [f64]
) {
    keltner_avx512(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn keltner_row_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    period: usize, multiplier: f64, ma_type: &str, first: usize,
    upper: &mut [f64], middle: &mut [f64], lower: &mut [f64]
) {
    keltner_avx512_short(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn keltner_row_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    period: usize, multiplier: f64, ma_type: &str, first: usize,
    upper: &mut [f64], middle: &mut [f64], lower: &mut [f64]
) {
    keltner_avx512_long(high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower)
}

// Batch support

#[derive(Clone, Debug)]
pub struct KeltnerBatchRange {
    pub period: (usize, usize, usize),
    pub multiplier: (f64, f64, f64),
}

impl Default for KeltnerBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 60, 10),
            multiplier: (2.0, 2.0, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KeltnerBatchBuilder {
    range: KeltnerBatchRange,
    kernel: Kernel,
}

impl Default for KeltnerBatchBuilder {
    fn default() -> Self {
        Self {
            range: KeltnerBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}
impl KeltnerBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    #[inline]
    pub fn multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.multiplier = (start, end, step);
        self
    }
    #[inline]
    pub fn multiplier_static(mut self, m: f64) -> Self {
        self.range.multiplier = (m, m, 0.0);
        self
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<KeltnerBatchOutput, KeltnerError> {
        let h = c.select_candle_field("high").map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?;
        let l = c.select_candle_field("low").map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?;
        let cl = c.select_candle_field("close").map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?;
        let src_v = source_type(c, src);
        self.apply_slice(&h, &l, &cl, src_v)
    }
    pub fn apply_slice(self, high: &[f64], low: &[f64], close: &[f64], source: &[f64]) -> Result<KeltnerBatchOutput, KeltnerError> {
        keltner_batch_with_kernel(high, low, close, source, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct KeltnerBatchOutput {
    pub upper_band: Vec<f64>,
    pub middle_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub combos: Vec<KeltnerParams>,
    pub rows: usize,
    pub cols: usize,
}
impl KeltnerBatchOutput {
    pub fn row_for_params(&self, p: &KeltnerParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20) &&
            (c.multiplier.unwrap_or(2.0) - p.multiplier.unwrap_or(2.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &KeltnerParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.upper_band[start..start + self.cols],
                &self.middle_band[start..start + self.cols],
                &self.lower_band[start..start + self.cols],
            )
        })
    }
}

fn expand_grid(r: &KeltnerBatchRange) -> Vec<KeltnerParams> {
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
    let mults = axis_f64(r.multiplier);
    let mut out = Vec::with_capacity(periods.len() * mults.len());
    for &p in &periods {
        for &m in &mults {
            out.push(KeltnerParams { period: Some(p), multiplier: Some(m), ma_type: None });
        }
    }
    out
}

pub fn keltner_batch_with_kernel(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    sweep: &KeltnerBatchRange, k: Kernel
) -> Result<KeltnerBatchOutput, KeltnerError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(KeltnerError::KeltnerInvalidPeriod { period: 0, data_len: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    keltner_batch_par_slice(high, low, close, source, sweep, simd)
}

pub fn keltner_batch_slice(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    sweep: &KeltnerBatchRange, kern: Kernel
) -> Result<KeltnerBatchOutput, KeltnerError> {
    keltner_batch_inner(high, low, close, source, sweep, kern, false)
}
pub fn keltner_batch_par_slice(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    sweep: &KeltnerBatchRange, kern: Kernel
) -> Result<KeltnerBatchOutput, KeltnerError> {
    keltner_batch_inner(high, low, close, source, sweep, kern, true)
}
fn keltner_batch_inner(
    high: &[f64], low: &[f64], close: &[f64], source: &[f64],
    sweep: &KeltnerBatchRange, kern: Kernel, parallel: bool
) -> Result<KeltnerBatchOutput, KeltnerError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KeltnerError::KeltnerInvalidPeriod { period: 0, data_len: 0 });
    }
    let len = close.len();
    let first = close.iter().position(|x| !x.is_nan()).ok_or(KeltnerError::KeltnerAllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(KeltnerError::KeltnerNotEnoughValidData { needed: max_p, valid: len - first });
    }
    let rows = combos.len();
    let cols = len;
    let mut upper = vec![f64::NAN; rows * cols];
    let mut middle = vec![f64::NAN; rows * cols];
    let mut lower = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, up: &mut [f64], mid: &mut [f64], low_out: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let mult = combos[row].multiplier.unwrap();
        keltner_row_scalar(high, low, close, source, period, mult, "ema", first, up, mid, low_out)
    };
    if parallel {
        upper.par_chunks_mut(cols)
            .zip(middle.par_chunks_mut(cols))
            .zip(lower.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((u, m), l))| do_row(row, u, m, l));
    } else {
        for ((row, u), (m, l)) in upper.chunks_mut(cols).enumerate()
            .zip(middle.chunks_mut(cols).zip(lower.chunks_mut(cols))) {
            do_row(row, u, m, l);
        }
    }
    Ok(KeltnerBatchOutput { upper_band: upper, middle_band: middle, lower_band: lower, combos, rows, cols })
}

// Streaming mode

#[derive(Debug, Clone)]
pub struct KeltnerStream {
    period: usize,
    multiplier: f64,
    ma_type: String,
    atr_stream: Vec<f64>,
    ma_stream: Vec<f64>,
    buffer_high: Vec<f64>,
    buffer_low: Vec<f64>,
    buffer_close: Vec<f64>,
    buffer_source: Vec<f64>,
    head: usize,
    filled: bool,
}

impl KeltnerStream {
    pub fn try_new(params: KeltnerParams) -> Result<Self, KeltnerError> {
        let period = params.period.unwrap_or(20);
        let multiplier = params.multiplier.unwrap_or(2.0);
        let ma_type = params.ma_type.unwrap_or("ema".to_string());
        if period == 0 {
            return Err(KeltnerError::KeltnerInvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            multiplier,
            ma_type,
            atr_stream: vec![f64::NAN; period],
            ma_stream: vec![f64::NAN; period],
            buffer_high: vec![f64::NAN; period],
            buffer_low: vec![f64::NAN; period],
            buffer_close: vec![f64::NAN; period],
            buffer_source: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, source: f64) -> Option<(f64, f64, f64)> {
        let i = self.head;
        self.buffer_high[i] = high;
        self.buffer_low[i] = low;
        self.buffer_close[i] = close;
        self.buffer_source[i] = source;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 { self.filled = true; }
        if !self.filled { return None; }
        let mut tr = 0.0;
        let prev = (i + self.period - 1) % self.period;
        if self.head == 0 {
            for j in 0..self.period {
                let idx = (i + j) % self.period;
                let h = self.buffer_high[idx];
                let l = self.buffer_low[idx];
                let c = self.buffer_close[prev];
                tr += (h - l).max((h - c).abs()).max((l - c).abs());
            }
            self.atr_stream[i] = tr / (self.period as f64);
        } else {
            let h = self.buffer_high[i];
            let l = self.buffer_low[i];
            let c = self.buffer_close[prev];
            let tr_val = (h - l).max((h - c).abs()).max((l - c).abs());
            let prev_rma = self.atr_stream[prev];
            self.atr_stream[i] = prev_rma + (tr_val - prev_rma) / (self.period as f64);
        }
        // For the middle (MA), just simple moving average for this stub:
        let mean = self.buffer_source.iter().sum::<f64>() / (self.period as f64);
        self.ma_stream[i] = mean;
        let ma = self.ma_stream[i];
        let atr = self.atr_stream[i];
        Some((ma + self.multiplier * atr, ma, ma - self.multiplier * atr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_keltner_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = KeltnerParams {
            period: Some(20),
            multiplier: Some(2.0),
            ma_type: Some("ema".to_string()),
        };
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner_with_kernel(&input, kernel)?;

        assert_eq!(result.upper_band.len(), candles.close.len());
        assert_eq!(result.middle_band.len(), candles.close.len());
        assert_eq!(result.lower_band.len(), candles.close.len());

        let last_five_index = candles.close.len().saturating_sub(5);
        let expected_upper = [
            61619.504155205745,
            61503.56119134791,
            61387.47897150178,
            61286.61078267451,
            61206.25688331261,
        ];
        let expected_middle = [
            59758.339871629956,
            59703.35512195091,
            59640.083205574636,
            59593.884805043715,
            59504.46720456336,
        ];
        let expected_lower = [
            57897.17558805417,
            57903.14905255391,
            57892.68743964749,
            57901.158827412924,
            57802.67752581411,
        ];
        let last_five_upper = &result.upper_band[last_five_index..];
        let last_five_middle = &result.middle_band[last_five_index..];
        let last_five_lower = &result.lower_band[last_five_index..];
        for i in 0..5 {
            let diff_u = (last_five_upper[i] - expected_upper[i]).abs();
            let diff_m = (last_five_middle[i] - expected_middle[i]).abs();
            let diff_l = (last_five_lower[i] - expected_lower[i]).abs();
            assert!(diff_u < 1e-1, "Upper band mismatch at index {}: expected {}, got {}", i, expected_upper[i], last_five_upper[i]);
            assert!(diff_m < 1e-1, "Middle band mismatch at index {}: expected {}, got {}", i, expected_middle[i], last_five_middle[i]);
            assert!(diff_l < 1e-1, "Lower band mismatch at index {}: expected {}, got {}", i, expected_lower[i], last_five_lower[i]);
        }
        Ok(())
    }

    fn check_keltner_default_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = KeltnerParams::default();
        let input = KeltnerInput::from_candles(&candles, "close", default_params);
        let result = keltner_with_kernel(&input, kernel)?;
        assert_eq!(result.upper_band.len(), candles.close.len());
        assert_eq!(result.middle_band.len(), candles.close.len());
        assert_eq!(result.lower_band.len(), candles.close.len());
        Ok(())
    }

    fn check_keltner_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KeltnerParams { period: Some(0), multiplier: Some(2.0), ma_type: Some("ema".to_string()) };
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("invalid period"));
        }
        Ok(())
    }

    fn check_keltner_large_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KeltnerParams { period: Some(999999), multiplier: Some(2.0), ma_type: Some("ema".to_string()) };
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("invalid period"));
        }
        Ok(())
    }

    fn check_keltner_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KeltnerParams::default();
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner_with_kernel(&input, kernel)?;
        assert_eq!(result.middle_band.len(), candles.close.len());
        if result.middle_band.len() > 240 {
            for (i, &val) in result.middle_band[240..].iter().enumerate() {
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

    fn check_keltner_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 20;
        let multiplier = 2.0;

        let params = KeltnerParams {
            period: Some(period),
            multiplier: Some(multiplier),
            ma_type: Some("ema".to_string()),
        };
        let input = KeltnerInput::from_candles(&candles, "close", params.clone());
        let batch_output = keltner_with_kernel(&input, kernel)?;

        let mut stream = KeltnerStream::try_new(params)?;
        let mut upper_stream = Vec::with_capacity(candles.close.len());
        let mut middle_stream = Vec::with_capacity(candles.close.len());
        let mut lower_stream = Vec::with_capacity(candles.close.len());

        for i in 0..candles.close.len() {
            let hi = candles.high[i];
            let lo = candles.low[i];
            let cl = candles.close[i];
            let src = candles.close[i]; // using "close" as the MA source for streaming
            match stream.update(hi, lo, cl, src) {
                Some((up, mid, low)) => {
                    upper_stream.push(up);
                    middle_stream.push(mid);
                    lower_stream.push(low);
                }
                None => {
                    upper_stream.push(f64::NAN);
                    middle_stream.push(f64::NAN);
                    lower_stream.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch_output.upper_band.len(), upper_stream.len());
        assert_eq!(batch_output.middle_band.len(), middle_stream.len());
        assert_eq!(batch_output.lower_band.len(), lower_stream.len());
        for (i, (&b, &s)) in batch_output.middle_band.iter().zip(middle_stream.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Keltner streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = KeltnerBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = KeltnerParams::default();
        let (upper, middle, lower) = output.values_for(&def).expect("default row missing");

        assert_eq!(upper.len(), c.close.len());
        assert_eq!(middle.len(), c.close.len());
        assert_eq!(lower.len(), c.close.len());

        Ok(())
    }

    macro_rules! generate_all_keltner_tests {
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

    generate_all_keltner_tests!(
        check_keltner_accuracy,
        check_keltner_default_params,
        check_keltner_zero_period,
        check_keltner_large_period,
        check_keltner_nan_handling,
        check_keltner_streaming
    );

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

