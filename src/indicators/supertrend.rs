//! # SuperTrend Indicator
//!
//! Trend-following indicator using ATR-based dynamic bands. Computes support/resistance bands
//! and outputs the trend value and a change flag. SIMD/AVX stubs provided for API parity.
//!
//! ## Parameters
//! - **period**: ATR lookback window (default: 10)
//! - **factor**: ATR multiplier (default: 3.0)
//!
//! ## Errors
//! - **EmptyData**: All slices empty
//! - **InvalidPeriod**: period = 0 or period > data length
//! - **NotEnoughValidData**: Not enough valid (non-NaN) rows
//! - **AllValuesNaN**: No non-NaN row exists
//!
//! ## Returns
//! - **Ok(SuperTrendOutput)**: { trend, changed } both Vec<f64> of input len
//! - **Err(SuperTrendError)**

use crate::indicators::atr::{atr, AtrData, AtrError, AtrInput, AtrOutput, AtrParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;

#[derive(Debug, Clone)]
pub enum SuperTrendData<'a> {
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
pub struct SuperTrendParams {
    pub period: Option<usize>,
    pub factor: Option<f64>,
}
impl Default for SuperTrendParams {
    fn default() -> Self {
        Self {
            period: Some(10),
            factor: Some(3.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendInput<'a> {
    pub data: SuperTrendData<'a>,
    pub params: SuperTrendParams,
}

impl<'a> SuperTrendInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: SuperTrendParams) -> Self {
        Self {
            data: SuperTrendData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: SuperTrendParams,
    ) -> Self {
        Self {
            data: SuperTrendData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SuperTrendData::Candles { candles },
            params: SuperTrendParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(3.0)
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendOutput {
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct SuperTrendBuilder {
    period: Option<usize>,
    factor: Option<f64>,
    kernel: Kernel,
}
impl Default for SuperTrendBuilder {
    fn default() -> Self {
        Self {
            period: None,
            factor: None,
            kernel: Kernel::Auto,
        }
    }
}
impl SuperTrendBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline]
    pub fn factor(mut self, x: f64) -> Self {
        self.factor = Some(x);
        self
    }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn apply(self, c: &Candles) -> Result<SuperTrendOutput, SuperTrendError> {
        let p = SuperTrendParams {
            period: self.period,
            factor: self.factor,
        };
        let i = SuperTrendInput::from_candles(c, p);
        supertrend_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<SuperTrendOutput, SuperTrendError> {
        let p = SuperTrendParams {
            period: self.period,
            factor: self.factor,
        };
        let i = SuperTrendInput::from_slices(high, low, close, p);
        supertrend_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn into_stream(self) -> Result<SuperTrendStream, SuperTrendError> {
        let p = SuperTrendParams {
            period: self.period,
            factor: self.factor,
        };
        SuperTrendStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SuperTrendError {
    #[error("supertrend: Empty data provided.")]
    EmptyData,
    #[error("supertrend: All values are NaN.")]
    AllValuesNaN,
    #[error("supertrend: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("supertrend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(transparent)]
    AtrError(#[from] AtrError),
}

#[inline]
pub fn supertrend(input: &SuperTrendInput) -> Result<SuperTrendOutput, SuperTrendError> {
    supertrend_with_kernel(input, Kernel::Auto)
}

pub fn supertrend_with_kernel(
    input: &SuperTrendInput,
    kernel: Kernel,
) -> Result<SuperTrendOutput, SuperTrendError> {
    let (high, low, close) = match &input.data {
        SuperTrendData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        SuperTrendData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(SuperTrendError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(SuperTrendError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }
    let factor = input.get_factor();
    let len = high.len();
    let mut first_valid_idx = None;
    for i in 0..len {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SuperTrendError::AllValuesNaN),
    };
    if (len - first_valid_idx) < period {
        return Err(SuperTrendError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let atr_input = AtrInput::from_slices(
        &high[first_valid_idx..],
        &low[first_valid_idx..],
        &close[first_valid_idx..],
        AtrParams {
            length: Some(period),
        },
    );
    let AtrOutput { values: atr_values } = atr(&atr_input)?;

    let mut trend = vec![f64::NAN; len];
    let mut changed = vec![0.0; len];

    unsafe {
        match match kernel {
            Kernel::Auto => detect_best_kernel(),
            other => other,
        } {
            Kernel::Scalar | Kernel::ScalarBatch => {
                supertrend_scalar(
                    high,
                    low,
                    close,
                    period,
                    factor,
                    first_valid_idx,
                    &atr_values,
                    &mut trend,
                    &mut changed,
                );
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                supertrend_avx2(
                    high,
                    low,
                    close,
                    period,
                    factor,
                    first_valid_idx,
                    &atr_values,
                    &mut trend,
                    &mut changed,
                );
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                supertrend_avx512(
                    high,
                    low,
                    close,
                    period,
                    factor,
                    first_valid_idx,
                    &atr_values,
                    &mut trend,
                    &mut changed,
                );
            }
            _ => unreachable!(),
        }
    }
    Ok(SuperTrendOutput { trend, changed })
}

// Scalar core (reference logic)
#[inline(always)]
pub fn supertrend_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    let len = high.len();
    let mut upper_basic = vec![f64::NAN; len - first_valid_idx];
    let mut lower_basic = vec![f64::NAN; len - first_valid_idx];
    let mut upper_band = vec![f64::NAN; len - first_valid_idx];
    let mut lower_band = vec![f64::NAN; len - first_valid_idx];
    for i in 0..(len - first_valid_idx) {
        let half_range = (high[first_valid_idx + i] + low[first_valid_idx + i]) / 2.0;
        upper_basic[i] = half_range + factor * atr_values[i];
        lower_basic[i] = half_range - factor * atr_values[i];
        upper_band[i] = upper_basic[i];
        lower_band[i] = lower_basic[i];
    }
    for i in period..(len - first_valid_idx) {
        let prev_close = close[first_valid_idx + i - 1];
        let prev_upper_band = upper_band[i - 1];
        let prev_lower_band = lower_band[i - 1];
        let curr_upper_basic = upper_basic[i];
        let curr_lower_basic = lower_basic[i];

        if prev_close <= prev_upper_band {
            upper_band[i] = f64::min(curr_upper_basic, prev_upper_band);
        }
        if prev_close >= prev_lower_band {
            lower_band[i] = f64::max(curr_lower_basic, prev_lower_band);
        }
        if prev_close <= prev_upper_band {
            trend[first_valid_idx + i - 1] = prev_upper_band;
        } else {
            trend[first_valid_idx + i - 1] = prev_lower_band;
        }
    }
    for i in period..(len - first_valid_idx) {
        let prev_close = close[first_valid_idx + i - 1];
        let prev_upper_band = upper_band[i - 1];
        let curr_upper_band = upper_band[i];
        let prev_lower_band = lower_band[i - 1];
        let curr_lower_band = lower_band[i];
        let prev_trend = trend[first_valid_idx + i - 1];

        if (prev_trend - prev_upper_band).abs() < f64::EPSILON {
            if close[first_valid_idx + i] <= curr_upper_band {
                trend[first_valid_idx + i] = curr_upper_band;
                changed[first_valid_idx + i] = 0.0;
            } else {
                trend[first_valid_idx + i] = curr_lower_band;
                changed[first_valid_idx + i] = 1.0;
            }
        } else if (prev_trend - prev_lower_band).abs() < f64::EPSILON {
            if close[first_valid_idx + i] >= curr_lower_band {
                trend[first_valid_idx + i] = curr_lower_band;
                changed[first_valid_idx + i] = 0.0;
            } else {
                trend[first_valid_idx + i] = curr_upper_band;
                changed[first_valid_idx + i] = 1.0;
            }
        }
    }
}

// AVX2/AVX512 stubs with correct API
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supertrend_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high, low, close, period, factor, first_valid_idx, atr_values, trend, changed,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supertrend_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    if period <= 32 {
        supertrend_avx512_short(
            high, low, close, period, factor, first_valid_idx, atr_values, trend, changed,
        );
    } else {
        supertrend_avx512_long(
            high, low, close, period, factor, first_valid_idx, atr_values, trend, changed,
        );
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supertrend_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high, low, close, period, factor, first_valid_idx, atr_values, trend, changed,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supertrend_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high, low, close, period, factor, first_valid_idx, atr_values, trend, changed,
    );
}

// Streaming (stateful) implementation for parity
#[derive(Debug, Clone)]
pub struct SuperTrendStream {
    pub period: usize,
    pub factor: f64,
    atr_stream: crate::indicators::atr::AtrStream,
    buffer_high: Vec<f64>,
    buffer_low: Vec<f64>,
    buffer_close: Vec<f64>,
    head: usize,
    filled: bool,
    prev_upper_band: f64,
    prev_lower_band: f64,
    prev_trend: f64,
}
impl SuperTrendStream {
    pub fn try_new(params: SuperTrendParams) -> Result<Self, SuperTrendError> {
        let period = params.period.unwrap_or(10);
        let factor = params.factor.unwrap_or(3.0);
        let atr_stream = crate::indicators::atr::AtrStream::try_new(AtrParams { length: Some(period) })?;
        Ok(Self {
            period,
            factor,
            atr_stream,
            buffer_high: vec![f64::NAN; period],
            buffer_low: vec![f64::NAN; period],
            buffer_close: vec![f64::NAN; period],
            head: 0,
            filled: false,
            prev_upper_band: f64::NAN,
            prev_lower_band: f64::NAN,
            prev_trend: f64::NAN,
        })
    }
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.buffer_high[self.head] = high;
        self.buffer_low[self.head] = low;
        self.buffer_close[self.head] = close;
        self.head = (self.head + 1) % self.period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        let atr_opt = self.atr_stream.update(high, low, close);
        if !self.filled || atr_opt.is_none() {
            return None;
        }
        let idx = if self.head == 0 { self.period - 1 } else { self.head - 1 };
        let avg = (self.buffer_high[idx] + self.buffer_low[idx]) / 2.0;
        let atr = atr_opt.unwrap();
        let upper_basic = avg + self.factor * atr;
        let lower_basic = avg - self.factor * atr;

        let upper_band = if self.prev_upper_band.is_nan() {
            upper_basic
        } else if self.buffer_close[(self.head + self.period - 2) % self.period] <= self.prev_upper_band {
            f64::min(upper_basic, self.prev_upper_band)
        } else {
            upper_basic
        };
        let lower_band = if self.prev_lower_band.is_nan() {
            lower_basic
        } else if self.buffer_close[(self.head + self.period - 2) % self.period] >= self.prev_lower_band {
            f64::max(lower_basic, self.prev_lower_band)
        } else {
            lower_basic
        };
        let prev_trend = self.prev_trend;
        let mut trend = f64::NAN;
        let mut changed = 0.0;
        if prev_trend.is_nan() || (prev_trend - self.prev_upper_band).abs() < f64::EPSILON {
            if close <= upper_band {
                trend = upper_band;
                changed = 0.0;
            } else {
                trend = lower_band;
                changed = 1.0;
            }
        } else if (prev_trend - self.prev_lower_band).abs() < f64::EPSILON {
            if close >= lower_band {
                trend = lower_band;
                changed = 0.0;
            } else {
                trend = upper_band;
                changed = 1.0;
            }
        }
        self.prev_upper_band = upper_band;
        self.prev_lower_band = lower_band;
        self.prev_trend = trend;
        Some((trend, changed))
    }
}

// Batch range builder + batch output
#[derive(Clone, Debug)]
pub struct SuperTrendBatchRange {
    pub period: (usize, usize, usize),
    pub factor: (f64, f64, f64),
}
impl Default for SuperTrendBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 50, 1),
            factor: (3.0, 3.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SuperTrendBatchBuilder {
    range: SuperTrendBatchRange,
    kernel: Kernel,
}
impl SuperTrendBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.factor = (start, end, step);
        self
    }
    pub fn factor_static(mut self, x: f64) -> Self {
        self.range.factor = (x, x, 0.0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<SuperTrendBatchOutput, SuperTrendError> {
        supertrend_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
    ) -> Result<SuperTrendBatchOutput, SuperTrendError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
    pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<SuperTrendBatchOutput, SuperTrendError> {
        SuperTrendBatchBuilder::new().kernel(k).apply_candles(c)
    }
}

pub struct SuperTrendBatchOutput {
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
    pub combos: Vec<SuperTrendParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SuperTrendBatchOutput {
    pub fn row_for_params(&self, p: &SuperTrendParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(10) == p.period.unwrap_or(10)
                && (c.factor.unwrap_or(3.0) - p.factor.unwrap_or(3.0)).abs() < 1e-12
        })
    }
    pub fn trend_for(&self, p: &SuperTrendParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.trend[start..start + self.cols]
        })
    }
    pub fn changed_for(&self, p: &SuperTrendParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.changed[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SuperTrendBatchRange) -> Vec<SuperTrendParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let periods = axis_usize(r.period);
    let factors = axis_f64(r.factor);
    let mut out = Vec::with_capacity(periods.len() * factors.len());
    for &p in &periods {
        for &f in &factors {
            out.push(SuperTrendParams {
                period: Some(p),
                factor: Some(f),
            });
        }
    }
    out
}

pub fn supertrend_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendBatchRange,
    k: Kernel,
) -> Result<SuperTrendBatchOutput, SuperTrendError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(SuperTrendError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    supertrend_batch_par_slice(high, low, close, sweep, simd)
}

#[inline(always)]
pub fn supertrend_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendBatchRange,
    kern: Kernel,
) -> Result<SuperTrendBatchOutput, SuperTrendError> {
    supertrend_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn supertrend_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendBatchRange,
    kern: Kernel,
) -> Result<SuperTrendBatchOutput, SuperTrendError> {
    supertrend_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn supertrend_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SuperTrendBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SuperTrendBatchOutput, SuperTrendError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SuperTrendError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let len = high.len();
    let mut first_valid_idx = None;
    for i in 0..len {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SuperTrendError::AllValuesNaN),
    };
    let max_p = combos
        .iter()
        .map(|c| c.period.unwrap_or(10))
        .max()
        .unwrap();
    if len - first_valid_idx < max_p {
        return Err(SuperTrendError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid_idx,
        });
    }
    let rows = combos.len();
    let cols = len;
    let mut trend = vec![f64::NAN; rows * cols];
    let mut changed = vec![0.0; rows * cols];

    let do_row = |row: usize, trend_row: &mut [f64], changed_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let factor = combos[row].factor.unwrap();
        // Calculate ATR for this parameter combo
        let atr_input = AtrInput::from_slices(
            &high[first_valid_idx..],
            &low[first_valid_idx..],
            &close[first_valid_idx..],
            AtrParams { length: Some(period) },
        );
        let AtrOutput { values: atr_values } = atr(&atr_input).unwrap();
        match kern {
            Kernel::Scalar => supertrend_row_scalar(
                high,
                low,
                close,
                period,
                factor,
                first_valid_idx,
                &atr_values,
                trend_row,
                changed_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => supertrend_row_avx2(
                high,
                low,
                close,
                period,
                factor,
                first_valid_idx,
                &atr_values,
                trend_row,
                changed_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => supertrend_row_avx512(
                high,
                low,
                close,
                period,
                factor,
                first_valid_idx,
                &atr_values,
                trend_row,
                changed_row,
            ),
            _ => unreachable!(),
        }
    };
    if parallel {
        trend
            .par_chunks_mut(cols)
            .zip(changed.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (tr, ch))| do_row(row, tr, ch));
    } else {
        for (row, (tr, ch)) in trend.chunks_mut(cols).zip(changed.chunks_mut(cols)).enumerate() {
            do_row(row, tr, ch);
        }
    }
    Ok(SuperTrendBatchOutput {
        trend,
        changed,
        combos,
        rows,
        cols,
    })
}

// Scalar row for batch API
#[inline(always)]
unsafe fn supertrend_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

// AVX2/AVX512 row stubs for API
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn supertrend_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn supertrend_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    if period <= 32 {
        supertrend_row_avx512_short(
            high,
            low,
            close,
            period,
            factor,
            first_valid_idx,
            atr_values,
            trend,
            changed,
        );
    } else {
        supertrend_row_avx512_long(
            high,
            low,
            close,
            period,
            factor,
            first_valid_idx,
            atr_values,
            trend,
            changed,
        );
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn supertrend_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn supertrend_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    factor: f64,
    first_valid_idx: usize,
    atr_values: &[f64],
    trend: &mut [f64],
    changed: &mut [f64],
) {
    supertrend_scalar(
        high,
        low,
        close,
        period,
        factor,
        first_valid_idx,
        atr_values,
        trend,
        changed,
    );
}

#[inline(always)]
fn expand_grid_supertrend(r: &SuperTrendBatchRange) -> Vec<SuperTrendParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_supertrend_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = SuperTrendParams { period: None, factor: None };
        let input = SuperTrendInput::from_candles(&candles, default_params);
        let output = supertrend_with_kernel(&input, kernel)?;
        assert_eq!(output.trend.len(), candles.close.len());
        assert_eq!(output.changed.len(), candles.close.len());

        Ok(())
    }

    fn check_supertrend_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = SuperTrendParams { period: Some(10), factor: Some(3.0) };
        let input = SuperTrendInput::from_candles(&candles, params);
        let st_result = supertrend_with_kernel(&input, kernel)?;

        assert_eq!(st_result.trend.len(), candles.close.len());
        assert_eq!(st_result.changed.len(), candles.close.len());

        let expected_last_five_trend = [
            61811.479454208165,
            61721.73150878735,
            61459.10835790861,
            61351.59752211775,
            61033.18776990598,
        ];
        let expected_last_five_changed = [0.0, 0.0, 0.0, 0.0, 0.0];

        let start_index = st_result.trend.len() - 5;
        let trend_slice = &st_result.trend[start_index..];
        let changed_slice = &st_result.changed[start_index..];

        for (i, &val) in trend_slice.iter().enumerate() {
            let exp = expected_last_five_trend[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "[{}] Trend mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                exp
            );
        }
        for (i, &val) in changed_slice.iter().enumerate() {
            let exp = expected_last_five_changed[i];
            assert!(
                (val - exp).abs() < 1e-9,
                "[{}] Changed mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                exp
            );
        }
        Ok(())
    }

    fn check_supertrend_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = SuperTrendInput::with_default_candles(&candles);
        let output = supertrend_with_kernel(&input, kernel)?;
        assert_eq!(output.trend.len(), candles.close.len());
        assert_eq!(output.changed.len(), candles.close.len());
        Ok(())
    }

    fn check_supertrend_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 12.0, 13.0];
        let low = [9.0, 11.0, 12.5];
        let close = [9.5, 11.5, 13.0];
        let params = SuperTrendParams { period: Some(0), factor: Some(3.0) };
        let input = SuperTrendInput::from_slices(&high, &low, &close, params);
        let res = supertrend_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail with zero period", test_name);
        Ok(())
    }

    fn check_supertrend_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 12.0, 13.0];
        let low = [9.0, 11.0, 12.5];
        let close = [9.5, 11.5, 13.0];
        let params = SuperTrendParams { period: Some(10), factor: Some(3.0) };
        let input = SuperTrendInput::from_slices(&high, &low, &close, params);
        let res = supertrend_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail with period > data.len()", test_name);
        Ok(())
    }

    fn check_supertrend_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [40.0];
        let close = [41.0];
        let params = SuperTrendParams { period: Some(10), factor: Some(3.0) };
        let input = SuperTrendInput::from_slices(&high, &low, &close, params);
        let res = supertrend_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Should fail for data smaller than period", test_name);
        Ok(())
    }

    fn check_supertrend_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = SuperTrendParams { period: Some(10), factor: Some(3.0) };
        let first_input = SuperTrendInput::from_candles(&candles, first_params);
        let first_result = supertrend_with_kernel(&first_input, kernel)?;

        let second_params = SuperTrendParams { period: Some(5), factor: Some(2.0) };
        let second_input = SuperTrendInput::from_slices(
            &first_result.trend,
            &first_result.trend,
            &first_result.trend,
            second_params,
        );
        let second_result = supertrend_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.trend.len(), first_result.trend.len());
        assert_eq!(second_result.changed.len(), first_result.changed.len());
        Ok(())
    }

    fn check_supertrend_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = SuperTrendParams { period: Some(10), factor: Some(3.0) };
        let input = SuperTrendInput::from_candles(&candles, params);
        let result = supertrend_with_kernel(&input, kernel)?;
        if result.trend.len() > 50 {
            for (i, &val) in result.trend[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_supertrend_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 10;
        let factor = 3.0;
        let params = SuperTrendParams { period: Some(period), factor: Some(factor) };
        let input = SuperTrendInput::from_candles(&candles, params.clone());
        let batch_output = supertrend_with_kernel(&input, kernel)?;

        let mut stream = SuperTrendStream::try_new(params.clone())?;
        let mut stream_trend = Vec::with_capacity(candles.close.len());
        let mut stream_changed = Vec::with_capacity(candles.close.len());

        for i in 0..candles.close.len() {
            let (h, l, c) = (
                candles.high[i],
                candles.low[i],
                candles.close[i],
            );
            match stream.update(h, l, c) {
                Some((trend, changed)) => {
                    stream_trend.push(trend);
                    stream_changed.push(changed);
                }
                None => {
                    stream_trend.push(f64::NAN);
                    stream_changed.push(0.0);
                }
            }
        }
        assert_eq!(batch_output.trend.len(), stream_trend.len());
        assert_eq!(batch_output.changed.len(), stream_changed.len());

        for (i, (&b, &s)) in batch_output.trend.iter().zip(stream_trend.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Streaming trend mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in batch_output.changed.iter().zip(stream_changed.iter()).enumerate() {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Streaming changed mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_supertrend_tests {
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

    generate_all_supertrend_tests!(
        check_supertrend_partial_params,
        check_supertrend_accuracy,
        check_supertrend_default_candles,
        check_supertrend_zero_period,
        check_supertrend_period_exceeds_length,
        check_supertrend_very_small_dataset,
        check_supertrend_reinput,
        check_supertrend_nan_handling,
        check_supertrend_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = SuperTrendBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        let def = SuperTrendParams::default();
        let row = output.trend_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Last few values of trend for reference.
        let expected = [
            61811.479454208165,
            61721.73150878735,
            61459.10835790861,
            61351.59752211775,
            61033.18776990598,
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
