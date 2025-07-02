//! # Empirical Mode Decomposition (EMD)
//!
//! Implements the Empirical Mode Decomposition indicator with band-pass filtering and moving averages,
//! yielding upperband, middleband, and lowerband outputs.
//!
//! ## Parameters
//! - **period**: Window for the band-pass filter (default: 20)
//! - **delta**: Band-pass phase parameter (default: 0.5)
//! - **fraction**: Peak/valley scaling factor (default: 0.1)
//!
//! ## Errors
//! - **AllValuesNaN**: emd: All input data values are `NaN`.
//! - **InvalidPeriod**: emd: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: emd: Not enough valid data points for requested `period`.
//! - **InvalidDelta**: emd: `delta` is `NaN` or infinite.
//! - **InvalidFraction**: emd: `fraction` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(EmdOutput)`** on success, containing upperband/middleband/lowerband as `Vec<f64>`.
//! - **`Err(EmdError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EmdInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EmdData::Candles { candles } => source_type(candles, "close"),
            EmdData::Slices { close, .. } => close,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EmdData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct EmdOutput {
    pub upperband: Vec<f64>,
    pub middleband: Vec<f64>,
    pub lowerband: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EmdParams {
    pub period: Option<usize>,
    pub delta: Option<f64>,
    pub fraction: Option<f64>,
}

impl Default for EmdParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            delta: Some(0.5),
            fraction: Some(0.1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmdInput<'a> {
    pub data: EmdData<'a>,
    pub params: EmdParams,
}

impl<'a> EmdInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: EmdParams) -> Self {
        Self {
            data: EmdData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: EmdParams,
    ) -> Self {
        Self {
            data: EmdData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, EmdParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_delta(&self) -> f64 {
        self.params.delta.unwrap_or(0.5)
    }
    #[inline]
    pub fn get_fraction(&self) -> f64 {
        self.params.fraction.unwrap_or(0.1)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EmdBuilder {
    period: Option<usize>,
    delta: Option<f64>,
    fraction: Option<f64>,
    kernel: Kernel,
}

impl Default for EmdBuilder {
    fn default() -> Self {
        Self {
            period: None,
            delta: None,
            fraction: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EmdBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn delta(mut self, d: f64) -> Self {
        self.delta = Some(d);
        self
    }
    #[inline(always)]
    pub fn fraction(mut self, f: f64) -> Self {
        self.fraction = Some(f);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EmdOutput, EmdError> {
        let p = EmdParams {
            period: self.period,
            delta: self.delta,
            fraction: self.fraction,
        };
        let i = EmdInput::from_candles(c, p);
        emd_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmdOutput, EmdError> {
        let p = EmdParams {
            period: self.period,
            delta: self.delta,
            fraction: self.fraction,
        };
        let i = EmdInput::from_slices(high, low, close, volume, p);
        emd_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EmdStream, EmdError> {
        let p = EmdParams {
            period: self.period,
            delta: self.delta,
            fraction: self.fraction,
        };
        EmdStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum EmdError {
    #[error("emd: All values are NaN.")]
    AllValuesNaN,

    #[error("emd: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("emd: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("emd: Invalid delta: {delta}")]
    InvalidDelta { delta: f64 },

    #[error("emd: Invalid fraction: {fraction}")]
    InvalidFraction { fraction: f64 },
}

#[inline]
pub fn emd(input: &EmdInput) -> Result<EmdOutput, EmdError> {
    emd_with_kernel(input, Kernel::Auto)
}

pub fn emd_with_kernel(input: &EmdInput, kernel: Kernel) -> Result<EmdOutput, EmdError> {
    let (high, low) = match &input.data {
        EmdData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            (high, low)
        }
        EmdData::Slices { high, low, .. } => (*high, *low),
    };

    let len = high.len();
    let period = input.get_period();
    let delta = input.get_delta();
    let fraction = input.get_fraction();

    let first = (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(EmdError::AllValuesNaN)?;

    if period == 0 || period > len {
        return Err(EmdError::InvalidPeriod { period, data_len: len });
    }
    let needed = (2 * period).max(50);
    if (len - first) < needed {
        return Err(EmdError::NotEnoughValidData { needed, valid: len - first });
    }
    if delta.is_nan() || delta.is_infinite() {
        return Err(EmdError::InvalidDelta { delta });
    }
    if fraction.is_nan() || fraction.is_infinite() {
        return Err(EmdError::InvalidFraction { fraction });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                emd_scalar(high, low, period, delta, fraction, first, len)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                emd_avx2(high, low, period, delta, fraction, first, len)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                emd_avx512(high, low, period, delta, fraction, first, len)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn emd_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    let mut upperband = vec![f64::NAN; len];
    let mut middleband = vec![f64::NAN; len];
    let mut lowerband = vec![f64::NAN; len];

    let beta = (2.0 * std::f64::consts::PI / period as f64).cos();
    let gamma = 1.0 / ((4.0 * std::f64::consts::PI * delta / period as f64).cos());
    let alpha = gamma - (gamma * gamma - 1.0).sqrt();
    let half_one_minus_alpha = 0.5 * (1.0 - alpha);

    let per_up_low = 50;
    let per_mid = 2 * period;

    let mut sum_up = 0.0;
    let mut sum_mb = 0.0;
    let mut sum_low = 0.0;
    let mut sp_ring = vec![0.0; per_up_low];
    let mut sv_ring = vec![0.0; per_up_low];
    let mut bp_ring = vec![0.0; per_mid];
    let mut idx_up_low = 0_usize;
    let mut idx_mid = 0_usize;

    let mut bp_prev1 = 0.0;
    let mut bp_prev2 = 0.0;
    let mut peak_prev = 0.0;
    let mut valley_prev = 0.0;
    let mut initialized = false;
    let up_low_sub = per_up_low - 1;
    let mid_sub = per_mid - 1;

    for i in 0..len {
        if i < first {
            continue;
        }
        let price = (high[i] + low[i]) * 0.5;
        if !initialized {
            bp_prev1 = price;
            bp_prev2 = price;
            peak_prev = price;
            valley_prev = price;
            initialized = true;
        }
        let bp_curr = if i >= first + 2 {
            let price_i2 = (high[i - 2] + low[i - 2]) * 0.5;
            half_one_minus_alpha * (price - price_i2) + beta * (1.0 + alpha) * bp_prev1
                - alpha * bp_prev2
        } else {
            price
        };
        let mut peak_curr = peak_prev;
        let mut valley_curr = valley_prev;
        if i >= first + 2 {
            if bp_prev1 > bp_curr && bp_prev1 > bp_prev2 {
                peak_curr = bp_prev1;
            }
            if bp_prev1 < bp_curr && bp_prev1 < bp_prev2 {
                valley_curr = bp_prev1;
            }
        }
        let sp = peak_curr * fraction;
        let sv = valley_curr * fraction;
        sum_up += sp;
        sum_low += sv;
        sum_mb += bp_curr;
        let old_sp = sp_ring[idx_up_low];
        let old_sv = sv_ring[idx_up_low];
        let old_bp = bp_ring[idx_mid];
        sp_ring[idx_up_low] = sp;
        sv_ring[idx_up_low] = sv;
        bp_ring[idx_mid] = bp_curr;
        if i >= first + per_up_low {
            sum_up -= old_sp;
            sum_low -= old_sv;
        }
        if i >= first + per_mid {
            sum_mb -= old_bp;
        }
        idx_up_low = (idx_up_low + 1) % per_up_low;
        idx_mid = (idx_mid + 1) % per_mid;
        if i >= first + up_low_sub {
            upperband[i] = sum_up / per_up_low as f64;
            lowerband[i] = sum_low / per_up_low as f64;
        }
        if i >= first + mid_sub {
            middleband[i] = sum_mb / per_mid as f64;
        }
        bp_prev2 = bp_prev1;
        bp_prev1 = bp_curr;
        peak_prev = peak_curr;
        valley_prev = valley_curr;
    }

    Ok(EmdOutput {
        upperband,
        middleband,
        lowerband,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    emd_scalar(high, low, period, delta, fraction, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    if period <= 32 {
        emd_avx512_short(high, low, period, delta, fraction, first, len)
    } else {
        emd_avx512_long(high, low, period, delta, fraction, first, len)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    emd_scalar(high, low, period, delta, fraction, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    emd_scalar(high, low, period, delta, fraction, first, len)
}

#[inline(always)]
pub fn emd_row_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_scalar(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_avx2(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_avx512(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_avx512_short(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    delta: f64,
    fraction: f64,
    first: usize,
    len: usize,
) -> Result<EmdOutput, EmdError> {
    unsafe { emd_avx512_long(high, low, period, delta, fraction, first, len) }
}

#[derive(Debug, Clone)]
pub struct EmdStream {
    period: usize,
    delta: f64,
    fraction: f64,
    per_up_low: usize,
    per_mid: usize,
    sum_up: f64,
    sum_low: f64,
    sum_mb: f64,
    sp_ring: Vec<f64>,
    sv_ring: Vec<f64>,
    bp_ring: Vec<f64>,
    idx_up_low: usize,
    idx_mid: usize,
    bp_prev1: f64,
    bp_prev2: f64,
    peak_prev: f64,
    valley_prev: f64,
    initialized: bool,
    count: usize,
}

impl EmdStream {
    pub fn try_new(params: EmdParams) -> Result<Self, EmdError> {
        let period = params.period.unwrap_or(20);
        let delta = params.delta.unwrap_or(0.5);
        let fraction = params.fraction.unwrap_or(0.1);

        if period == 0 {
            return Err(EmdError::InvalidPeriod { period, data_len: 0 });
        }
        if delta.is_nan() || delta.is_infinite() {
            return Err(EmdError::InvalidDelta { delta });
        }
        if fraction.is_nan() || fraction.is_infinite() {
            return Err(EmdError::InvalidFraction { fraction });
        }

        Ok(Self {
            period,
            delta,
            fraction,
            per_up_low: 50,
            per_mid: 2 * period,
            sum_up: 0.0,
            sum_low: 0.0,
            sum_mb: 0.0,
            sp_ring: vec![0.0; 50],
            sv_ring: vec![0.0; 50],
            bp_ring: vec![0.0; 2 * period],
            idx_up_low: 0,
            idx_mid: 0,
            bp_prev1: 0.0,
            bp_prev2: 0.0,
            peak_prev: 0.0,
            valley_prev: 0.0,
            initialized: false,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
        let price = (high + low) * 0.5;
        let beta = (2.0 * std::f64::consts::PI / self.period as f64).cos();
        let gamma = 1.0 / ((4.0 * std::f64::consts::PI * self.delta / self.period as f64).cos());
        let alpha = gamma - (gamma * gamma - 1.0).sqrt();
        let half_one_minus_alpha = 0.5 * (1.0 - alpha);

        if !self.initialized {
            self.bp_prev1 = price;
            self.bp_prev2 = price;
            self.peak_prev = price;
            self.valley_prev = price;
            self.initialized = true;
        }
        let bp_curr = if self.count >= 2 {
            let price_i2 = price;
            half_one_minus_alpha * (price - price_i2)
                + beta * (1.0 + alpha) * self.bp_prev1
                - alpha * self.bp_prev2
        } else {
            price
        };
        let mut peak_curr = self.peak_prev;
        let mut valley_curr = self.valley_prev;
        if self.count >= 2 {
            if self.bp_prev1 > bp_curr && self.bp_prev1 > self.bp_prev2 {
                peak_curr = self.bp_prev1;
            }
            if self.bp_prev1 < bp_curr && self.bp_prev1 < self.bp_prev2 {
                valley_curr = self.bp_prev1;
            }
        }
        let sp = peak_curr * self.fraction;
        let sv = valley_curr * self.fraction;
        self.sum_up += sp;
        self.sum_low += sv;
        self.sum_mb += bp_curr;
        let old_sp = self.sp_ring[self.idx_up_low];
        let old_sv = self.sv_ring[self.idx_up_low];
        let old_bp = self.bp_ring[self.idx_mid];
        self.sp_ring[self.idx_up_low] = sp;
        self.sv_ring[self.idx_up_low] = sv;
        self.bp_ring[self.idx_mid] = bp_curr;
        if self.count >= self.per_up_low {
            self.sum_up -= old_sp;
            self.sum_low -= old_sv;
        }
        if self.count >= self.per_mid {
            self.sum_mb -= old_bp;
        }
        self.idx_up_low = (self.idx_up_low + 1) % self.per_up_low;
        self.idx_mid = (self.idx_mid + 1) % self.per_mid;
        let mut ub = None;
        let mut lb = None;
        let mut mb = None;
        if self.count + 1 >= self.per_up_low {
            ub = Some(self.sum_up / self.per_up_low as f64);
            lb = Some(self.sum_low / self.per_up_low as f64);
        }
        if self.count + 1 >= self.per_mid {
            mb = Some(self.sum_mb / self.per_mid as f64);
        }
        self.bp_prev2 = self.bp_prev1;
        self.bp_prev1 = bp_curr;
        self.peak_prev = peak_curr;
        self.valley_prev = valley_curr;
        self.count += 1;
        (ub, mb, lb)
    }
}

// Batch struct/logic
#[derive(Clone, Debug)]
pub struct EmdBatchRange {
    pub period: (usize, usize, usize),
    pub delta: (f64, f64, f64),
    pub fraction: (f64, f64, f64),
}

impl Default for EmdBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 20, 0),
            delta: (0.5, 0.5, 0.0),
            fraction: (0.1, 0.1, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmdBatchBuilder {
    range: EmdBatchRange,
    kernel: Kernel,
}

impl EmdBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
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
    pub fn delta_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.delta = (start, end, step);
        self
    }
    #[inline]
    pub fn delta_static(mut self, x: f64) -> Self {
        self.range.delta = (x, x, 0.0);
        self
    }
    #[inline]
    pub fn fraction_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.fraction = (start, end, step);
        self
    }
    #[inline]
    pub fn fraction_static(mut self, x: f64) -> Self {
        self.range.fraction = (x, x, 0.0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmdBatchOutput, EmdError> {
        emd_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<EmdBatchOutput, EmdError> {
        EmdBatchBuilder::new().kernel(k).apply_slices(high, low, close, volume)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<EmdBatchOutput, EmdError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        let volume = source_type(c, "volume");
        self.apply_slices(high, low, close, volume)
    }
}

pub fn emd_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &EmdBatchRange,
    k: Kernel,
) -> Result<EmdBatchOutput, EmdError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EmdError::InvalidPeriod { period: 0, data_len: 0 })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    emd_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EmdBatchOutput {
    pub upperband: Vec<f64>,
    pub middleband: Vec<f64>,
    pub lowerband: Vec<f64>,
    pub combos: Vec<EmdParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn expand_grid(r: &EmdBatchRange) -> Vec<EmdParams> {
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
    let deltas = axis_f64(r.delta);
    let fractions = axis_f64(r.fraction);

    let mut out = Vec::with_capacity(periods.len() * deltas.len() * fractions.len());
    for &p in &periods {
        for &d in &deltas {
            for &f in &fractions {
                out.push(EmdParams {
                    period: Some(p),
                    delta: Some(d),
                    fraction: Some(f),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn emd_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &EmdBatchRange,
    kern: Kernel,
) -> Result<EmdBatchOutput, EmdError> {
    emd_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn emd_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &EmdBatchRange,
    kern: Kernel,
) -> Result<EmdBatchOutput, EmdError> {
    emd_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn emd_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &EmdBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EmdBatchOutput, EmdError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EmdError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let len = high.len();
    let first = (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(EmdError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = (2 * max_p).max(50);
    if len - first < needed {
        return Err(EmdError::NotEnoughValidData { needed, valid: len - first });
    }

    let rows = combos.len();
    let cols = len;

    let mut upperband = vec![f64::NAN; rows * cols];
    let mut middleband = vec![f64::NAN; rows * cols];
    let mut lowerband = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, ub: &mut [f64], mb: &mut [f64], lb: &mut [f64]| {
        let prm = &combos[row];
        let period = prm.period.unwrap();
        let delta = prm.delta.unwrap();
        let fraction = prm.fraction.unwrap();
        let out = unsafe { emd_row_scalar(high, low, period, delta, fraction, first, cols) }
            .expect("emd row computation failed");
        ub.copy_from_slice(&out.upperband);
        mb.copy_from_slice(&out.middleband);
        lb.copy_from_slice(&out.lowerband);
    };
        if parallel {
            upperband
                .par_chunks_mut(cols)
                .zip(middleband.par_chunks_mut(cols))
                .zip(lowerband.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, ((ub, mb), lb))| {
                    do_row(row, ub, mb, lb);
                });
        } else {
            for row in 0..rows {
                let ub = &mut upperband[row * cols..(row + 1) * cols];
                let mb = &mut middleband[row * cols..(row + 1) * cols];
                let lb = &mut lowerband[row * cols..(row + 1) * cols];
                do_row(row, ub, mb, lb);
            }
        }


    Ok(EmdBatchOutput {
        upperband,
        middleband,
        lowerband,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn expand_grid_for_emdbatch(r: &EmdBatchRange) -> Vec<EmdParams> {
    expand_grid(r)
}

// API parity: this is required for batch indicator discovery/row mapping
impl EmdBatchOutput {
    pub fn row_for_params(&self, p: &EmdParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.delta.unwrap_or(0.5) - p.delta.unwrap_or(0.5)).abs() < 1e-12
                && (c.fraction.unwrap_or(0.1) - p.fraction.unwrap_or(0.1)).abs() < 1e-12
        })
    }
    pub fn bands_for(&self, p: &EmdParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.upperband[start..start + self.cols],
                &self.middleband[start..start + self.cols],
                &self.lowerband[start..start + self.cols],
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_emd_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = EmdParams::default();
        let input = EmdInput::from_candles(&candles, params);
        let emd_result = emd_with_kernel(&input, kernel)?;

        let expected_last_five_upper = [
            50.33760237677157,
            50.28850695686447,
            50.23941153695737,
            50.19031611705027,
            48.709744457737344,
        ];
        let expected_last_five_middle = [
            -368.71064280396706,
            -399.11033986231377,
            -421.9368852621732,
            -437.879217150269,
            -447.3257167904511,
        ];
        let expected_last_five_lower = [
            -60.67834136221248,
            -60.93110347122829,
            -61.68154077026321,
            -62.43197806929814,
            -63.18241536833306,
        ];

        let len = candles.close.len();
        let start_idx = len - 5;
        let actual_ub = &emd_result.upperband[start_idx..];
        let actual_mb = &emd_result.middleband[start_idx..];
        let actual_lb = &emd_result.lowerband[start_idx..];
        for i in 0..5 {
            assert!(
                (actual_ub[i] - expected_last_five_upper[i]).abs() < 1e-6,
                "Upperband mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_upper[i],
                actual_ub[i]
            );
            assert!(
                (actual_mb[i] - expected_last_five_middle[i]).abs() < 1e-6,
                "Middleband mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_middle[i],
                actual_mb[i]
            );
            assert!(
                (actual_lb[i] - expected_last_five_lower[i]).abs() < 1e-6,
                "Lowerband mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_lower[i],
                actual_lb[i]
            );
        }
        Ok(())
    }

    fn check_emd_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty_data: [f64; 0] = [];
        let params = EmdParams::default();
        let input =
            EmdInput::from_slices(&empty_data, &empty_data, &empty_data, &empty_data, params);
        let result = emd_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error on empty data");
        Ok(())
    }

    fn check_emd_all_nans(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let params = EmdParams::default();
        let input = EmdInput::from_slices(&data, &data, &data, &data, params);
        let result = emd_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for all-NaN data");
        Ok(())
    }

    fn check_emd_invalid_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = EmdParams {
            period: Some(0),
            ..Default::default()
        };
        let input = EmdInput::from_slices(&data, &data, &data, &data, params);
        let result = emd_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for zero period");
        Ok(())
    }

    fn check_emd_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![10.0; 10];
        let params = EmdParams {
            period: Some(20),
            ..Default::default()
        };
        let input = EmdInput::from_slices(&data, &data, &data, &data, params);
        let result = emd_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for not enough valid data");
        Ok(())
    }

    fn check_emd_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EmdInput::with_default_candles(&candles);
        let result = emd_with_kernel(&input, kernel);
        assert!(
            result.is_ok(),
            "Expected EMD to succeed with default params"
        );
        Ok(())
    }

    macro_rules! generate_all_emd_tests {
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

    generate_all_emd_tests!(
        check_emd_accuracy,
        check_emd_empty_data,
        check_emd_all_nans,
        check_emd_invalid_period,
        check_emd_not_enough_valid_data,
        check_emd_default_candles
    );
    #[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EmdBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        let def = EmdParams::default();
        let (ub, mb, lb) = output.bands_for(&def).expect("default row missing");

        assert_eq!(ub.len(), c.close.len(), "Upperband length mismatch");
        assert_eq!(mb.len(), c.close.len(), "Middleband length mismatch");
        assert_eq!(lb.len(), c.close.len(), "Lowerband length mismatch");

        // Spot check last values vs. single-batch computation (if desired, could hardcode here)
        let expected_last_five_upper = [
            50.33760237677157,
            50.28850695686447,
            50.23941153695737,
            50.19031611705027,
            48.709744457737344,
        ];
        let expected_last_five_middle = [
            -368.71064280396706,
            -399.11033986231377,
            -421.9368852621732,
            -437.879217150269,
            -447.3257167904511,
        ];
        let expected_last_five_lower = [
            -60.67834136221248,
            -60.93110347122829,
            -61.68154077026321,
            -62.43197806929814,
            -63.18241536833306,
        ];
        let len = ub.len();
        for i in 0..5 {
            assert!(
                (ub[len - 5 + i] - expected_last_five_upper[i]).abs() < 1e-6,
                "[{test}] upperband mismatch idx {i}: {} vs {}",
                ub[len - 5 + i], expected_last_five_upper[i]
            );
            assert!(
                (mb[len - 5 + i] - expected_last_five_middle[i]).abs() < 1e-6,
                "[{test}] middleband mismatch idx {i}: {} vs {}",
                mb[len - 5 + i], expected_last_five_middle[i]
            );
            assert!(
                (lb[len - 5 + i] - expected_last_five_lower[i]).abs() < 1e-6,
                "[{test}] lowerband mismatch idx {i}: {} vs {}",
                lb[len - 5 + i], expected_last_five_lower[i]
            );
        }

        Ok(())
    }

    fn check_batch_param_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Sweep period over 20 and 22, delta over 0.5 and 0.6, fraction over 0.1 and 0.2
        let output = EmdBatchBuilder::new()
            .kernel(kernel)
            .period_range(20, 22, 2)
            .delta_range(0.5, 0.6, 0.1)
            .fraction_range(0.1, 0.2, 0.1)
            .apply_candles(&c)?;

        assert!(
            output.rows == 8,
            "Expected 8 rows (2*2*2 grid), got {}",
            output.rows
        );
        assert_eq!(output.cols, c.close.len());

        // Verify that bands_for returns correct shapes for all combos
        for params in &output.combos {
            let (ub, mb, lb) = output
                .bands_for(params)
                .expect("row for params missing in sweep");
            assert_eq!(ub.len(), output.cols);
            assert_eq!(mb.len(), output.cols);
            assert_eq!(lb.len(), output.cols);
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
    gen_batch_tests!(check_batch_param_sweep);
}

}
