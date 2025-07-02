//! # Dynamic Trend Index (DTI) by William Blau
//!
//! A momentum-based indicator that computes the difference between upward and downward
//! price movements, then applies a triple EMA smoothing to that difference (and its
//! absolute value), producing a value typically scaled between `-100` and `100`.
//!
//! ## Parameters
//! - **r**: The period of the first EMA smoothing. Defaults to 14.
//! - **s**: The period of the second EMA smoothing. Defaults to 10.
//! - **u**: The period of the third EMA smoothing. Defaults to 5.
//!
//! ## Errors
//! - **EmptyData**: dti: Input data slice is empty.
//! - **CandleFieldError**: dti: Error reading candle data fields.
//! - **InvalidPeriod**: dti: One or more of the EMA periods is zero or exceeds the data length.
//! - **NotEnoughValidData**: dti: Fewer valid (non-`NaN`) data points remain after the first
//!   valid index than are needed to compute at least one of the EMAs.
//! - **AllValuesNaN**: dti: All input high/low values are `NaN`.
//!
//! ## Returns
//! - **`Ok(DtiOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the indicator can be fully calculated.
//! - **`Err(DtiError)`** otherwise.

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

#[derive(Debug, Clone)]
pub enum DtiData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct DtiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DtiParams {
    pub r: Option<usize>,
    pub s: Option<usize>,
    pub u: Option<usize>,
}

impl Default for DtiParams {
    fn default() -> Self {
        Self {
            r: Some(14),
            s: Some(10),
            u: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DtiInput<'a> {
    pub data: DtiData<'a>,
    pub params: DtiParams,
}

impl<'a> DtiInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DtiParams) -> Self {
        Self {
            data: DtiData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DtiParams) -> Self {
        Self {
            data: DtiData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DtiParams::default())
    }

    #[inline]
    pub fn get_r(&self) -> usize {
        self.params.r.unwrap_or(14)
    }
    #[inline]
    pub fn get_s(&self) -> usize {
        self.params.s.unwrap_or(10)
    }
    #[inline]
    pub fn get_u(&self) -> usize {
        self.params.u.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DtiBuilder {
    r: Option<usize>,
    s: Option<usize>,
    u: Option<usize>,
    kernel: Kernel,
}

impl Default for DtiBuilder {
    fn default() -> Self {
        Self {
            r: None,
            s: None,
            u: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DtiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn r(mut self, n: usize) -> Self {
        self.r = Some(n);
        self
    }
    #[inline(always)]
    pub fn s(mut self, n: usize) -> Self {
        self.s = Some(n);
        self
    }
    #[inline(always)]
    pub fn u(mut self, n: usize) -> Self {
        self.u = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<DtiOutput, DtiError> {
        let p = DtiParams {
            r: self.r,
            s: self.s,
            u: self.u,
        };
        let i = DtiInput::from_candles(c, p);
        dti_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DtiOutput, DtiError> {
        let p = DtiParams {
            r: self.r,
            s: self.s,
            u: self.u,
        };
        let i = DtiInput::from_slices(high, low, p);
        dti_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<DtiStream, DtiError> {
        let p = DtiParams {
            r: self.r,
            s: self.s,
            u: self.u,
        };
        DtiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DtiError {
    #[error("dti: Empty data provided.")]
    EmptyData,
    #[error("dti: Candle field error: {0}")]
    CandleFieldError(String),
    #[error("dti: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dti: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dti: All high/low values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn dti(input: &DtiInput) -> Result<DtiOutput, DtiError> {
    dti_with_kernel(input, Kernel::Auto)
}

pub fn dti_with_kernel(input: &DtiInput, kernel: Kernel) -> Result<DtiOutput, DtiError> {
    let (high, low) = match &input.data {
        DtiData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
            (high, low)
        }
        DtiData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(DtiError::EmptyData);
    }
    let len = high.len();
    if low.len() != len {
        return Err(DtiError::EmptyData);
    }

    let first_valid_idx = match (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(DtiError::AllValuesNaN),
    };

    let r = input.get_r();
    let s = input.get_s();
    let u = input.get_u();

    for &period in &[r, s, u] {
        if period == 0 || period > len {
            return Err(DtiError::InvalidPeriod {
                period,
                data_len: len,
            });
        }
        if (len - first_valid_idx) < period {
            return Err(DtiError::NotEnoughValidData {
                needed: period,
                valid: len - first_valid_idx,
            });
        }
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                dti_scalar(high, low, r, s, u, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                dti_avx2(high, low, r, s, u, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                dti_avx512(high, low, r, s, u, first_valid_idx, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(DtiOutput { values: out })
}

#[inline]
pub fn dti_scalar(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    let len = high.len();
    let alpha_r = 2.0 / (r as f64 + 1.0);
    let alpha_s = 2.0 / (s as f64 + 1.0);
    let alpha_u = 2.0 / (u as f64 + 1.0);

    let alpha_r_1 = 1.0 - alpha_r;
    let alpha_s_1 = 1.0 - alpha_s;
    let alpha_u_1 = 1.0 - alpha_u;

    let mut e0_r = 0.0;
    let mut e0_s = 0.0;
    let mut e0_u = 0.0;
    let mut e1_r = 0.0;
    let mut e1_s = 0.0;
    let mut e1_u = 0.0;

    out[first_valid_idx] = f64::NAN;
    for i in (first_valid_idx + 1)..len {
        let dh = high[i] - high[i - 1];
        let dl = low[i] - low[i - 1];
        let x_hmu = if dh > 0.0 { dh } else { 0.0 };
        let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
        let x_price = x_hmu - x_lmd;
        let x_price_abs = x_price.abs();

        e0_r = alpha_r * x_price + alpha_r_1 * e0_r;
        e0_s = alpha_s * e0_r + alpha_s_1 * e0_s;
        e0_u = alpha_u * e0_s + alpha_u_1 * e0_u;

        e1_r = alpha_r * x_price_abs + alpha_r_1 * e1_r;
        e1_s = alpha_s * e1_r + alpha_s_1 * e1_s;
        e1_u = alpha_u * e1_s + alpha_u_1 * e1_u;

        if !e1_u.is_nan() && e1_u != 0.0 {
            out[i] = 100.0 * e0_u / e1_u;
        
            } else {
            out[i] = 0.0;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dti_avx2(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_scalar(high, low, r, s, u, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dti_avx512(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    if r.max(s).max(u) <= 32 {
        unsafe { dti_avx512_short(high, low, r, s, u, first_valid_idx, out) }
    } else {
        unsafe { dti_avx512_long(high, low, r, s, u, first_valid_idx, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dti_avx512_short(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_scalar(high, low, r, s, u, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dti_avx512_long(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_scalar(high, low, r, s, u, first_valid_idx, out)
}

#[inline(always)]
pub fn dti_row_scalar(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_scalar(high, low, r, s, u, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn dti_row_avx2(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_avx2(high, low, r, s, u, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn dti_row_avx512(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_avx512(high, low, r, s, u, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn dti_row_avx512_short(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    unsafe { dti_avx512_short(high, low, r, s, u, first_valid_idx, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn dti_row_avx512_long(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    unsafe { dti_avx512_long(high, low, r, s, u, first_valid_idx, out) }
}

#[derive(Debug, Clone)]
pub struct DtiStream {
    r: usize,
    s: usize,
    u: usize,
    alpha_r: f64,
    alpha_s: f64,
    alpha_u: f64,
    alpha_r_1: f64,
    alpha_s_1: f64,
    alpha_u_1: f64,
    e0_r: f64,
    e0_s: f64,
    e0_u: f64,
    e1_r: f64,
    e1_s: f64,
    e1_u: f64,
    last_high: Option<f64>,
    last_low: Option<f64>,
    initialized: bool,
}

impl DtiStream {
    pub fn try_new(params: DtiParams) -> Result<Self, DtiError> {
        let r = params.r.unwrap_or(14);
        let s = params.s.unwrap_or(10);
        let u = params.u.unwrap_or(5);
        if r == 0 || s == 0 || u == 0 {
            return Err(DtiError::InvalidPeriod { period: 0, data_len: 0 });
        }
        let alpha_r = 2.0 / (r as f64 + 1.0);
        let alpha_s = 2.0 / (s as f64 + 1.0);
        let alpha_u = 2.0 / (u as f64 + 1.0);
        let alpha_r_1 = 1.0 - alpha_r;
        let alpha_s_1 = 1.0 - alpha_s;
        let alpha_u_1 = 1.0 - alpha_u;
        Ok(Self {
            r,
            s,
            u,
            alpha_r,
            alpha_s,
            alpha_u,
            alpha_r_1,
            alpha_s_1,
            alpha_u_1,
            e0_r: 0.0,
            e0_s: 0.0,
            e0_u: 0.0,
            e1_r: 0.0,
            e1_s: 0.0,
            e1_u: 0.0,
            last_high: None,
            last_low: None,
            initialized: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        if let (Some(last_h), Some(last_l)) = (self.last_high, self.last_low) {
            let dh = high - last_h;
            let dl = low - last_l;
            let x_hmu = if dh > 0.0 { dh } else { 0.0 };
            let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
            let x_price = x_hmu - x_lmd;
            let x_price_abs = x_price.abs();
            self.e0_r = self.alpha_r * x_price + self.alpha_r_1 * self.e0_r;
            self.e0_s = self.alpha_s * self.e0_r + self.alpha_s_1 * self.e0_s;
            self.e0_u = self.alpha_u * self.e0_s + self.alpha_u_1 * self.e0_u;
            self.e1_r = self.alpha_r * x_price_abs + self.alpha_r_1 * self.e1_r;
            self.e1_s = self.alpha_s * self.e1_r + self.alpha_s_1 * self.e1_s;
            self.e1_u = self.alpha_u * self.e1_s + self.alpha_u_1 * self.e1_u;
            self.last_high = Some(high);
            self.last_low = Some(low);
            if !self.e1_u.is_nan() && self.e1_u != 0.0 {
                Some(100.0 * self.e0_u / self.e1_u)
            
                } else {
                Some(0.0)
        } else {
            self.last_high = Some(high);
            self.last_low = Some(low);
            self.initialized = true;
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct DtiBatchRange {
    pub r: (usize, usize, usize),
    pub s: (usize, usize, usize),
    pub u: (usize, usize, usize),
}

impl Default for DtiBatchRange {
    fn default() -> Self {
        Self {
            r: (14, 14, 0),
            s: (10, 10, 0),
            u: (5, 5, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DtiBatchBuilder {
    range: DtiBatchRange,
    kernel: Kernel,
}

impl DtiBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn r_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.r = (start, end, step);
        self
    }
    #[inline]
    pub fn r_static(mut self, p: usize) -> Self {
        self.range.r = (p, p, 0);
        self
    }
    #[inline]
    pub fn s_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.s = (start, end, step);
        self
    }
    #[inline]
    pub fn s_static(mut self, x: usize) -> Self {
        self.range.s = (x, x, 0);
        self
    }
    #[inline]
    pub fn u_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.u = (start, end, step);
        self
    }
    #[inline]
    pub fn u_static(mut self, s: usize) -> Self {
        self.range.u = (s, s, 0);
        self
    }
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DtiBatchOutput, DtiError> {
        dti_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<DtiBatchOutput, DtiError> {
        DtiBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<DtiBatchOutput, DtiError> {
        let high = c.select_candle_field("high").map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
        let low = c.select_candle_field("low").map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<DtiBatchOutput, DtiError> {
        DtiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct DtiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DtiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DtiBatchOutput {
    pub fn row_for_params(&self, p: &DtiParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.r.unwrap_or(14) == p.r.unwrap_or(14)
                && c.s.unwrap_or(10) == p.s.unwrap_or(10)
                && c.u.unwrap_or(5) == p.u.unwrap_or(5)
        })
    }
    pub fn values_for(&self, p: &DtiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DtiBatchRange) -> Vec<DtiParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let rs = axis_usize(r.r);
    let ss = axis_usize(r.s);
    let us = axis_usize(r.u);
    let mut out = Vec::with_capacity(rs.len() * ss.len() * us.len());
    for &rr in &rs {
        for &ssv in &ss {
            for &uu in &us {
                out.push(DtiParams {
                    r: Some(rr),
                    s: Some(ssv),
                    u: Some(uu),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn dti_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &DtiBatchRange,
    k: Kernel,
) -> Result<DtiBatchOutput, DtiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(DtiError::InvalidPeriod { period: 0, data_len: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    dti_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn dti_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DtiBatchRange,
    kern: Kernel,
) -> Result<DtiBatchOutput, DtiError> {
    dti_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn dti_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DtiBatchRange,
    kern: Kernel,
) -> Result<DtiBatchOutput, DtiError> {
    dti_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn dti_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &DtiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DtiBatchOutput, DtiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DtiError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let len = high.len();
    if low.len() != len {
        return Err(DtiError::EmptyData);
    }
    let first_valid = (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan()).ok_or(DtiError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.r.unwrap().max(c.s.unwrap()).max(c.u.unwrap())).max().unwrap();
    if len - first_valid < max_p {
        return Err(DtiError::NotEnoughValidData { needed: max_p, valid: len - first_valid });
    }
    let rows = combos.len();
    let cols = len;
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        match kern {
            Kernel::Scalar => dti_row_scalar(
                high,
                low,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                first_valid,
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dti_row_avx2(
                high,
                low,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                first_valid,
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dti_row_avx512(
                high,
                low,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                first_valid,
                out_row,
            ),
            _ => unreachable!(),
        }
    };
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        values

                    .par_chunks_mut(cols)

                    .enumerate()

                    .for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, slice) in values.chunks_mut(cols).enumerate() {

                    do_row(row, slice);

        }

    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(DtiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn expand_grid_dti(r: &DtiBatchRange) -> Vec<DtiParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    fn check_dti_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = DtiParams {
            r: None,
            s: None,
            u: None,
        };
        let input = DtiInput::from_candles(&candles, default_params);
        let output = dti_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_dti_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DtiInput::with_default_candles(&candles);
        let result = dti_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -39.0091620347991,
            -39.75219264093014,
            -40.53941417932286,
            -41.2787749205189,
            -42.93758699380749,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] DTI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_dti_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DtiInput::with_default_candles(&candles);
        let output = dti_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_dti_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 11.0];
        let params = DtiParams {
            r: Some(0),
            s: Some(10),
            u: Some(5),
        };
        let input = DtiInput::from_slices(&high, &low, params);
        let res = dti_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DTI should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_dti_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0];
        let low = [9.0, 10.0];
        let params = DtiParams {
            r: Some(14),
            s: Some(10),
            u: Some(5),
        };
        let input = DtiInput::from_slices(&high, &low, params);
        let res = dti_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DTI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_dti_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = DtiParams::default();
        let input = DtiInput::from_slices(&high, &low, params);
        let res = dti_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] DTI should fail with all NaN", test_name);
        Ok(())
    }
    fn check_dti_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let params = DtiParams::default();
        let input = DtiInput::from_slices(&high, &low, params);
        let res = dti_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] DTI should fail with empty data", test_name);
        Ok(())
    }
    fn check_dti_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = candles.select_candle_field("high")?;
        let low = candles.select_candle_field("low")?;
        let params = DtiParams::default();
        let input = DtiInput::from_slices(high, low, params.clone());
        let batch_output = dti_with_kernel(&input, kernel)?.values;
        let mut stream = DtiStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(high.len());
        for (&h, &l) in high.iter().zip(low.iter()) {
            match stream.update(h, l) {
                Some(val) => stream_values.push(val),
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
                diff < 1e-9,
                "[{}] DTI streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }
    macro_rules! generate_all_dti_tests {
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
    generate_all_dti_tests!(
        check_dti_partial_params,
        check_dti_accuracy,
        check_dti_default_candles,
        check_dti_zero_period,
        check_dti_period_exceeds_length,
        check_dti_all_nan,
        check_dti_empty_data,
        check_dti_streaming
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = DtiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;
        let def = DtiParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -39.0091620347991,
            -39.75219264093014,
            -40.53941417932286,
            -41.2787749205189,
            -42.93758699380749,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
