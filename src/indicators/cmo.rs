//! # Chande Momentum Oscillator (CMO)
//!
//! A momentum oscillator that compares the sum of recent gains to recent losses over a given period.
//! The CMO oscillates between +100 and -100, with values near +100 indicating strong upward momentum and values near -100 indicating strong downward momentum.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default: 14).
//! - **source**: Candle field (e.g., `"close"`, default: `"close"`).
//!
//! ## Returns
//! - **`Ok(CmoOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(CmoError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: Implemented (warm-up vectorized only). Underperforms (<5%) vs scalar at 100k; Auto short-circuits to scalar.
//! - **AVX512 kernel**: Implemented (warm-up vectorized only). Underperforms vs scalar; Auto short-circuits to scalar.
//! - **Streaming**: Implemented (Wilder-style O(1) update; FMA-friendly)
//! - **Memory optimization**: ✅ Uses alloc_with_nan_prefix (zero-copy)
//! - **Batch operations**: ✅ Implemented with parallel processing support

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for CmoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CmoData::Slice(slice) => slice,
            CmoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CmoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CmoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CmoParams {
    pub period: Option<usize>,
}

impl Default for CmoParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct CmoInput<'a> {
    pub data: CmoData<'a>,
    pub params: CmoParams,
}

impl<'a> CmoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CmoParams) -> Self {
        Self {
            data: CmoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CmoParams) -> Self {
        Self {
            data: CmoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CmoParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn data_len(&self) -> usize {
        match &self.data {
            CmoData::Slice(slice) => slice.len(),
            CmoData::Candles { candles, .. } => candles.close.len(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CmoBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CmoBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CmoBuilder {
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
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CmoOutput, CmoError> {
        let p = CmoParams {
            period: self.period,
        };
        let i = CmoInput::from_candles(c, "close", p);
        cmo_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CmoOutput, CmoError> {
        let p = CmoParams {
            period: self.period,
        };
        let i = CmoInput::from_slice(d, p);
        cmo_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CmoStream, CmoError> {
        let p = CmoParams {
            period: self.period,
        };
        CmoStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CmoError {
    #[error("cmo: Empty data provided.")]
    EmptyData,

    #[error("cmo: Invalid period: period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("cmo: All values are NaN.")]
    AllValuesNaN,

    #[error("cmo: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("cmo: Invalid output length: expected={expected}, got={got}")]
    InvalidOutputLen { expected: usize, got: usize },
}

#[inline]
pub fn cmo(input: &CmoInput) -> Result<CmoOutput, CmoError> {
    cmo_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn cmo_prepare<'a>(
    input: &'a CmoInput,
    k: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), CmoError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CmoError::EmptyData);
    }
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(CmoError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CmoError::AllValuesNaN)?;
    if len - first <= period {
        return Err(CmoError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }
    let mut chosen = match k {
        // SIMD underperforms (<5%) for CMO; keep Auto on scalar.
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };
    // Normalize any batch kernels to their single-series equivalents here.
    if chosen.is_batch() {
        chosen = match chosen {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => chosen,
        };
    }
    Ok((data, period, first, chosen))
}

#[inline(always)]
fn cmo_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => cmo_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cmo_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => cmo_avx512(data, period, first, out),
            _ => unreachable!(),
        }
    }
}

pub fn cmo_with_kernel(input: &CmoInput, kernel: Kernel) -> Result<CmoOutput, CmoError> {
    let (data, period, first, chosen) = cmo_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), first + period);
    cmo_compute_into(data, period, first, chosen, &mut out);
    Ok(CmoOutput { values: out })
}

#[inline]
pub fn cmo_into_slice(dst: &mut [f64], input: &CmoInput, kern: Kernel) -> Result<(), CmoError> {
    let (data, period, first, chosen) = cmo_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(CmoError::InvalidOutputLen {
            expected: data.len(),
            got: dst.len(),
        });
    }
    cmo_compute_into(data, period, first, chosen, dst);
    let warmup_end = first + period;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline]
pub fn cmo_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    let mut prev_price = data[first_valid];

    let start_loop = first_valid + 1;
    let init_end = first_valid + period;

    let period_f = period as f64;
    let period_m1 = (period - 1) as f64;
    let inv_period = 1.0 / period_f;

    for i in start_loop..data.len() {
        let curr = data[i];
        let diff = curr - prev_price;
        prev_price = curr;

        let abs_diff = diff.abs();
        let gain = 0.5 * (diff + abs_diff);
        let loss = 0.5 * (abs_diff - diff);

        if i <= init_end {
            avg_gain += gain;
            avg_loss += loss;
            if i == init_end {
                avg_gain *= inv_period;
                avg_loss *= inv_period;
                let sum_gl = avg_gain + avg_loss;
                out[i] = if sum_gl != 0.0 {
                    100.0 * ((avg_gain - avg_loss) / sum_gl)
                } else {
                    0.0
                };
            }
        } else {
            avg_gain *= period_m1;
            avg_loss *= period_m1;
            avg_gain += gain;
            avg_loss += loss;
            avg_gain *= inv_period;
            avg_loss *= inv_period;
            let sum_gl = avg_gain + avg_loss;
            out[i] = if sum_gl != 0.0 {
                100.0 * ((avg_gain - avg_loss) / sum_gl)
            } else {
                0.0
            };
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cmo_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe {
        if period <= 32 {
            cmo_avx512_short(data, period, first_valid, out)
        } else {
            cmo_avx512_long(data, period, first_valid, out)
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cmo_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    // Vectorize warm-up, then scalar rolling update for parity
    unsafe { cmo_avx2_impl(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cmo_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    cmo_avx512_impl(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cmo_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    cmo_avx512_impl(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn cmo_avx2_impl(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    debug_assert!(out.len() == data.len());

    #[inline(always)]
    unsafe fn hsum256_pd(v: __m256d) -> f64 {
        let hi = _mm256_extractf128_pd(v, 1);
        let lo = _mm256_castpd256_pd128(v);
        let sum2 = _mm_add_pd(lo, hi);
        let hi64 = _mm_unpackhi_pd(sum2, sum2);
        _mm_cvtsd_f64(_mm_add_sd(sum2, hi64))
    }

    let len = data.len();
    let start = first_valid + 1;
    let init_end = first_valid + period;

    let inv_period = 1.0 / (period as f64);
    let period_m1 = (period - 1) as f64;

    // Warm-up accumulation
    let mut acc_gain_v = _mm256_setzero_pd();
    let mut acc_loss_v = _mm256_setzero_pd();
    let half_v = _mm256_set1_pd(0.5);
    let abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFu64 as i64));

    let mut sum_gain = 0.0f64;
    let mut sum_loss = 0.0f64;

    let mut i = start;
    while i + 3 <= init_end {
        let curr_v = _mm256_loadu_pd(data.as_ptr().add(i));
        let prev_v = _mm256_loadu_pd(data.as_ptr().add(i - 1));
        let diff_v = _mm256_sub_pd(curr_v, prev_v);

        let ad_v = _mm256_and_pd(diff_v, abs_mask);
        let gain_v = _mm256_mul_pd(_mm256_add_pd(ad_v, diff_v), half_v);
        let loss_v = _mm256_mul_pd(_mm256_sub_pd(ad_v, diff_v), half_v);

        acc_gain_v = _mm256_add_pd(acc_gain_v, gain_v);
        acc_loss_v = _mm256_add_pd(acc_loss_v, loss_v);

        i += 4;
    }

    sum_gain += hsum256_pd(acc_gain_v);
    sum_loss += hsum256_pd(acc_loss_v);

    // Scalar tail of warm-up
    let mut prev = if i == start {
        *data.get_unchecked(first_valid)
    } else {
        *data.get_unchecked(i - 1)
    };

    while i <= init_end {
        let curr = *data.get_unchecked(i);
        let diff = curr - prev;
        prev = curr;

        let ad = diff.abs();
        sum_gain += 0.5 * (ad + diff);
        sum_loss += 0.5 * (ad - diff);
        i += 1;
    }

    // First output at warm-up end
    let mut avg_gain = sum_gain * inv_period;
    let mut avg_loss = sum_loss * inv_period;
    {
        let sum_gl = avg_gain + avg_loss;
        *out.get_unchecked_mut(init_end) = if sum_gl != 0.0 {
            100.0 * ((avg_gain - avg_loss) / sum_gl)
        } else {
            0.0
        };
    }

    // Rolling update (scalar)
    while i < len {
        let curr = *data.get_unchecked(i);
        let diff = curr - prev;
        prev = curr;

        let ad = diff.abs();
        let gain = 0.5 * (ad + diff);
        let loss = 0.5 * (ad - diff);

        avg_gain *= period_m1;
        avg_loss *= period_m1;
        avg_gain += gain;
        avg_loss += loss;
        avg_gain *= inv_period;
        avg_loss *= inv_period;

        let sum_gl = avg_gain + avg_loss;
        *out.get_unchecked_mut(i) = if sum_gl != 0.0 {
            100.0 * ((avg_gain - avg_loss) / sum_gl)
        } else {
            0.0
        };

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn cmo_avx512_impl(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    debug_assert!(out.len() == data.len());

    #[inline(always)]
    unsafe fn hsum256_pd(v: __m256d) -> f64 {
        let hi = _mm256_extractf128_pd(v, 1);
        let lo = _mm256_castpd256_pd128(v);
        let sum2 = _mm_add_pd(lo, hi);
        let hi64 = _mm_unpackhi_pd(sum2, sum2);
        _mm_cvtsd_f64(_mm_add_sd(sum2, hi64))
    }

    #[inline(always)]
    unsafe fn hsum512_pd(v: __m512d) -> f64 {
        let lo256 = _mm512_castpd512_pd256(v);
        let hi256 = _mm512_extractf64x4_pd(v, 1);
        hsum256_pd(_mm256_add_pd(lo256, hi256))
    }

    let len = data.len();
    let start = first_valid + 1;
    let init_end = first_valid + period;

    let inv_period = 1.0 / (period as f64);
    let period_m1 = (period - 1) as f64;

    // Warm-up accumulation (8 diffs per iter)
    let mut acc_gain_v = _mm512_setzero_pd();
    let mut acc_loss_v = _mm512_setzero_pd();
    let half_v = _mm512_set1_pd(0.5);
    let abs_mask_i = _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64);

    let mut sum_gain = 0.0f64;
    let mut sum_loss = 0.0f64;

    let mut i = start;
    while i + 7 <= init_end {
        let curr_v = _mm512_loadu_pd(data.as_ptr().add(i));
        let prev_v = _mm512_loadu_pd(data.as_ptr().add(i - 1));
        let diff_v = _mm512_sub_pd(curr_v, prev_v);

        let diff_i = _mm512_castpd_si512(diff_v);
        let abs_i = _mm512_and_si512(diff_i, abs_mask_i);
        let ad_v = _mm512_castsi512_pd(abs_i);

        let gain_v = _mm512_mul_pd(_mm512_add_pd(ad_v, diff_v), half_v);
        let loss_v = _mm512_mul_pd(_mm512_sub_pd(ad_v, diff_v), half_v);

        acc_gain_v = _mm512_add_pd(acc_gain_v, gain_v);
        acc_loss_v = _mm512_add_pd(acc_loss_v, loss_v);

        i += 8;
    }

    sum_gain += hsum512_pd(acc_gain_v);
    sum_loss += hsum512_pd(acc_loss_v);

    // Scalar tail of warm-up
    let mut prev = if i == start {
        *data.get_unchecked(first_valid)
    } else {
        *data.get_unchecked(i - 1)
    };

    while i <= init_end {
        let curr = *data.get_unchecked(i);
        let diff = curr - prev;
        prev = curr;

        let ad = diff.abs();
        sum_gain += 0.5 * (ad + diff);
        sum_loss += 0.5 * (ad - diff);
        i += 1;
    }

    // First output at warm-up end
    let mut avg_gain = sum_gain * inv_period;
    let mut avg_loss = sum_loss * inv_period;
    {
        let sum_gl = avg_gain + avg_loss;
        *out.get_unchecked_mut(init_end) = if sum_gl != 0.0 {
            100.0 * ((avg_gain - avg_loss) / sum_gl)
        } else {
            0.0
        };
    }

    // Rolling update (scalar)
    while i < len {
        let curr = *data.get_unchecked(i);
        let diff = curr - prev;
        prev = curr;

        let ad = diff.abs();
        let gain = 0.5 * (ad + diff);
        let loss = 0.5 * (ad - diff);

        avg_gain *= period_m1;
        avg_loss *= period_m1;
        avg_gain += gain;
        avg_loss += loss;
        avg_gain *= inv_period;
        avg_loss *= inv_period;

        let sum_gl = avg_gain + avg_loss;
        *out.get_unchecked_mut(i) = if sum_gl != 0.0 {
            100.0 * ((avg_gain - avg_loss) / sum_gl)
        } else {
            0.0
        };

        i += 1;
    }
}

#[inline(always)]
pub fn cmo_batch_with_kernel(
    data: &[f64],
    sweep: &CmoBatchRange,
    k: Kernel,
) -> Result<CmoBatchOutput, CmoError> {
    let kernel = match k {
        // SIMD batch underperforms/unused for CMO; default to scalar batch
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => {
            return Err(CmoError::InvalidPeriod {
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
    cmo_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CmoBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CmoBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 40, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CmoBatchBuilder {
    range: CmoBatchRange,
    kernel: Kernel,
}

impl CmoBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<CmoBatchOutput, CmoError> {
        cmo_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CmoBatchOutput, CmoError> {
        CmoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CmoBatchOutput, CmoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CmoBatchOutput, CmoError> {
        CmoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct CmoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CmoParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CmoBatchOutput {
    pub fn row_for_params(&self, p: &CmoParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &CmoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CmoBatchRange) -> Vec<CmoParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(CmoParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn cmo_batch_slice(
    data: &[f64],
    sweep: &CmoBatchRange,
    kern: Kernel,
) -> Result<CmoBatchOutput, CmoError> {
    cmo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cmo_batch_par_slice(
    data: &[f64],
    sweep: &CmoBatchRange,
    kern: Kernel,
) -> Result<CmoBatchOutput, CmoError> {
    cmo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cmo_batch_inner(
    data: &[f64],
    sweep: &CmoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CmoBatchOutput, CmoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CmoError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CmoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first <= max_p {
        return Err(CmoError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Step 1: Allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Step 2: Calculate warmup periods for each row
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    // Step 3: Initialize NaN prefixes for each row
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Step 4: Convert to mutable slice for computation
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Precompute per-step gain/loss and their prefix sums once for all rows
    let len = data.len();
    let start = first + 1;
    let mut gains = vec![0.0f64; len];
    let mut losses = vec![0.0f64; len];
    for i in start..len {
        let diff = data[i] - data[i - 1];
        let ad = diff.abs();
        gains[i] = 0.5 * (ad + diff);
        losses[i] = 0.5 * (ad - diff);
    }
    let mut pg = vec![0.0f64; len + 1];
    let mut pl = vec![0.0f64; len + 1];
    for i in 0..len {
        pg[i + 1] = pg[i] + gains[i];
        pl[i + 1] = pl[i] + losses[i];
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        cmo_row_from_gl(&gains, &losses, &pg, &pl, first, period, out_row);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // Step 5: Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(CmoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn cmo_batch_inner_into(
    data: &[f64],
    sweep: &CmoBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CmoParams>, CmoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CmoError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CmoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first <= max_p {
        return Err(CmoError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }
    let cols = data.len();

    // 2a) Treat caller buffer as MaybeUninit and initialize only warm prefixes.
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    // 2b) Precompute per-step gain/loss and prefix sums once
    let len = data.len();
    let start = first + 1;
    let mut gains = vec![0.0f64; len];
    let mut losses = vec![0.0f64; len];
    for i in start..len {
        let diff = data[i] - data[i - 1];
        let ad = diff.abs();
        gains[i] = 0.5 * (ad + diff);
        losses[i] = 0.5 * (ad - diff);
    }
    let mut pg = vec![0.0f64; len + 1];
    let mut pl = vec![0.0f64; len + 1];
    for i in 0..len {
        pg[i + 1] = pg[i] + gains[i];
        pl[i + 1] = pl[i] + losses[i];
    }

    // 2c) Now compute rows writing real values after warmup using precompute.
    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let row_dst: &mut [f64] =
            std::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len());
        cmo_row_from_gl(&gains, &losses, &pg, &pl, first, period, row_dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, row_mu)| do_row(r, row_mu));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, row_mu) in out_mu.chunks_mut(cols).enumerate() {
                do_row(r, row_mu);
            }
        }
    } else {
        for (r, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, row_mu);
        }
    }

    Ok(combos)
}

#[inline]
pub fn cmo_batch_into_slice(
    out: &mut [f64],
    data: &[f64],
    sweep: &CmoBatchRange,
    k: Kernel,
) -> Result<Vec<CmoParams>, CmoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(CmoError::InvalidPeriod {
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
    cmo_batch_inner_into(data, sweep, simd, true, out)
}

#[inline(always)]
unsafe fn cmo_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cmo_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn cmo_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cmo_avx2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn cmo_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        cmo_row_avx512_short(data, first, period, out);
    } else {
        cmo_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn cmo_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cmo_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn cmo_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cmo_avx512_long(data, period, first, out)
}

#[inline(always)]
unsafe fn cmo_row_from_gl(
    gains: &[f64],
    losses: &[f64],
    pg: &[f64],
    pl: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    let len = out.len();
    let start = first + 1;
    let init_end = first + period;
    let inv_period = 1.0 / (period as f64);
    let period_m1 = (period - 1) as f64;

    // Warm-up sums via prefix differences
    let sum_gain = pg[init_end + 1] - pg[start];
    let sum_loss = pl[init_end + 1] - pl[start];
    let mut avg_gain = sum_gain * inv_period;
    let mut avg_loss = sum_loss * inv_period;

    // First output at warm-up end
    {
        let sum_gl = avg_gain + avg_loss;
        *out.get_unchecked_mut(init_end) = if sum_gl != 0.0 {
            100.0 * ((avg_gain - avg_loss) / sum_gl)
        } else {
            0.0
        };
    }

    // Rolling update using precomputed gains/losses
    let mut i = init_end + 1;
    while i < len {
        let g = *gains.get_unchecked(i);
        let l = *losses.get_unchecked(i);

        avg_gain *= period_m1;
        avg_loss *= period_m1;
        avg_gain += g;
        avg_loss += l;
        avg_gain *= inv_period;
        avg_loss *= inv_period;

        let sum_gl = avg_gain + avg_loss;
        *out.get_unchecked_mut(i) = if sum_gl != 0.0 {
            100.0 * ((avg_gain - avg_loss) / sum_gl)
        } else {
            0.0
        };
        i += 1;
    }
}

#[derive(Debug, Clone)]
pub struct CmoStream {
    period: usize,
    inv_period: f64, // 1/period (precomputed)
    avg_gain: f64,
    avg_loss: f64,
    prev: f64,
    head: usize,
    started: bool,
    filled: bool,
}

impl CmoStream {
    pub fn try_new(params: CmoParams) -> Result<Self, CmoError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(CmoError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            inv_period: 1.0 / (period as f64),
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev: 0.0,
            head: 0,
            started: false,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Seed on the first observed value.
        if !self.started {
            self.prev = value;
            self.started = true;
            return None;
        }

        // Branchless up/down decomposition
        let diff = value - self.prev;
        self.prev = value;

        let ad = diff.abs();
        let gain = 0.5 * (ad + diff);
        let loss = 0.5 * (ad - diff);

        // Warm-up: accumulate raw sums, then normalize once
        if !self.filled {
            self.avg_gain += gain;
            self.avg_loss += loss;
            self.head += 1;

            if self.head == self.period {
                self.avg_gain *= self.inv_period;
                self.avg_loss *= self.inv_period;
                self.filled = true;

                let denom = self.avg_gain + self.avg_loss;
                return Some(if denom != 0.0 {
                    100.0 * (self.avg_gain - self.avg_loss) / denom
                } else {
                    0.0
                });
            }
            return None;
        }

        // Rolling EMA-like (Wilder) update: avg += (x - avg) * inv_period
        let ip = self.inv_period;
        self.avg_gain = (gain - self.avg_gain).mul_add(ip, self.avg_gain);
        self.avg_loss = (loss - self.avg_loss).mul_add(ip, self.avg_loss);

        let denom = self.avg_gain + self.avg_loss;
        Some(if denom != 0.0 {
            100.0 * (self.avg_gain - self.avg_loss) / denom
        } else {
            0.0
        })
    }
}

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::CudaCmo;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;

#[cfg(feature = "python")]
#[pyfunction(name = "cmo")]
#[pyo3(signature = (data, period=None, kernel=None))]
pub fn cmo_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = CmoParams { period };
    let input = CmoInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| cmo_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CmoStream")]
pub struct CmoStreamPy {
    stream: CmoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CmoStreamPy {
    #[new]
    fn new(period: Option<usize>) -> PyResult<Self> {
        let params = CmoParams { period };
        let stream =
            CmoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(CmoStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cmo_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn cmo_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;

    let sweep = CmoBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            cmo_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "cmo_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn cmo_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, pyo3::types::PyDict>)> {
    use crate::cuda::cuda_available;
    use numpy::IntoPyArray;
    use pyo3::types::PyDict;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let prices = data_f32.as_slice()?;
    let sweep = CmoBatchRange {
        period: period_range,
    };
    let (inner, combos) = py.allow_threads(|| {
        let cuda = CudaCmo::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.cmo_batch_dev(prices, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    let periods: Vec<u64> = combos.iter().map(|c| c.period.unwrap() as u64).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;

    Ok((DeviceArrayF32Py { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "cmo_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn cmo_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let flat = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = CmoParams { period: Some(period) };
    let inner = py.allow_threads(|| {
        let cuda = CudaCmo::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.cmo_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_cmo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = CmoParams { period: None };
        let input = CmoInput::from_candles(&candles, "close", default_params);
        let output = cmo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let params_10 = CmoParams { period: Some(10) };
        let input_10 = CmoInput::from_candles(&candles, "hl2", params_10);
        let output_10 = cmo_with_kernel(&input_10, kernel)?;
        assert_eq!(output_10.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cmo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CmoParams { period: Some(14) };
        let input = CmoInput::from_candles(&candles, "close", params);
        let cmo_result = cmo_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -13.152504931406101,
            -14.649876201213106,
            -16.760170709240303,
            -14.274505732779227,
            -21.984038127126716,
        ];
        let start_idx = cmo_result.values.len() - 5;
        let last_five = &cmo_result.values[start_idx..];
        for (i, &actual) in last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (actual - expected).abs() < 1e-6,
                "[{}] CMO mismatch at final 5 index {}: expected {}, got {}",
                test_name,
                i,
                expected,
                actual
            );
        }
        Ok(())
    }

    fn check_cmo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CmoInput::with_default_candles(&candles);
        match input.data {
            CmoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CmoData::Candles variant"),
        }
        let output = cmo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cmo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = CmoParams { period: Some(0) };
        let input = CmoInput::from_slice(&data, params);
        let result = cmo_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for period=0",
            test_name
        );
        Ok(())
    }

    fn check_cmo_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = CmoParams { period: Some(10) };
        let input = CmoInput::from_slice(&data, params);
        let result = cmo_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for period>data.len()",
            test_name
        );
        Ok(())
    }

    fn check_cmo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single = [42.0];
        let params = CmoParams { period: Some(14) };
        let input = CmoInput::from_slice(&single, params);
        let result = cmo_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Expected error for insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cmo_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = CmoParams { period: Some(14) };
        let first_input = CmoInput::from_candles(&candles, "close", first_params);
        let first_result = cmo_with_kernel(&first_input, kernel)?;
        let second_params = CmoParams { period: Some(14) };
        let second_input = CmoInput::from_slice(&first_result.values, second_params);
        let second_result = cmo_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Expected no NaN after index 28, found NaN at {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_cmo_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase coverage
        let test_periods = vec![7, 14, 21, 28];

        for period in test_periods {
            let params = CmoParams {
                period: Some(period),
            };
            let input = CmoInput::from_candles(&candles, "close", params);
            let output = cmo_with_kernel(&input, kernel)?;

            // Check every value for poison patterns
            for (i, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in the warmup period
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_cmo_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_cmo_tests {
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

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_cmo_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Test strategy: period 1-50 (including edge case), data length period-400
        // Avoid near-zero values to prevent edge cases
        let strat = (1usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64)
                        .prop_filter("finite and non-zero", |x| x.is_finite() && x.abs() > 1e-10),
                    (period + 1).max(2)..400, // Need period+1 data points for CMO
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = CmoParams {
                    period: Some(period),
                };
                let input = CmoInput::from_slice(&data, params);

                let CmoOutput { values: out } = cmo_with_kernel(&input, kernel).unwrap();
                let CmoOutput { values: ref_out } =
                    cmo_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Output length should match input length
                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                // Property 2: Warmup period validation
                // CMO warmup is first_valid + period
                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup = first_valid + period;

                for i in 0..warmup.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Property 3: First valid output should be at warmup index
                if warmup < out.len() {
                    prop_assert!(
                        !out[warmup].is_nan(),
                        "Expected valid value at index {} (first after warmup), got NaN",
                        warmup
                    );
                }

                // Properties for valid outputs (after warmup)
                for i in warmup..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Property 4: Output bounds check - CMO is bounded [-100, 100]
                    prop_assert!(
                        y.is_nan() || (y >= -100.0 - 1e-9 && y <= 100.0 + 1e-9),
                        "CMO value {} at index {} outside bounds [-100, 100]",
                        y,
                        i
                    );

                    // Property 5: Finite inputs produce finite outputs
                    if data[..=i].iter().all(|x| x.is_finite()) {
                        prop_assert!(
                            y.is_finite(),
                            "Expected finite output at index {}, got {}",
                            i,
                            y
                        );
                    }

                    // Property 6: Kernel consistency (reduced ULP tolerance for better precision)
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y_bits,
                            r_bits,
                            "Finite/NaN mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 8,
                        "Kernel mismatch at index {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                // Property 7: Constant data should produce CMO of 0 (tighter tolerance)
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && warmup < data.len() {
                    let cmo_val = out[warmup];
                    prop_assert!(
                        cmo_val.abs() <= 1e-9,
                        "Constant data should produce CMO of 0, got {} at index {}",
                        cmo_val,
                        warmup
                    );
                }

                // Property 8: Monotonically increasing data should produce positive CMO
                let is_increasing = data.windows(2).all(|w| w[1] >= w[0] - 1e-10);
                if is_increasing && warmup < data.len() {
                    for i in warmup..data.len() {
                        prop_assert!(
							out[i].is_nan() || out[i] >= -1e-6,
							"Monotonically increasing data should produce non-negative CMO, got {} at index {}",
							out[i],
							i
						);
                    }
                }

                // Property 9: Monotonically decreasing data should produce negative CMO
                let is_decreasing = data.windows(2).all(|w| w[1] <= w[0] + 1e-10);
                if is_decreasing && warmup < data.len() {
                    for i in warmup..data.len() {
                        prop_assert!(
							out[i].is_nan() || out[i] <= 1e-6,
							"Monotonically decreasing data should produce non-positive CMO, got {} at index {}",
							out[i],
							i
						);
                    }
                }

                // Property 10: Extreme movements test - strong gains should approach +100
                if period > 1 && warmup + 5 < data.len() {
                    // Check if we have strong upward movement
                    let has_strong_gains = (warmup..data.len().min(warmup + 10))
                        .zip(warmup.saturating_sub(1)..data.len().saturating_sub(1).min(warmup + 9))
                        .all(|(i, j)| data[i] > data[j] * 1.1); // 10% gains each step

                    if has_strong_gains {
                        let last_idx = data.len() - 1;
                        prop_assert!(
                            out[last_idx].is_nan() || out[last_idx] >= 50.0,
                            "Strong gains should produce CMO > 50, got {} at index {}",
                            out[last_idx],
                            last_idx
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_cmo_tests!(
        check_cmo_partial_params,
        check_cmo_accuracy,
        check_cmo_default_candles,
        check_cmo_zero_period,
        check_cmo_period_exceeds_length,
        check_cmo_very_small_dataset,
        check_cmo_reinput,
        check_cmo_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_cmo_tests!(check_cmo_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = CmoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = CmoParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations
        let output = CmoBatchBuilder::new()
            .kernel(kernel)
            .period_range(7, 28, 7) // Test periods: 7, 14, 21, 28
            .apply_candles(&c, "close")?;

        // Check every value in the entire batch matrix for poison patterns
        for (idx, &val) in output.values.iter().enumerate() {
            // Skip NaN values as they're expected in warmup periods
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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
    gen_batch_tests!(check_batch_no_poison);
}

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cmo_js(data: &[f64], period: Option<usize>) -> Result<Vec<f64>, JsValue> {
    let params = CmoParams { period };
    let input = CmoInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()]; // Single allocation
    cmo_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cmo_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: Option<usize>,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = CmoParams { period };
        let input = CmoInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // CRITICAL: Aliasing check
            let mut temp = vec![0.0; len];
            cmo_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            cmo_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cmo_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cmo_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CmoBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CmoBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CmoParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cmo_batch)]
pub fn cmo_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: CmoBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let (p_start, p_end, p_step) = config.period_range;

    let batch_range = CmoBatchRange {
        period: (p_start, p_end, p_step),
    };

    let output = cmo_batch_with_kernel(data, &batch_range, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = CmoBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cmo_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to cmo_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = CmoBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        cmo_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
