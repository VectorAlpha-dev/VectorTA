//! # Ehlers Kaufman Adaptive Moving Average (EKAMA)
//!
//! Variation of Kaufman's Adaptive Moving Average with Ehlers' modifications for
//! improved responsiveness. Dynamically adjusts smoothing factor based on price
//! efficiency ratio to adapt between trending and ranging markets.
//!
//! ## Parameters
//! - **period**: Lookback period for efficiency calculation (default: 20)
//!
//! ## Returns
//! - **values**: Adaptive moving average that adjusts to market conditions
//!
//! ## Developer Status
//! - **AVX2/AVX512**: Enabled for initial ER denominator (vector abs-diff sum);
//!   main recurrence remains scalar (recursive filter). Auto selection falls back
//!   to scalar when `nightly-avx` is disabled. Expect modest wins on long series.
//! - **Scalar path**: Minor optimizations (branch hoist, mul_add, no powi).
//! - **Batch (row-specific)**: Uses shared prefix sums of abs-diffs to seed ER
//!   denominators in O(1) per row; reduces overhead for wide sweeps. Outputs
//!   unchanged; warmup handling remains identical.
//! - **Streaming update**: O(n) reference stream retained; could be made O(1)
//!   with a rolling window if needed in future.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "cuda")]
use crate::cuda::moving_averages::CudaEhlersKama;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EhlersKamaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersKamaData::Slice(slice) => slice,
            EhlersKamaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhlersKamaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersKamaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct EhlersKamaParams {
    pub period: Option<usize>,
}

impl Default for EhlersKamaParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersKamaInput<'a> {
    pub data: EhlersKamaData<'a>,
    pub params: EhlersKamaParams,
}

impl<'a> EhlersKamaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EhlersKamaParams) -> Self {
        Self {
            data: EhlersKamaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EhlersKamaParams) -> Self {
        Self {
            data: EhlersKamaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EhlersKamaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersKamaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for EhlersKamaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersKamaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<EhlersKamaOutput, EhlersKamaError> {
        let p = EhlersKamaParams {
            period: self.period,
        };
        let i = EhlersKamaInput::from_candles(c, "close", p);
        ehlers_kama_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EhlersKamaOutput, EhlersKamaError> {
        let p = EhlersKamaParams {
            period: self.period,
        };
        let i = EhlersKamaInput::from_slice(d, p);
        ehlers_kama_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersKamaStream, EhlersKamaError> {
        let p = EhlersKamaParams {
            period: self.period,
        };
        EhlersKamaStream::try_new(p)
    }

    pub fn with_default_candles(c: &Candles) -> Result<EhlersKamaOutput, EhlersKamaError> {
        Self::new().kernel(Kernel::Auto).apply(c)
    }

    pub fn with_default_slice(d: &[f64]) -> Result<EhlersKamaOutput, EhlersKamaError> {
        Self::new().kernel(Kernel::Auto).apply_slice(d)
    }
}

#[derive(Debug, Error)]
pub enum EhlersKamaError {
    #[error("ehlers_kama: Input data slice is empty.")]
    EmptyInputData,
    #[error("ehlers_kama: All values are NaN.")]
    AllValuesNaN,
    #[error("ehlers_kama: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ehlers_kama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn ehlers_kama(input: &EhlersKamaInput) -> Result<EhlersKamaOutput, EhlersKamaError> {
    ehlers_kama_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn ehlers_kama_compute_into(
    data: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => ehlers_kama_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ehlers_kama_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ehlers_kama_avx512(data, period, first, out),
            _ => ehlers_kama_scalar(data, period, first, out), // Fallback to scalar
        }
    }
}

#[inline]
pub fn ehlers_kama_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), data.len());
    let len = data.len();
    if len == 0 {
        return;
    }

    // Start index is first_valid + period - 1
    let start = first_valid + period - 1;
    if start >= len {
        return;
    }

    // Calculate initial delta sum with PERIOD terms (not period-1).
    // Use max to avoid a per-iteration branch (k > first_valid is always true now).
    let mut delta_sum = 0.0;
    // Guard underflow by branching once here; avoids per-iteration checks below.
    let delta_start = if start >= period { start - period + 1 } else { first_valid + 1 };
    for k in delta_start..=start {
        delta_sum += (data[k] - data[k - 1]).abs();
    }

    // Pine-style initialization: no EMA warmup, just use previous price
    let mut prev_kama = data[start - 1]; // Pine-style seed

    // Apply adaptive formula for the first output
    // Direction uses correct lookback: start - (period - 1)
    let a0 = data[start];
    let direction = (a0 - data[start - (period - 1)]).abs();
    let ef = if delta_sum == 0.0 { 0.0 } else { (direction / delta_sum).min(1.0) };
    // s = ((0.6667 * ef) + 0.0645)^2, avoid powi and use mul_add for precision/speed
    let s_term = 0.6667f64.mul_add(ef, 0.0645);
    let mut s = s_term * s_term;
    prev_kama = s.mul_add(a0 - prev_kama, prev_kama);
    out[start] = prev_kama;

    // Continue with adaptive calculation
    for i in (start + 1)..len {
        // Rolling delta: maintain period terms by dropping from i-period
        let drop_idx = i - period;
        if drop_idx > first_valid {
            delta_sum -= (data[drop_idx] - data[drop_idx - 1]).abs();
        }
        let a = data[i];
        delta_sum += (a - data[i - 1]).abs();

        // Direction uses full period window
        let direction = (a - data[i - (period - 1)]).abs();
        let ef = if delta_sum == 0.0 { 0.0 } else { (direction / delta_sum).min(1.0) };

        // Ehlers smoothing constant
        // Original formula: s = ((0.6667 * ef) + 0.0645)^2
        let s_term = 0.6667f64.mul_add(ef, 0.0645);
        s = s_term * s_term;

        prev_kama = s.mul_add(a - prev_kama, prev_kama);
        out[i] = prev_kama;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn ehlers_kama_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), data.len());
    let len = data.len();
    if len == 0 { return; }
    let start = first_valid + period - 1;
    if start >= len { return; }

    use core::arch::x86_64::*;
    let d = data.as_ptr();
    let signmask = _mm256_set1_pd(-0.0);

    #[inline(always)]
    unsafe fn hsum256_pd(v: __m256d) -> f64 {
        let hi = _mm256_extractf128_pd(v, 1);
        let lo = _mm256_castpd256_pd128(v);
        let sum2 = _mm_add_pd(lo, hi);
        let hi64 = _mm_unpackhi_pd(sum2, sum2);
        let sum1 = _mm_add_sd(sum2, hi64);
        _mm_cvtsd_f64(sum1)
    }

    // Initial delta_sum over PERIOD consecutive diffs
    // Avoid underflow by computing start + 1 - period (== first_valid) instead of start - period + 1
    let mut k = core::cmp::max(first_valid + 1, start + 1 - period);
    let end = start;
    let mut acc_v = _mm256_setzero_pd();
    let mut delta_sum = 0.0f64;

    // Head to 4-aligned
    while ((end + 1).wrapping_sub(k)) & 3 != 0 {
        let a = *d.add(k);
        let b = *d.add(k - 1);
        delta_sum += (a - b).abs();
        k += 1;
        if k > end { break; }
    }
    // Vector body
    while k + 3 <= end {
        let curr = _mm256_loadu_pd(d.add(k));
        let prev = _mm256_loadu_pd(d.add(k - 1));
        let diff = _mm256_sub_pd(curr, prev);
        let adiff = _mm256_andnot_pd(signmask, diff);
        acc_v = _mm256_add_pd(acc_v, adiff);
        k += 4;
    }
    delta_sum += hsum256_pd(acc_v);

    // Tail
    while k <= end {
        let a = *d.add(k);
        let b = *d.add(k - 1);
        delta_sum += (a - b).abs();
        k += 1;
    }

    // Recurrence identical to scalar (with FMA updates)
    let o = out.as_mut_ptr();
    let mut prev_kama = *d.add(start - 1);
    let a0 = *d.add(start);
    let dir0 = (a0 - *d.add(start - (period - 1))).abs();
    let ef0 = if delta_sum == 0.0 { 0.0 } else { (dir0 / delta_sum).min(1.0) };
    let mut s_term = 0.6667f64.mul_add(ef0, 0.0645);
    let mut s = s_term * s_term;
    prev_kama = s.mul_add(a0 - prev_kama, prev_kama);
    *o.add(start) = prev_kama;

    let mut i = start + 1;
    while i < len {
        let drop_idx = i - period;
        if drop_idx > first_valid {
            let da = *d.add(drop_idx);
            let db = *d.add(drop_idx - 1);
            delta_sum -= (da - db).abs();
        }
        let a = *d.add(i);
        let b = *d.add(i - 1);
        delta_sum += (a - b).abs();

        let dir = (a - *d.add(i - (period - 1))).abs();
        let ef = if delta_sum == 0.0 { 0.0 } else { (dir / delta_sum).min(1.0) };

        s_term = 0.6667f64.mul_add(ef, 0.0645);
        s = s_term * s_term;

        prev_kama = s.mul_add(a - prev_kama, prev_kama);
        *o.add(i) = prev_kama;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
pub unsafe fn ehlers_kama_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), data.len());
    let len = data.len();
    if len == 0 { return; }
    let start = first_valid + period - 1;
    if start >= len { return; }

    use core::arch::x86_64::*;
    let d = data.as_ptr();
    let signmask = _mm512_set1_pd(-0.0);

    #[inline(always)]
    unsafe fn hsum256_pd(v: __m256d) -> f64 {
        let hi = _mm256_extractf128_pd(v, 1);
        let lo = _mm256_castpd256_pd128(v);
        let sum2 = _mm_add_pd(lo, hi);
        let hi64 = _mm_unpackhi_pd(sum2, sum2);
        let sum1 = _mm_add_sd(sum2, hi64);
        _mm_cvtsd_f64(sum1)
    }
    #[inline(always)]
    unsafe fn hsum512_pd(v: __m512d) -> f64 {
        let lo = _mm512_castpd512_pd256(v);
        let hi = _mm512_extractf64x4_pd(v, 1);
        hsum256_pd(lo) + hsum256_pd(hi)
    }

    // Initial delta_sum over PERIOD consecutive diffs
    let mut k = core::cmp::max(first_valid + 1, start + 1 - period);
    let end = start;
    let mut acc_v = _mm512_setzero_pd();
    let mut delta_sum = 0.0f64;

    // Head to 8-aligned
    while ((end + 1).wrapping_sub(k)) & 7 != 0 {
        let a = *d.add(k);
        let b = *d.add(k - 1);
        delta_sum += (a - b).abs();
        k += 1;
        if k > end { break; }
    }
    // Vector body (8-wide)
    while k + 7 <= end {
        let curr = _mm512_loadu_pd(d.add(k));
        let prev = _mm512_loadu_pd(d.add(k - 1));
        let diff = _mm512_sub_pd(curr, prev);
        let adiff = _mm512_andnot_pd(signmask, diff);
        acc_v = _mm512_add_pd(acc_v, adiff);
        k += 8;
    }
    delta_sum += hsum512_pd(acc_v);

    // Tail
    while k <= end {
        let a = *d.add(k);
        let b = *d.add(k - 1);
        delta_sum += (a - b).abs();
        k += 1;
    }

    // Recurrence identical to scalar (with FMA updates)
    let o = out.as_mut_ptr();
    let mut prev_kama = *d.add(start - 1);
    let a0 = *d.add(start);
    let dir0 = (a0 - *d.add(start - (period - 1))).abs();
    let ef0 = if delta_sum == 0.0 { 0.0 } else { (dir0 / delta_sum).min(1.0) };
    let mut s_term = 0.6667f64.mul_add(ef0, 0.0645);
    let mut s = s_term * s_term;
    prev_kama = s.mul_add(a0 - prev_kama, prev_kama);
    *o.add(start) = prev_kama;

    let mut i = start + 1;
    while i < len {
        let drop_idx = i - period;
        if drop_idx > first_valid {
            let da = *d.add(drop_idx);
            let db = *d.add(drop_idx - 1);
            delta_sum -= (da - db).abs();
        }
        let a = *d.add(i);
        let b = *d.add(i - 1);
        delta_sum += (a - b).abs();

        let dir = (a - *d.add(i - (period - 1))).abs();
        let ef = if delta_sum == 0.0 { 0.0 } else { (dir / delta_sum).min(1.0) };

        s_term = 0.6667f64.mul_add(ef, 0.0645);
        s = s_term * s_term;

        prev_kama = s.mul_add(a - prev_kama, prev_kama);
        *o.add(i) = prev_kama;
        i += 1;
    }
}

#[inline(always)]
fn ehlers_kama_prepare<'a>(
    input: &'a EhlersKamaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), EhlersKamaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(EhlersKamaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersKamaError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(EhlersKamaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(EhlersKamaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((data, period, first, chosen))
}

pub fn ehlers_kama_with_kernel(
    input: &EhlersKamaInput,
    kernel: Kernel,
) -> Result<EhlersKamaOutput, EhlersKamaError> {
    let (data, period, first, chosen) = ehlers_kama_prepare(input, kernel)?;
    let warmup_end = first + period - 1;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_end);
    ehlers_kama_compute_into(data, period, first, chosen, &mut out);
    Ok(EhlersKamaOutput { values: out })
}

/// Compute Ehlers KAMA directly into the provided output slice.
/// The output slice must be the same length as the input data.
#[inline]
pub fn ehlers_kama_into_slice(
    dst: &mut [f64],
    input: &EhlersKamaInput,
    kern: Kernel,
) -> Result<(), EhlersKamaError> {
    let (data, period, first, chosen) = ehlers_kama_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(EhlersKamaError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    // single compute
    ehlers_kama_compute_into(data, period, first, chosen, dst);

    // apply NaN warmup prefix without extra copies
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct EhlersKamaStream {
    period: usize,
    buffer: Vec<f64>, // Store all values for O(n) calculation
    prev_kama: f64,
}

impl EhlersKamaStream {
    pub fn try_new(params: EhlersKamaParams) -> Result<Self, EhlersKamaError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(EhlersKamaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: Vec::new(),
            prev_kama: f64::NAN,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Add new value to buffer
        self.buffer.push(value);
        let len = self.buffer.len();

        // Need at least period values to compute
        if len < self.period {
            return None;
        }

        // Mirror the batch implementation exactly
        // Calculate delta_sum with PERIOD consecutive differences
        let mut delta_sum = 0.0;

        // Current index in the stream corresponds to (len - 1) in batch terms
        let current_idx = len - 1;

        // We need period consecutive differences ending at current_idx
        // This means differences from (current_idx - period + 1) to current_idx
        let delta_start = if current_idx >= self.period {
            current_idx - self.period + 1
        } else {
            // When current_idx < period, start from 1 (first possible difference)
            1
        };

        for k in delta_start..=current_idx {
            if k > 0 && k < len {
                delta_sum += (self.buffer[k] - self.buffer[k - 1]).abs();
            }
        }

        // Initialize KAMA on first calculation
        if len == self.period {
            // Pine-style seed: use previous value
            self.prev_kama = self.buffer[current_idx - 1];

            // Apply adaptive formula for first output
            // Direction uses correct lookback: current_idx - (period - 1)
            let direction =
                (self.buffer[current_idx] - self.buffer[current_idx - (self.period - 1)]).abs();
            let ef = if delta_sum == 0.0 {
                0.0
            } else {
                (direction / delta_sum).min(1.0)
            };
            let s = ((0.6667 * ef) + 0.0645).powi(2);
            let kama = s * self.buffer[current_idx] + (1.0 - s) * self.prev_kama;
            self.prev_kama = kama;
            return Some(kama);
        }

        // Continue with adaptive calculation for subsequent values
        // Direction uses full period window
        let direction =
            (self.buffer[current_idx] - self.buffer[current_idx - (self.period - 1)]).abs();

        // Calculate efficiency ratio
        let ef = if delta_sum == 0.0 {
            0.0
        } else {
            (direction / delta_sum).min(1.0)
        };

        // Ehlers smoothing constant
        let s = ((0.6667 * ef) + 0.0645).powi(2);

        // Update KAMA
        let kama = s * self.buffer[current_idx] + (1.0 - s) * self.prev_kama;
        self.prev_kama = kama;

        Some(kama)
    }
}

// Batch processing support
#[derive(Clone, Debug)]
pub struct EhlersKamaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for EhlersKamaBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 50, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EhlersKamaBatchBuilder {
    range: EhlersKamaBatchRange,
    kernel: Kernel,
}

impl EhlersKamaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<EhlersKamaBatchOutput, EhlersKamaError> {
        ehlers_kama_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        s: &str,
    ) -> Result<EhlersKamaBatchOutput, EhlersKamaError> {
        let data = source_type(c, s);
        ehlers_kama_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<EhlersKamaBatchOutput, EhlersKamaError> {
        EhlersKamaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn with_default_candles(c: &Candles) -> Result<EhlersKamaBatchOutput, EhlersKamaError> {
        EhlersKamaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct EhlersKamaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersKamaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersKamaBatchOutput {
    pub fn row_for_params(&self, p: &EhlersKamaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(20) == p.period.unwrap_or(20))
    }

    pub fn values_for(&self, p: &EhlersKamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_periods((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}

#[inline(always)]
pub fn ehlers_kama_batch_slice(
    data: &[f64],
    r: &EhlersKamaBatchRange,
    kern: Kernel,
) -> Result<EhlersKamaBatchOutput, EhlersKamaError> {
    ehlers_kama_batch_inner(data, r, kern, false)
}

#[inline(always)]
pub fn ehlers_kama_batch_par_slice(
    data: &[f64],
    r: &EhlersKamaBatchRange,
    kern: Kernel,
) -> Result<EhlersKamaBatchOutput, EhlersKamaError> {
    ehlers_kama_batch_inner(data, r, kern, true)
}

pub fn ehlers_kama_batch_with_kernel(
    data: &[f64],
    r: &EhlersKamaBatchRange,
    k: Kernel,
) -> Result<EhlersKamaBatchOutput, EhlersKamaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EhlersKamaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    // Map batch to compute kernel
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ehlers_kama_batch_inner(data, r, simd, true)
}

#[inline(always)]
fn ehlers_kama_batch_inner(
    data: &[f64],
    r: &EhlersKamaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EhlersKamaBatchOutput, EhlersKamaError> {
    let len = data.len();
    if len == 0 {
        return Err(EhlersKamaError::AllValuesNaN);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersKamaError::AllValuesNaN)?;
    let periods = expand_periods(r.period);
    if periods.is_empty() {
        return Err(EhlersKamaError::InvalidPeriod {
            period: 0,
            data_len: len,
        });
    }

    // Validate longest period
    let max_p = *periods.iter().max().unwrap();
    if len - first < max_p {
        return Err(EhlersKamaError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = periods.len();
    let cols = len;

    // allocate matrix uninitialized then prefix NaNs per-row
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = periods.iter().map(|&p| first + p - 1).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Prepare slice to f64
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    ehlers_kama_batch_inner_into(data, &periods, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    let combos = periods
        .into_iter()
        .map(|p| EhlersKamaParams { period: Some(p) })
        .collect();

    Ok(EhlersKamaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn ehlers_kama_batch_inner_into(
    data: &[f64],
    periods: &[usize],
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), EhlersKamaError> {
    let len = data.len();
    let cols = len;
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EhlersKamaError::AllValuesNaN)?;

    // resolve once
    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    // map batch → compute kernel (same as alma.rs)
    let compute_kernel = match actual {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        k => k,
    };

    // Shared precompute across rows: prefix sums of absolute diffs to seed ER denominator in O(1)
    // ps[k] = sum_{t=1..k} |data[t]-data[t-1]|, with diffs only when t > first
    let mut ps = vec![0.0f64; len];
    for k in 1..len {
        let ad = if k > first { (data[k] - data[k - 1]).abs() } else { 0.0 };
        ps[k] = ps[k - 1] + ad;
    }

    let do_row = |row: usize, dst_row: &mut [f64]| {
        let p = periods[row];
        let start = first + p - 1;
        if start >= len { return; }

        // Seed using ps in O(1)
        let mut prev_kama = data[start - 1];
        let a0 = data[start];
        // Initial denominator over PERIOD consecutive diffs: ps[start] - ps[first]
        let mut delta_sum = ps[start] - ps[first];
        let dir0 = (a0 - data[start - (p - 1)]).abs();
        let ef0 = if delta_sum == 0.0 { 0.0 } else { (dir0 / delta_sum).min(1.0) };
        let mut s_term = 0.6667f64.mul_add(ef0, 0.0645);
        let mut s = s_term * s_term;
        prev_kama = s.mul_add(a0 - prev_kama, prev_kama);
        dst_row[start] = prev_kama;

        // Main loop: denominator via prefix sums (O(1)); recurrence identical
        for i in (start + 1)..len {
            delta_sum = ps[i] - ps[i - p];
            let a = data[i];
            let dir = (a - data[i - (p - 1)]).abs();
            let ef = if delta_sum == 0.0 { 0.0 } else { (dir / delta_sum).min(1.0) };
            s_term = 0.6667f64.mul_add(ef, 0.0645);
            s = s_term * s_term;
            prev_kama = s.mul_add(a - prev_kama, prev_kama);
            dst_row[i] = prev_kama;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
        #[cfg(target_arch = "wasm32")]
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_kama")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn ehlers_kama_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};
    use pyo3::exceptions::PyValueError;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = EhlersKamaParams {
        period: Some(period),
    };
    let input = EhlersKamaInput::from_slice(slice_in, params);
    let result_vec: Vec<f64> = py
        .allow_threads(|| ehlers_kama_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_kama_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn ehlers_kama_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let periods = expand_periods(period_range);
    let rows = periods.len();
    let cols = slice_in.len();

    // Create an uninitialized NumPy buffer (flat) and compute directly into it.
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    // Warm prefixes: initialize directly in the NumPy memory without extra buffers.
    let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let warm: Vec<usize> = periods.iter().map(|&p| first + p - 1).collect();
    // Reinterpret NumPy memory as MaybeUninit to use init_matrix_prefixes safely.
    let out_mu: &mut [std::mem::MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(
            out_slice.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out_slice.len(),
        )
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    // Resolve kernel exactly like ALMA batch.
    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let resolved = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match resolved {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };
        ehlers_kama_batch_inner_into(slice_in, &periods, simd, true, out_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item("periods", periods.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ehlers_kama_cuda_batch_dev")]
#[pyo3(signature = (data, period_range, device_id=0))]
pub fn ehlers_kama_cuda_batch_dev_py(
    py: Python<'_>,
    data: numpy::PyReadonlyArray1<'_, f64>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data.as_slice()?;
    let sweep = EhlersKamaBatchRange {
        period: period_range,
    };
    let data_f32: Vec<f32> = slice_in.iter().map(|&v| v as f32).collect();

    let inner = py.allow_threads(|| {
        let cuda =
            CudaEhlersKama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.ehlers_kama_batch_dev(&data_f32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ehlers_kama_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn ehlers_kama_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = EhlersKamaParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda =
            CudaEhlersKama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.ehlers_kama_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersKamaStream")]
pub struct EhlersKamaStreamPy {
    stream: EhlersKamaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersKamaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        use pyo3::exceptions::PyValueError;
        let params = EhlersKamaParams {
            period: Some(period),
        };
        Ok(Self {
            stream: EhlersKamaStream::try_new(params)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_kama_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = EhlersKamaParams {
        period: Some(period),
    };
    let input = EhlersKamaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    ehlers_kama_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EhlersKamaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EhlersKamaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersKamaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ehlers_kama_batch)]
pub fn ehlers_kama_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: EhlersKamaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = EhlersKamaBatchRange {
        period: config.period_range,
    };
    let output = ehlers_kama_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = EhlersKamaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_kama_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_kama_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_kama_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ehlers_kama_into"));
    }
    if period == 0 || period > len {
        return Err(JsValue::from_str("Invalid period"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = EhlersKamaParams {
            period: Some(period),
        };
        let input = EhlersKamaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            ehlers_kama_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ehlers_kama_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ehlers_kama_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_kama_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let periods = expand_periods((period_start, period_end, period_step));
        let rows = periods.len();
        let cols = len;

        // treat caller memory as uninit for prefix init
        let out_mu =
            std::slice::from_raw_parts_mut(out_ptr as *mut std::mem::MaybeUninit<f64>, rows * cols);

        // compute per-row warm prefixes once
        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let warm: Vec<usize> = periods.iter().map(|&p| first + p - 1).collect();
        init_matrix_prefixes(out_mu, cols, &warm);

        // compute directly into caller memory
        let out_f64 = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
        ehlers_kama_batch_inner_into(data, &periods, detect_best_kernel(), false, out_f64)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

// ==================== PYTHON MODULE REGISTRATION ====================
#[cfg(feature = "python")]
pub fn register_ehlers_kama_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ehlers_kama_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_kama_batch_py, m)?)?;
    m.add_class::<EhlersKamaStreamPy>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_ehlers_kama_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = EhlersKamaParams { period: None };
        let input = EhlersKamaInput::from_candles(&candles, "close", default_params);
        let output = ehlers_kama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_ehlers_kama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EhlersKamaInput::from_candles(&candles, "close", EhlersKamaParams::default());
        let result = ehlers_kama_with_kernel(&input, kernel)?;

        assert_eq!(result.values.len(), candles.close.len());

        // Check last 5 values match expected (with period=20)
        let expected_last_5 = [
            59721.60663208,
            59717.43599957,
            59708.31467709,
            59704.78675836,
            59701.81308504,
        ];

        // Get last 6 values and check first 5 (due to Pine non-repainting mode alignment)
        let start = result.values.len() - 6;
        for (i, &expected) in expected_last_5.iter().enumerate() {
            let actual = result.values[start + i];
            assert!(
                (actual - expected).abs() < 1e-6,
                "[{}] EKAMA mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected,
                actual
            );
        }

        Ok(())
    }

    fn check_ehlers_kama_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EhlersKamaInput::with_default_candles(&candles);
        match input.data {
            EhlersKamaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EhlersKamaData::Candles"),
        }
        let output = ehlers_kama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_ehlers_kama_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = EhlersKamaParams { period: Some(0) };
        let input = EhlersKamaInput::from_slice(&input_data, params);
        let res = ehlers_kama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EKAMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_kama_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = EhlersKamaParams { period: Some(10) };
        let input = EhlersKamaInput::from_slice(&data_small, params);
        let res = ehlers_kama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EKAMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_kama_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = EhlersKamaParams { period: Some(20) };
        let input = EhlersKamaInput::from_slice(&single_point, params);
        let res = ehlers_kama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EKAMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_kama_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EhlersKamaInput::from_slice(&empty, EhlersKamaParams::default());
        let res = ehlers_kama_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EhlersKamaError::EmptyInputData)),
            "[{}] EKAMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_ehlers_kama_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_with_nan = vec![
            f64::NAN,
            f64::NAN,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
            110.0,
            120.0,
            130.0,
            140.0,
            150.0,
            160.0,
            170.0,
            180.0,
            190.0,
            200.0,
            210.0,
            220.0,
            230.0,
        ];
        let params = EhlersKamaParams { period: Some(5) };
        let input = EhlersKamaInput::from_slice(&data_with_nan, params);
        let result = ehlers_kama_with_kernel(&input, kernel)?;

        // First values should be NaN due to NaN prefix and warmup
        assert!(result.values[0].is_nan());
        assert!(result.values[1].is_nan());

        // After warmup, should have valid values
        let warmup_end = 2 + 5 - 1; // NaN prefix + period - 1
        assert!(
            !result.values[warmup_end + 5].is_nan(),
            "[{}] EKAMA should produce valid values after warmup",
            test_name
        );

        Ok(())
    }

    fn check_ehlers_kama_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = (1..=200).map(|x| x as f64).collect::<Vec<_>>();
        let p = 5;
        let input = EhlersKamaInput::from_slice(&data, EhlersKamaParams { period: Some(p) });
        let batch = ehlers_kama_with_kernel(&input, kernel)?.values;

        let mut s = EhlersKamaStream::try_new(EhlersKamaParams { period: Some(p) })?;
        let mut stream = Vec::with_capacity(data.len());
        for &v in &data {
            stream.push(s.update(v).unwrap_or(f64::NAN));
        }

        assert_eq!(batch.len(), stream.len());
        for (i, (&b, &st)) in batch.iter().zip(&stream).enumerate() {
            if b.is_nan() {
                assert!(st.is_nan(), "[{test_name}] NaN mismatch at {i}");
            } else {
                assert!(
                    (b - st).abs() < 1e-9,
                    "[{test_name}] mismatch at {i}: {b} vs {st}"
                );
            }
        }
        Ok(())
    }

    fn check_ehlers_kama_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = EhlersKamaParams { period: Some(20) };
        let first_input = EhlersKamaInput::from_candles(&candles, "close", first_params);
        let first_result = ehlers_kama_with_kernel(&first_input, kernel)?;

        let second_params = EhlersKamaParams { period: Some(20) };
        let second_input = EhlersKamaInput::from_slice(&first_result.values, second_params);
        let second_result = ehlers_kama_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());

        // Check that we get valid values after double application
        let warmup = 40; // double warmup
        let has_valid = second_result.values[warmup..].iter().any(|&v| !v.is_nan());
        assert!(
            has_valid,
            "[{}] EKAMA reinput should produce valid values",
            test_name
        );

        Ok(())
    }

    fn check_ehlers_kama_into_slice(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0,
            140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
        ];

        let params = EhlersKamaParams { period: Some(5) };
        let input = EhlersKamaInput::from_slice(&data, params);

        // Calculate using normal method
        let normal_result = ehlers_kama_with_kernel(&input, kernel)?;

        // Calculate using into_slice method
        let mut into_result = vec![0.0; data.len()];
        ehlers_kama_into_slice(&mut into_result, &input, kernel)?;

        // Compare results
        for i in 0..data.len() {
            if normal_result.values[i].is_nan() {
                assert!(
                    into_result[i].is_nan(),
                    "[{}] into_slice mismatch at {}: expected NaN",
                    test_name,
                    i
                );
            } else {
                let diff = (normal_result.values[i] - into_result[i]).abs();
                assert!(
                    diff < 1e-10,
                    "[{}] into_slice mismatch at {}: {} vs {}",
                    test_name,
                    i,
                    normal_result.values[i],
                    into_result[i]
                );
            }
        }

        Ok(())
    }

    fn check_ehlers_kama_builder(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let result = EhlersKamaBuilder::new()
            .period(15)
            .kernel(kernel)
            .apply(&candles)?;

        assert_eq!(result.values.len(), candles.close.len());

        // Test with slice
        let data = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0,
            140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
        ];
        let slice_result = EhlersKamaBuilder::new()
            .period(5)
            .kernel(kernel)
            .apply_slice(&data)?;

        assert_eq!(slice_result.values.len(), data.len());

        Ok(())
    }

    // Generate all test variants using macros
    macro_rules! generate_all_ehlers_kama_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    #[cfg(debug_assertions)]
    fn check_ehlers_kama_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test various parameter combinations
        let test_params = vec![
            EhlersKamaParams { period: Some(5) },
            EhlersKamaParams { period: Some(10) },
            EhlersKamaParams { period: Some(20) },
            EhlersKamaParams { period: Some(50) },
            EhlersKamaParams { period: Some(100) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = EhlersKamaInput::from_candles(&candles, "close", params.clone());
            let output = ehlers_kama_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(20)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(20)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(20)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ehlers_kama_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_ehlers_kama_tests!(
        check_ehlers_kama_partial_params,
        check_ehlers_kama_accuracy,
        check_ehlers_kama_default_candles,
        check_ehlers_kama_zero_period,
        check_ehlers_kama_period_exceeds_length,
        check_ehlers_kama_very_small_dataset,
        check_ehlers_kama_empty_input,
        check_ehlers_kama_nan_handling,
        check_ehlers_kama_streaming,
        check_ehlers_kama_reinput,
        check_ehlers_kama_into_slice,
        check_ehlers_kama_builder
    );

    #[cfg(debug_assertions)]
    generate_all_ehlers_kama_tests!(check_ehlers_kama_no_poison);

    // Batch processing tests
    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EhlersKamaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = EhlersKamaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Validate some expected values for default params (period=20)
        // These are example values - actual values will depend on the data
        let expected_last_5 = [
            59721.60663208,
            59717.43599957,
            59708.31467709,
            59704.78675836,
            59701.81308504,
        ];
        // Get last 6 values and check first 5 (due to Pine non-repainting mode alignment)
        let start = row.len() - 6;
        for (i, &v) in row[start..start + 5].iter().enumerate() {
            assert!(
                (v - expected_last_5[i]).abs() < 1e-6,
                "[{test_name}] default-row mismatch at idx {i}: {v} vs {expected_last_5:?}"
            );
        }

        Ok(())
    }

    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EhlersKamaBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 5)
            .apply_candles(&c, "close")?;

        assert_eq!(output.rows, 5); // (30-10)/5 + 1
        assert_eq!(output.cols, c.close.len());

        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (5, 25, 5),   // period range
            (10, 10, 0),  // single period
            (2, 5, 1),    // small range
            (30, 60, 15), // large range
            (8, 12, 1),   // dense range
        ];

        for (cfg_idx, (p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = EhlersKamaBatchBuilder::new()
                .kernel(kernel)
                .period_range(*p_start, *p_end, *p_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];
                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test_name,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(20)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test_name,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(20)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test_name,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(20)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ehlers_kama_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = EhlersKamaParams {
                    period: Some(period),
                };
                let input = EhlersKamaInput::from_slice(&data, params);

                let EhlersKamaOutput { values: out } =
                    ehlers_kama_with_kernel(&input, kernel).unwrap();
                let EhlersKamaOutput { values: ref_out } =
                    ehlers_kama_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property: output length equals input length
                prop_assert_eq!(out.len(), data.len());

                // Property: first period-1 values are NaN
                for i in 0..(period - 1) {
                    prop_assert!(out[i].is_nan());
                }

                // For valid outputs, check range and kernel consistency
                for i in (period - 1)..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // KAMA is an adaptive average, so it doesn't strictly need to be
                    // within the current window range (it can lag behind rapid changes)

                    // Special case: period=1 should equal the data
                    if period == 1 {
                        prop_assert!((y - data[i]).abs() <= f64::EPSILON);
                    }

                    // If data is constant, KAMA should converge to that value
                    if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) {
                        prop_assert!((y - data[0]).abs() <= 1e-6);
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {i}: {y} vs {r}"
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    // Kernels should produce very similar results
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
                    );
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_ehlers_kama_tests!(check_ehlers_kama_property);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_ehlers_kama_simd128_correctness() {
        // Use real CSV data like all other tests
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to read test data");

        // Use default parameters (period=20) for consistency
        let params = EhlersKamaParams::default();
        let input = EhlersKamaInput::from_candles(&candles, "close", params);

        // Compute with scalar version
        let scalar_output = ehlers_kama_with_kernel(&input, Kernel::Scalar).unwrap();

        // Compute with SIMD128 (via Scalar kernel on WASM)
        let simd128_output = ehlers_kama_with_kernel(&input, Kernel::Scalar).unwrap();

        // Validate output length
        assert_eq!(scalar_output.values.len(), simd128_output.values.len());
        assert_eq!(scalar_output.values.len(), candles.close.len());

        // Compare results
        for (i, (scalar_val, simd_val)) in scalar_output
            .values
            .iter()
            .zip(simd128_output.values.iter())
            .enumerate()
        {
            if scalar_val.is_nan() {
                assert!(
                    simd_val.is_nan(),
                    "SIMD128 mismatch at index {}: scalar=NaN, simd128={}",
                    i,
                    simd_val
                );
            } else {
                assert!(
                    (scalar_val - simd_val).abs() < 1e-10,
                    "SIMD128 mismatch at index {}: scalar={}, simd128={}",
                    i,
                    scalar_val,
                    simd_val
                );
            }
        }
    }

    #[test]
    fn test_stream_debug() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let period = 5;

        // Batch calculation
        let params = EhlersKamaParams {
            period: Some(period),
        };
        let input = EhlersKamaInput::from_slice(&data, params.clone());
        let batch_output = ehlers_kama(&input).unwrap();

        // Streaming calculation
        let mut stream = EhlersKamaStream::try_new(params).unwrap();
        let mut stream_results = vec![];

        for (i, &val) in data.iter().enumerate() {
            let result = stream.update(val);
            stream_results.push(result);

            if let Some(kama) = result {
                println!("Stream update at index {}: value={}, kama={}", i, val, kama);

                // Debug the internals
                if i == 4 {
                    println!("  buffer: {:?}", stream.buffer);
                    println!("  prev_kama: {}", stream.prev_kama);
                }
            }
        }

        println!("Batch result: {:?}", batch_output.values);
        println!("Stream results: {:?}", stream_results);

        // Check if they match
        for i in 0..data.len() {
            if batch_output.values[i].is_nan() {
                assert!(stream_results[i].is_none());
            } else {
                assert!(stream_results[i].is_some());
                let diff = (batch_output.values[i] - stream_results[i].unwrap()).abs();
                if diff > 1e-10 {
                    panic!(
                        "Mismatch at index {}: batch={}, stream={}",
                        i,
                        batch_output.values[i],
                        stream_results[i].unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn test_stream_first_output() {
        // Test that streaming produces correct first output
        let data = vec![
            2761.7, 2740.0, 2763.0, 2772.4, 2779.7, 2769.7, 2759.0, 2663.3, 2570.0, 2572.2, 2484.2,
            2560.9, 2508.6, 2481.9, 2538.1, 2432.9, 2469.0, 2527.738, 2545.1, 2536.9,
        ];
        let period = 20;

        // Batch calculation
        let params = EhlersKamaParams {
            period: Some(period),
        };
        let input = EhlersKamaInput::from_slice(&data, params.clone());
        let batch_output = ehlers_kama(&input).unwrap();

        // Streaming calculation
        let mut stream = EhlersKamaStream::try_new(params).unwrap();
        let mut stream_result = None;

        for (i, &val) in data.iter().enumerate() {
            stream_result = stream.update(val);
            if i == 19 {
                // Debug the calculation at index 19
                println!("Stream at index 19:");
                println!("  buffer len: {}", stream.buffer.len());
                println!("  current value: {}", val);
                println!("  prev_kama: {}", stream.prev_kama);

                // Manual calculation to verify
                let mut delta_sum = 0.0;
                for k in 1..20 {
                    delta_sum += (stream.buffer[k] - stream.buffer[k - 1]).abs();
                }
                println!("  delta_sum: {}", delta_sum);

                let direction = (stream.buffer[19] - stream.buffer[0]).abs();
                println!("  direction: {}", direction);

                let ef = if delta_sum == 0.0 {
                    0.0
                } else {
                    (direction / delta_sum).min(1.0)
                };
                println!("  ef: {}", ef);

                let s = ((0.6667 * ef) + 0.0645).powi(2);
                println!("  s: {}", s);

                let expected_kama = s * stream.buffer[19] + (1.0 - s) * stream.buffer[18];
                println!("  expected_kama: {}", expected_kama);

                if let Some(result) = stream_result {
                    println!("  actual result: {}", result);
                }
            }
        }

        println!("Batch at index 19: {}", batch_output.values[19]);

        // Check they match
        assert!(stream_result.is_some());
        let diff = (batch_output.values[19] - stream_result.unwrap()).abs();
        assert!(
            diff < 1e-10,
            "First output mismatch: batch={}, stream={}",
            batch_output.values[19],
            stream_result.unwrap()
        );
    }
}
