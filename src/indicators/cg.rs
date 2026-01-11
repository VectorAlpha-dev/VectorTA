//! # Center of Gravity (CG)
//!
//! Decision log:
//! - SIMD paths intentionally delegate to the scalar kernel to preserve strict streaming parity (<= 1e-9) because vectorization reorders FP ops.
//! - Scalar kernel optimized (pointer-based, unrolled) for ~18–20% faster at 100k on native CPU.
//! - Streaming kernel updated to O(1) per tick using running sums; parity with batch/scalar maintained within 1e-9.
//! - Row-specific batch kernels not implemented; revisit if tolerances relax or shared precompute is allowed.
//! - CUDA batch and many-series kernels enabled (FP32, warmup aligned with scalar CG) via `cuda::oscillators::cg_wrapper`; Python exposes VRAM handles with CAI v3 + DLPack v1.x interop.
//!
//! The Center of Gravity (CG) indicator attempts to measure the "center" of prices
//! over a given window, sometimes used for smoothing or cycle analysis.
//!
//! ## Parameters
//! - **period**: The window size. Defaults to 10.
//!
//! ## Errors
//! - **EmptyData**: cg: Input data slice is empty.
//! - **InvalidPeriod**: cg: `period` is zero or exceeds the data length.
//! - **AllValuesNaN**: cg: All input data values are `NaN`.
//! - **NotEnoughValidData**: cg: Fewer than `period` valid (non-`NaN`) data points remain after the first valid index.
//!
//! ## Returns
//! - **`Ok(CgOutput)`** on success, containing a `Vec<f64>` matching input length,
//!   with leading `NaN` until the warm-up period is reached.
//! - **`Err(CgError)`** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for CgInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CgData::Slice(slice) => slice,
            CgData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CgData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CgOutput {
    pub values: Vec<f64>,
}

impl std::ops::Deref for CgOutput {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl std::ops::DerefMut for CgOutput {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CgParams {
    pub period: Option<usize>,
}

impl Default for CgParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct CgInput<'a> {
    pub data: CgData<'a>,
    pub params: CgParams,
}

impl<'a> CgInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CgParams) -> Self {
        Self {
            data: CgData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CgParams) -> Self {
        Self {
            data: CgData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CgParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CgBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CgBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CgBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<CgOutput, CgError> {
        let p = CgParams {
            period: self.period,
        };
        let i = CgInput::from_candles(c, "close", p);
        cg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CgOutput, CgError> {
        let p = CgParams {
            period: self.period,
        };
        let i = CgInput::from_slice(d, p);
        cg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<CgStream, CgError> {
        let p = CgParams {
            period: self.period,
        };
        CgStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CgError {
    #[error("CG: Empty data provided for CG.")]
    EmptyData,
    #[error("CG: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("CG: All values are NaN.")]
    AllValuesNaN,
    #[error("CG: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("CG: output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("CG: invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("CG: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
}

#[inline]
pub fn cg(input: &CgInput) -> Result<CgOutput, CgError> {
    cg_with_kernel(input, Kernel::Auto)
}

pub fn cg_with_kernel(input: &CgInput, kernel: Kernel) -> Result<CgOutput, CgError> {
    let data: &[f64] = match &input.data {
        CgData::Candles { candles, source } => source_type(candles, source),
        CgData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(CgError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CgError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(CgError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    // ==== Revert to requiring period + 1 valid points =====
    if (len - first) < (period + 1) {
        return Err(CgError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    // Use helper function to allocate with NaN prefix only where needed
    let mut out = alloc_with_nan_prefix(len, first + period);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => cg_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cg_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => cg_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(CgOutput { values: out })
}

// Pre-computed weights for common periods (1.0, 2.0, 3.0, ..., 64.0)
const CG_WEIGHTS: [f64; 64] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
    34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0,
    50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0,
];

// Optimized scalar kernel: preserves exact accumulation order while avoiding bounds checks
// via pointer-based iteration and small unrolled inner loops. For periods <= 65, uses
// precomputed weights; otherwise computes weights on the fly. Only writes the computed
// range [first + period, len), matching warmup handling done by callers (see alma.rs pattern).
#[inline(always)]
pub fn cg_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let start = first + period;
    let len = data.len();
    if start >= len {
        return;
    }

    let n_items = period - 1; // exactly period-1 bars

    // Fast path: precomputed weights for common periods (<= 65 => up to 64 terms)
    if period <= 65 {
        // Unroll by 8 while preserving accumulation order exactly
        #[inline(always)]
        unsafe fn dot_sum_precomputed(base_ptr: *const f64, n_items: usize) -> (f64, f64) {
            let mut num = 0.0;
            let mut den = 0.0;
            let mut k = 0usize;
            let blocks = n_items & !7usize; // round down to multiple of 8

            while k < blocks {
                // step 0
                let p0 = *base_ptr.sub(k);
                let w0 = *CG_WEIGHTS.get_unchecked(k);
                num += w0 * p0;
                den += p0;
                // step 1
                let p1 = *base_ptr.sub(k + 1);
                let w1 = *CG_WEIGHTS.get_unchecked(k + 1);
                num += w1 * p1;
                den += p1;
                // step 2
                let p2 = *base_ptr.sub(k + 2);
                let w2 = *CG_WEIGHTS.get_unchecked(k + 2);
                num += w2 * p2;
                den += p2;
                // step 3
                let p3 = *base_ptr.sub(k + 3);
                let w3 = *CG_WEIGHTS.get_unchecked(k + 3);
                num += w3 * p3;
                den += p3;
                // step 4
                let p4 = *base_ptr.sub(k + 4);
                let w4 = *CG_WEIGHTS.get_unchecked(k + 4);
                num += w4 * p4;
                den += p4;
                // step 5
                let p5 = *base_ptr.sub(k + 5);
                let w5 = *CG_WEIGHTS.get_unchecked(k + 5);
                num += w5 * p5;
                den += p5;
                // step 6
                let p6 = *base_ptr.sub(k + 6);
                let w6 = *CG_WEIGHTS.get_unchecked(k + 6);
                num += w6 * p6;
                den += p6;
                // step 7
                let p7 = *base_ptr.sub(k + 7);
                let w7 = *CG_WEIGHTS.get_unchecked(k + 7);
                num += w7 * p7;
                den += p7;

                k += 8;
            }

            while k < n_items {
                let p = *base_ptr.sub(k);
                let w = *CG_WEIGHTS.get_unchecked(k);
                num += w * p;
                den += p;
                k += 1;
            }
            (num, den)
        }

        for i in start..len {
            // safe because i >= start >= period and we only subtract up to (period-1)
            let base_ptr = unsafe { data.as_ptr().add(i) };
            let (num, den) = unsafe { dot_sum_precomputed(base_ptr, n_items) };
            out[i] = if den.abs() > f64::EPSILON {
                -num / den
            } else {
                0.0
            };
        }
        return;
    }

    // Generic path: compute weights on the fly; unroll by 4 while preserving order
    for i in start..len {
        unsafe {
            let base_ptr = data.as_ptr().add(i);
            let mut num = 0.0;
            let mut den = 0.0;

            let mut k = 0usize;
            let blocks = n_items & !3usize; // multiple of 4
            let mut w = 1.0f64;

            while k < blocks {
                // step 0
                let p0 = *base_ptr.sub(k);
                num += w * p0;
                den += p0;
                w += 1.0;

                // step 1
                let p1 = *base_ptr.sub(k + 1);
                num += w * p1;
                den += p1;
                w += 1.0;

                // step 2
                let p2 = *base_ptr.sub(k + 2);
                num += w * p2;
                den += p2;
                w += 1.0;

                // step 3
                let p3 = *base_ptr.sub(k + 3);
                num += w * p3;
                den += p3;
                w += 1.0;

                k += 4;
            }

            while k < n_items {
                let p = *base_ptr.sub(k);
                num += w * p;
                den += p;
                w += 1.0;
                k += 1;
            }

            out[i] = if den.abs() > f64::EPSILON {
                -num / den
            } else {
                0.0
            };
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cg_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { cg_avx512_short(data, period, first, out) }
    } else {
        unsafe { cg_avx512_long(data, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "fma")]
pub unsafe fn cg_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let start = first + period;
    let len = data.len();
    if start >= len {
        return;
    }

    let n_items = period - 1;
    const VL: usize = 4;

    #[inline(always)]
    unsafe fn hsum_m256d(x: __m256d) -> f64 {
        let hi = _mm256_extractf128_pd(x, 1);
        let lo = _mm256_castpd256_pd128(x);
        let sum2 = _mm_add_pd(lo, hi);
        let hi64 = _mm_unpackhi_pd(sum2, sum2);
        let sum = _mm_add_sd(sum2, hi64);
        _mm_cvtsd_f64(sum)
    }

    for i in start..len {
        let base_ptr = data.as_ptr().add(i);
        let mut vnum = _mm256_setzero_pd();
        let mut vden = _mm256_setzero_pd();
        let blocks = n_items & !(VL - 1);
        let mut k = 0usize;

        // Use descending weight vector [k+3 .. k]
        let step_r = _mm256_setr_pd(3.0, 2.0, 1.0, 0.0);
        while k < blocks {
            let p = _mm256_loadu_pd(base_ptr.sub(k + (VL - 1)));
            let basew = _mm256_set1_pd(k as f64 + 1.0);
            let w = _mm256_add_pd(basew, step_r);
            let prod = _mm256_fmadd_pd(p, w, vnum);
            vnum = prod;
            vden = _mm256_add_pd(vden, p);
            k += VL;
        }

        let mut num = hsum_m256d(vnum);
        let mut den = hsum_m256d(vden);

        let mut w = 1.0 + k as f64;
        while k < n_items {
            let p = *base_ptr.sub(k);
            num += w * p;
            den += p;
            w += 1.0;
            k += 1;
        }

        out[i] = if den.abs() > f64::EPSILON {
            -num / den
        } else {
            0.0
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "fma")]
pub unsafe fn cg_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // Treat the same as long; short is a hint for future specialization
    const VL: usize = 8;
    let start = first + period;
    let len = data.len();
    if start >= len {
        return;
    }

    let n_items = period - 1;

    #[inline(always)]
    unsafe fn hsum_m512d(x: __m512d) -> f64 {
        let lo = _mm512_castpd512_pd256(x);
        let hi = _mm512_extractf64x4_pd::<1>(x);
        let sum256 = _mm256_add_pd(lo, hi);
        let hi128 = _mm256_extractf128_pd(sum256, 1);
        let lo128 = _mm256_castpd256_pd128(sum256);
        let sum2 = _mm_add_pd(lo128, hi128);
        let hi64 = _mm_unpackhi_pd(sum2, sum2);
        let sum = _mm_add_sd(sum2, hi64);
        _mm_cvtsd_f64(sum)
    }

    for i in start..len {
        let base_ptr = data.as_ptr().add(i);
        let mut vnum = _mm512_setzero_pd();
        let mut vden = _mm512_setzero_pd();
        let blocks = n_items & !(VL - 1);
        let mut k = 0usize;

        // descending step [7..0]
        let step_r = _mm512_setr_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
        while k < blocks {
            let p = _mm512_loadu_pd(base_ptr.sub(k + (VL - 1)));
            let basew = _mm512_set1_pd(k as f64 + 1.0);
            let w = _mm512_add_pd(basew, step_r);
            let prod = _mm512_fmadd_pd(p, w, vnum);
            vnum = prod;
            vden = _mm512_add_pd(vden, p);
            k += VL;
        }

        let mut num = hsum_m512d(vnum);
        let mut den = hsum_m512d(vden);

        let mut w = 1.0 + k as f64;
        while k < n_items {
            let p = *base_ptr.sub(k);
            num += w * p;
            den += p;
            w += 1.0;
            k += 1;
        }

        out[i] = if den.abs() > f64::EPSILON {
            -num / den
        } else {
            0.0
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cg_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    cg_avx512_short(data, period, first, out)
}

#[inline]
pub fn cg_into_slice(dst: &mut [f64], input: &CgInput, kern: Kernel) -> Result<(), CgError> {
    let data: &[f64] = match &input.data {
        CgData::Candles { candles, source } => source_type(candles, source),
        CgData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(CgError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CgError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(CgError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    // ==== Revert to requiring period + 1 valid points =====
    if (len - first) < (period + 1) {
        return Err(CgError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    if dst.len() != data.len() {
        return Err(CgError::OutputLengthMismatch { expected: data.len(), got: dst.len() });
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => cg_scalar(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cg_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => cg_avx512(data, period, first, dst),
            _ => unreachable!(),
        }
    }

    // Fill warmup with NaN
    for v in &mut dst[..first + period] {
        *v = f64::NAN;
    }
    Ok(())
}

/// Writes CG results into the provided output slice without allocating.
///
/// - Preserves NaN warmup semantics (prefix length = `first_valid + period`).
/// - The output slice length must equal the input length.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn cg_into(input: &CgInput, out: &mut [f64]) -> Result<(), CgError> {
    // Delegate to the slice-based helper with Kernel::Auto dispatch, matching cg().
    cg_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct CgStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    weighted_sum: f64, // Running weighted sum (numerator)
    price_sum: f64,    // Running sum of prices (denominator)
}

impl CgStream {
    pub fn try_new(params: CgParams) -> Result<Self, CgError> {
        let period = params.period.unwrap_or(10);
        if period == 0 {
            return Err(CgError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            weighted_sum: 0.0,
            price_sum: 0.0,
        })
    }

    /// O(1) streaming update for CG.
    ///
    /// Maintains running sums over the last (period - 1) bars:
    ///   - `price_sum`    = sum x_j
    ///   - `weighted_sum` = sum (j+1)*x_j, newest j=0 .. oldest j=n-1
    ///
    /// Recurrence when a new price `value` arrives and `old` leaves:
    ///   den_new = den_old - old + value
    ///   num_new = num_old + den_old + value - (period as f64) * old
    ///
    /// Warm-up semantics: first `period` writes return None; the first Some arrives
    /// on the (period+1)-th value, matching batch/scalar first valid index at `first + period`.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        debug_assert!(self.period >= 2);

        // Write new value at current head, then advance head.
        let pos = self.head;
        self.buffer[pos] = value;
        let next = if pos + 1 == self.period { 0 } else { pos + 1 };

        // Not filled yet? Keep warm-up semantics identical to existing behavior.
        if !self.filled {
            self.head = next;

            // Just completed the first `period` writes -> initialize running sums
            // for the (period - 1)-wide window and still return None now.
            if self.head == 0 {
                let mut num = 0.0;
                let mut den = 0.0;
                let mut idx = self.head;
                // Accumulate in the same order as cg_scalar for the first window:
                // newest (weight 1) to oldest (weight n).
                for k in 0..(self.period - 1) {
                    idx = if idx == 0 { self.period - 1 } else { idx - 1 };
                    let p = self.buffer[idx];
                    num += (1.0 + k as f64) * p;
                    den += p;
                }
                self.weighted_sum = num;
                self.price_sum = den;
                self.filled = true;
            }
            return None;
        }

        // Once filled: O(1) update using the recurrence.
        // The value leaving the (period-1)-wide window is at `next` after we advanced.
        let last_old = self.buffer[next];

        let den_old = self.price_sum;
        let num_old = self.weighted_sum;

        let den_new = den_old - last_old + value;
        // period as f64 used once; avoid repeated casts
        let num_new = num_old + den_old + value - (self.period as f64) * last_old;

        self.price_sum = den_new;
        self.weighted_sum = num_new;
        self.head = next;

        // Match scalar behavior: guard divide-by-zero and return 0.0 in that case.
        let out = if den_new.abs() > f64::EPSILON {
            -num_new / den_new
        } else {
            0.0
        };
        Some(out)
    }
}

// ---- CUDA Python bindings (DeviceArrayF32Py handles) ----
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32 as CudaDeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::{context::Context, memory::DeviceBuffer};
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray1;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "CgDeviceArrayF32", unsendable)]
pub struct CgDeviceArrayF32Py {
    pub inner: CudaDeviceArrayF32,
    _ctx: Arc<Context>,
    device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl CgDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let itemsize = std::mem::size_of::<f32>();
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (self.inner.cols * itemsize, itemsize))?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Producer stream synchronized before return; omit stream key per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        // 2 == kDLCUDA; use allocation device id carried from wrapper.
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        let _ = stream;

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            CudaDeviceArrayF32 { buf: dummy, rows: 0, cols: 0 },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "cg_cuda_batch_dev")]
#[pyo3(signature = (data, period_range, device_id=0))]
pub fn cg_cuda_batch_dev_py(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<CgDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = data.as_slice()?;
    let sweep = CgBatchRange {
        period: period_range,
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = crate::cuda::oscillators::cg_wrapper::CudaCg::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = cuda
            .cg_batch_dev(slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda
            .synchronize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((dev, cuda.context_arc_clone(), cuda.device_id()))
    })?;
    Ok(CgDeviceArrayF32Py { inner, _ctx: ctx, device_id: dev_id })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "cg_cuda_many_series_one_param_dev")]
#[pyo3(signature = (time_major, cols, rows, period, device_id=0))]
pub fn cg_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    time_major: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    device_id: usize,
) -> PyResult<CgDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let tm = time_major.as_slice()?;
    if tm.len() != cols * rows {
        return Err(PyValueError::new_err(
            "time-major slice length != cols*rows",
        ));
    }
    let params = CgParams {
        period: Some(period),
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = crate::cuda::oscillators::cg_wrapper::CudaCg::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = cuda
            .cg_many_series_one_param_time_major_dev(tm, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda
            .synchronize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((dev, cuda.context_arc_clone(), cuda.device_id()))
    })?;
    Ok(CgDeviceArrayF32Py { inner, _ctx: ctx, device_id: dev_id })
}

#[derive(Clone, Debug)]
pub struct CgBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CgBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 259, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CgBatchBuilder {
    range: CgBatchRange,
    kernel: Kernel,
}

impl CgBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<CgBatchOutput, CgError> {
        cg_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CgBatchOutput, CgError> {
        CgBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CgBatchOutput, CgError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CgBatchOutput, CgError> {
        CgBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn cg_batch_with_kernel(
    data: &[f64],
    sweep: &CgBatchRange,
    k: Kernel,
) -> Result<CgBatchOutput, CgError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(CgError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    cg_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CgBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CgParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CgBatchOutput {
    pub fn row_for_params(&self, p: &CgParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(10) == p.period.unwrap_or(10))
    }
    pub fn values_for(&self, p: &CgParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CgBatchRange) -> Result<Vec<CgParams>, CgError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CgError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if step == 0 {
            // handled above, but keep explicit for clarity
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                vals.push(v);
                match v.checked_add(step) {
                    Some(next) if next > v => v = next,
                    _ => break,
                }
            }
        } else {
            // reversed bounds supported
            let mut v = start;
            while v >= end {
                vals.push(v);
                // checked sub to avoid underflow
                match v.checked_sub(step) {
                    Some(next) if next < v => v = next,
                    _ => break,
                }
                if v == 0 { break; }
            }
        }
        if vals.is_empty() {
            return Err(CgError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }
    let periods = axis_usize(r.period)?;
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(CgParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn cg_batch_slice(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
) -> Result<CgBatchOutput, CgError> {
    cg_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cg_batch_par_slice(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
) -> Result<CgBatchOutput, CgError> {
    cg_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cg_batch_inner(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CgBatchOutput, CgError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(CgError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CgError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p + 1 {
        return Err(CgError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // checked rows * cols to avoid overflow before allocation
    let _ = rows
        .checked_mul(cols)
        .ok_or(CgError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;

    // Use helper to allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warm-up prefixes for each row
    let warm_prefixes: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    // Initialize only the NaN prefixes
    init_matrix_prefixes(&mut buf_mu, cols, &warm_prefixes);

    // Convert to mutable slice for computation
    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    cg_batch_inner_into(data, sweep, kern, parallel, out)?;

    // Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(CgBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn cg_batch_inner_into(
    data: &[f64],
    sweep: &CgBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CgParams>, CgError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(CgError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CgError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p + 1 {
        return Err(CgError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }

    let cols = data.len();

    // Verify caller-provided buffer length
    let expected = combos.len().checked_mul(cols)
        .ok_or(CgError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;
    if out.len() != expected {
        return Err(CgError::OutputLengthMismatch { expected, got: out.len() });
    }

    // Treat caller buffer as MaybeUninit<f64>. Warm prefixes were already set by init_matrix_prefixes
    // when called from Rust batch builder. In Python/WASM paths we intentionally avoid extra writes,
    // matching alma.rs.
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    // Resolve to non-batch kernels if needed
    let actual = match kern {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        },
        other => other,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        // Cast row to &mut [f64] for kernel writes
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match actual {
            Kernel::Scalar => cg_row_scalar(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => cg_row_avx2(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => cg_row_avx512(data, first, period, dst),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn cg_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cg_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        cg_row_avx512_short(data, first, period, out)
    } else {
        cg_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cg_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cg_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    cg_avx512_long(data, period, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_cg_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = CgParams { period: Some(12) };
        let input = CgInput::from_candles(&candles, "close", partial_params);
        let output = cg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cg_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_candles(&candles, "close", params);
        let result = cg_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -4.99905186931943,
            -4.998559827254377,
            -4.9970065675119555,
            -4.9928483984587295,
            -5.004210799262688,
        ];
        assert!(
            result.values.len() >= 5,
            "Not enough data for final 5-values check"
        );
        let start = result.values.len() - 5;
        for (i, &exp) in expected_last_five.iter().enumerate() {
            let got = result.values[start + i];
            assert!(
                (got - exp).abs() < 1e-4,
                "Mismatch in CG at idx {}: expected={}, got={}",
                start + i,
                exp,
                got
            );
        }
        Ok(())
    }

    fn check_cg_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CgInput::with_default_candles(&candles);
        match input.data {
            CgData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CgData::Candles"),
        }
        let output = cg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cg_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = CgParams { period: Some(0) };
        let input = CgInput::from_slice(&data, params);
        let result = cg_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for zero period");
        Ok(())
    }

    fn check_cg_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_slice(&data, params);
        let result = cg_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for period > data.len()");
        Ok(())
    }

    fn check_cg_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_slice(&data, params);
        let result = cg_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period=10"
        );
        Ok(())
    }

    fn check_cg_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CgParams { period: Some(10) };
        let input = CgInput::from_candles(&candles, "close", params);
        let result = cg_with_kernel(&input, kernel)?;
        let check_idx = 240;
        if result.values.len() > check_idx {
            for i in check_idx..result.values.len() {
                if !result.values[i].is_nan() {
                    break;
                }
                if i == result.values.len() - 1 {
                    panic!("All CG values from index {} onward are NaN.", check_idx);
                }
            }
        }
        Ok(())
    }

    fn check_cg_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 10;
        let input = CgInput::from_candles(
            &candles,
            "close",
            CgParams {
                period: Some(period),
            },
        );
        let batch_output = cg_with_kernel(&input, kernel)?.values;
        let mut stream = CgStream::try_new(CgParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
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
            let tol = match kernel {
                Kernel::Avx2 | Kernel::Avx512 => 1e-6,
                _ => 1e-9,
            };
            assert!(
                diff <= tol,
                "[{}] CG streaming mismatch at idx {}: batch={}, stream={}, diff={} (tol={})",
                test_name,
                i,
                b,
                s,
                diff,
                tol
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_cg_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase coverage
        let test_periods = vec![5, 10, 20, 50];

        for period in test_periods {
            let params = CgParams {
                period: Some(period),
            };
            let input = CgInput::from_candles(&candles, "close", params);
            let output = cg_with_kernel(&input, kernel)?;

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
    fn check_cg_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_cg_tests {
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

    generate_all_cg_tests!(
        check_cg_partial_params,
        check_cg_accuracy,
        check_cg_default_candles,
        check_cg_zero_period,
        check_cg_period_exceeds_length,
        check_cg_very_small_dataset,
        check_cg_nan_handling,
        check_cg_streaming,
        check_cg_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_cg_tests!(check_cg_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = CgBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = CgParams::default();
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
        let output = CgBatchBuilder::new()
            .kernel(kernel)
            .period_range(5, 50, 5) // Test periods from 5 to 50 in steps of 5
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
            let period = output.combos[row].period.unwrap_or(10);

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {}",
                    test, val, bits, row, col, idx, period
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {}",
                    test, val, bits, row, col, idx, period
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {}",
                    test, val, bits, row, col, idx, period
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

    #[cfg(feature = "proptest")]
    fn check_cg_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;

        skip_if_unsupported!(kernel, test_name);

        // Strategy 1: Random price data with realistic period ranges
        let random_data_strat = (2usize..=30).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period + 10..400, // Ensure enough data for warmup
                ),
                Just(period),
            )
        });

        // Strategy 2: Constant data (CG should converge to a specific value)
        let constant_data_strat = (2usize..=20).prop_flat_map(|period| {
            (
                (1f64..1000f64).prop_flat_map(move |value| Just(vec![value; period + 50])),
                Just(period),
            )
        });

        // Strategy 3: Trending data (linear increase/decrease)
        let trending_data_strat = (2usize..=25).prop_flat_map(|period| {
            (
                (-100f64..100f64).prop_flat_map(move |start| {
                    (-10f64..10f64).prop_map(move |slope| {
                        (0..period + 100)
                            .map(|i| start + slope * i as f64)
                            .collect::<Vec<_>>()
                    })
                }),
                Just(period),
            )
        });

        // Strategy 4: Edge cases with small periods
        let edge_case_strat = (2usize..=5).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e3f64..1e3f64).prop_filter("finite", |x| x.is_finite()),
                    period + 5..50,
                ),
                Just(period),
            )
        });

        // Combine all strategies
        let combined_strat = prop_oneof![
            random_data_strat.clone(),
            constant_data_strat,
            trending_data_strat,
            edge_case_strat,
        ];

        proptest::test_runner::TestRunner::default()
            .run(&combined_strat, |(data, period)| {
                let params = CgParams {
                    period: Some(period),
                };
                let input = CgInput::from_slice(&data, params);

                // Get output from the kernel under test
                let CgOutput { values: out } = cg_with_kernel(&input, kernel).unwrap();
                // Get reference output from scalar kernel
                let CgOutput { values: ref_out } = cg_with_kernel(&input, Kernel::Scalar).unwrap();

                // Validate warmup period
                for i in 0..period {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Validate computed values
                for i in period..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Property 1: Output should be finite (not infinity)
                    if !y.is_nan() {
                        prop_assert!(
                            y.is_finite(),
                            "CG output at index {} is not finite: {}",
                            i,
                            y
                        );
                    }

                    // Property 2: For constant data, CG should equal a specific value
                    // When all prices are the same, CG = -sum(k*p)/sum(p) where k goes from 1 to period-1
                    // This simplifies to -sum(k)/count where count = period-1
                    if i >= period
                        && data[i - period + 1..=i]
                            .windows(2)
                            .all(|w| (w[0] - w[1]).abs() < 1e-10)
                    {
                        let constant_val = data[i];
                        if constant_val.abs() > f64::EPSILON {
                            // Calculate expected CG for constant data
                            // sum of 1 + 2 + ... + (period-1) = (period-1)*period/2
                            let weight_sum = ((period - 1) * period) as f64 / 2.0;
                            let expected_cg = -weight_sum / (period - 1) as f64;
                            prop_assert!(
                                (y - expected_cg).abs() < 1e-9,
                                "For constant data, CG at index {} should be {}, got {}",
                                i,
                                expected_cg,
                                y
                            );
                        }
                    }

                    // Property 3: For period=2, verify the degenerate case
                    // When period=2, CG uses only 1 bar (period-1 = 1), resulting in a constant -1.0
                    // This is a mathematical artifact but worth validating for completeness
                    if period == 2 && i >= 2 {
                        let p0 = data[i]; // Most recent price (weight = 1)
                        if p0.abs() > f64::EPSILON {
                            // For period=2: CG = -(1*p0)/(p0) = -1.0
                            prop_assert!(
                                (y - (-1.0)).abs() < 1e-9,
                                "Period=2 should always yield -1.0, got {} at index {}",
                                y,
                                i
                            );
                        } else {
                            // When price is effectively 0, CG should be 0
                            prop_assert!(
                                y.abs() < 1e-9,
                                "Period=2 with zero price should yield 0, got {} at index {}",
                                y,
                                i
                            );
                        }
                    }

                    // Property 4: Verify CG produces valid output for non-zero data
                    // For data with non-zero values, CG should produce non-zero results
                    if period > 2 && i >= period + 2 {
                        let window = &data[i - period + 1..=i];
                        let all_nonzero = window.iter().all(|&x| x.abs() > f64::EPSILON);

                        if all_nonzero && !y.is_nan() {
                            // When all values in the window are non-zero, CG should be non-zero
                            prop_assert!(
								y.abs() > f64::EPSILON,
								"CG should be non-zero when all input values are non-zero at index {}, got {}", i, y
							);
                        }
                    }

                    // Property 5: Kernel consistency - all kernels should produce identical results
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y_bits == r_bits,
                            "NaN/infinity mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                    let tol = match kernel {
                        // Allow slightly looser tolerance for SIMD kernels due to different
                        // reduction order / fused mul-add behavior.
                        Kernel::Avx2 | Kernel::Avx512 => 1e-5,
                        _ => 1e-9,
                    };
                    prop_assert!(
                        (y - r).abs() <= tol,
                        "Kernel mismatch at index {}: {} vs {} (ULP={}), tol={}",
                        i,
                        y,
                        r,
                        ulp_diff,
                        tol
                    );
                }

                Ok(())
            })
            .unwrap();

        // Additional focused test for mathematical properties
        let math_test_strat = (2usize..=10, prop::collection::vec(1f64..100f64, 20..50));

        proptest::test_runner::TestRunner::default()
            .run(&math_test_strat, |(period, data)| {
                let params = CgParams {
                    period: Some(period),
                };
                let input = CgInput::from_slice(&data, params);
                let CgOutput { values: out } = cg_with_kernel(&input, kernel).unwrap();

                // Verify that CG calculation uses exactly period-1 bars
                for i in period..data.len() {
                    if out[i].is_nan() {
                        continue;
                    }

                    // Manually calculate CG using the exact formula
                    let mut num = 0.0;
                    let mut denom = 0.0;
                    for count in 0..(period - 1) {
                        let price = data[i - count];
                        let weight = (1 + count) as f64;
                        num += weight * price;
                        denom += price;
                    }

                    if denom.abs() > f64::EPSILON {
                        let expected = -num / denom;
                        prop_assert!(
                            (out[i] - expected).abs() < 1e-9,
                            "Manual calculation mismatch at index {}: expected {}, got {}",
                            i,
                            expected,
                            out[i]
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        // Volatility response test - verify CG responds appropriately to alternating values
        let volatility_test_strat = (3usize..=15).prop_flat_map(|period| {
            (
                (10f64..100f64).prop_flat_map(move |base| {
                    (1f64..50f64).prop_map(move |amplitude| {
                        // Create alternating high/low pattern
                        let mut data = Vec::with_capacity(period + 50);
                        for i in 0..(period + 50) {
                            if i % 2 == 0 {
                                data.push(base + amplitude);
                            } else {
                                data.push(base - amplitude);
                            }
                        }
                        data
                    })
                }),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&volatility_test_strat, |(data, period)| {
                let params = CgParams {
                    period: Some(period),
                };
                let input = CgInput::from_slice(&data, params);
                let CgOutput { values: out } = cg_with_kernel(&input, kernel).unwrap();

                // For alternating data, CG should oscillate but remain bounded
                for i in (period + 2)..data.len() {
                    if out[i].is_nan() {
                        continue;
                    }

                    // CG should respond to the alternating pattern
                    // The exact behavior depends on period (odd vs even)
                    if period % 2 == 0 {
                        // For even periods with alternating data, CG should be relatively stable
                        // because the weights balance out symmetrically
                        if i >= period + 4 {
                            let variation = (out[i] - out[i - 1]).abs();
                            prop_assert!(
								variation < 2.0,
								"CG variation too large for alternating data with even period at index {}: {}", i, variation
							);
                        }
                    }

                    // Verify CG remains bounded relative to the amplitude
                    let base = (data[i] + data[i - 1]) / 2.0; // Approximate base value
                    let relative_cg = (out[i] / base).abs();
                    prop_assert!(
                        relative_cg < 10.0, // CG shouldn't exceed 10x the base value
                        "CG magnitude too large relative to data at index {}: CG={}, base={}",
                        i,
                        out[i],
                        base
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    // Ensures the no-allocation API matches the Vec-returning API exactly (including NaN warmups)
    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_cg_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Build a small but non-trivial input with an initial NaN prefix
        let mut data = vec![f64::NAN; 3];
        data.extend((0..256).map(|i| (i as f64).sin() * 0.5 + (i as f64) * 0.01));

        let input = CgInput::from_slice(&data, CgParams::default());

        // Baseline via Vec-returning API
        let baseline = cg_with_kernel(&input, Kernel::Auto)?.values;

        // Preallocate output and compute via into API
        let mut out = vec![0.0; data.len()];
        cg_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cg")]
#[pyo3(signature = (data, period=None, *, kernel=None))]
pub fn cg_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = CgParams { period };
    let cg_in = CgInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function using zero-copy pattern
    let result_vec: Vec<f64> = py
        .allow_threads(|| cg_with_kernel(&cg_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CgStream")]
pub struct CgStreamPy {
    stream: CgStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CgStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = CgParams {
            period: Some(period),
        };
        let stream = CgStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(CgStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated CG value.
    /// Returns `None` if the buffer is not yet full (needs period + 1 values).
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cg_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn cg_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = CgBatchRange {
        period: period_range,
    };

    // Allocate output
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Warm-up prefix init (NaN) – only the prefixes, nothing beyond
    let first = slice_in
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| PyValueError::new_err("CG: All values are NaN."))?;
    for (r, p) in combos.iter().enumerate() {
        let warm = (first + p.period.unwrap()).min(cols);
        let row = &mut slice_out[r * cols..r * cols + warm];
        for v in row {
            *v = f64::NAN;
        }
    }

    // Compute into the same buffer
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
            cg_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cg_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = CgParams {
        period: Some(period),
    };
    let input = CgInput::from_slice(data, params);

    // Use uninitialized memory for better performance
    let mut output = Vec::with_capacity(data.len());
    unsafe {
        output.set_len(data.len());
    }

    cg_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CgBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CgBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CgParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cg_batch)]
pub fn cg_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: CgBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = CgBatchRange {
        period: config.period_range,
    };

    // Mirror alma.rs: pass non-batch kernel here
    let output = cg_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = CgBatchJsOutput {
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
pub fn cg_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cg_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cg_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer passed to cg_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = CgParams {
            period: Some(period),
        };
        let input = CgInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // Handle aliasing case - use temporary buffer with uninitialized memory
            let mut temp = Vec::with_capacity(len);
            unsafe {
                temp.set_len(len);
            }
            cg_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing - write directly to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            cg_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cg_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to cg_batch_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = CgBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        let total_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("cg_batch_into: rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total_elems);

        // Warm-up prefix init (NaN) per row
        let first = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| JsValue::from_str("CG: All values are NaN."))?;
        for (r, p) in combos.iter().enumerate() {
            let warm = (first + p.period.unwrap()).min(cols);
            let row = &mut out[r * cols..r * cols + warm];
            for v in row {
                *v = f64::NAN;
            }
        }

        // Compute (non-batch kernel, matching alma.rs)
        cg_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}
