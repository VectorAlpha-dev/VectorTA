//! # Wilder's Moving Average (Wilders)
//!
//! A moving average introduced by J. Welles Wilder, commonly used in indicators such as
//! the Average Directional Index (ADX). Places a heavier emphasis on new data than an SMA,
//! but less so than an EMA. Features include kernel selection, batch calculation, AVX stubs,
//! and streaming in parity with alma.rs.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Returns
//! - **`Ok(WildersOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(WildersError)`** otherwise.
//!
//! ## Developer Status
//! - **AVX2/AVX512**: Implemented for initial warmup sum (8/4‑wide accumulation);
//!   rolling recurrence remains scalar due to data dependency. Runtime selection follows alma.rs.
//! - **Streaming update**: O(1) warm‑up via incremental sum; steady‑state uses `mul_add` (FMA) for speed and single rounding.
//! - **Memory optimization**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix) ✓
//! - **SIMD note**: For single-series, speedups mainly come from large `period` where warmup dominates.
//!   Recurrence uses `mul_add` when available for slight speed and improved numerical stability.
//!
//! Decision: Streaming enforces a contiguous run of `period` finite values before first output (parity with batch),
//! then updates with y = (x − y)·(1/period) + y via fused `mul_add`. Matches batch to within FP roundoff.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::wilders_wrapper::DeviceArrayF32Wilders;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

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
use std::mem::MaybeUninit;
use thiserror::Error;

// --- Input/Output/Params/Builder Structs ---

#[derive(Debug, Clone)]
pub enum WildersData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for WildersInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            WildersData::Slice(slice) => slice,
            WildersData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WildersOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct WildersParams {
    pub period: Option<usize>,
}

impl Default for WildersParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct WildersInput<'a> {
    pub data: WildersData<'a>,
    pub params: WildersParams,
}

impl<'a> WildersInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: WildersParams) -> Self {
        Self {
            data: WildersData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: WildersParams) -> Self {
        Self {
            data: WildersData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", WildersParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WildersBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for WildersBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl WildersBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<WildersOutput, WildersError> {
        let p = WildersParams {
            period: self.period,
        };
        let i = WildersInput::from_candles(c, "close", p);
        wilders_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<WildersOutput, WildersError> {
        let p = WildersParams {
            period: self.period,
        };
        let i = WildersInput::from_slice(d, p);
        wilders_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<WildersStream, WildersError> {
        let p = WildersParams {
            period: self.period,
        };
        WildersStream::try_new(p)
    }
}

// --- Error Handling ---

#[derive(Debug, Error)]
pub enum WildersError {
    #[error("wilders: Input data slice is empty.")]
    EmptyInputData,
    #[error("wilders: All values are NaN.")]
    AllValuesNaN,
    #[error("wilders: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("wilders: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("wilders: Output length mismatch: output = {output_len}, data = {data_len}")]
    OutputLengthMismatch { output_len: usize, data_len: usize },
    #[error("wilders: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("wilders: Invalid kernel for batch operation: {0:?}")]
    InvalidKernelForBatch(Kernel),
    // Back-compat: keep the old variant name to avoid breaking callers. New code should use InvalidKernelForBatch.
    #[error("wilders: Invalid kernel type for batch operation: {kernel}")]
    #[allow(dead_code)]
    InvalidKernelType { kernel: String },
}

// --- API parity main function & kernel dispatch ---

#[inline]
pub fn wilders(input: &WildersInput) -> Result<WildersOutput, WildersError> {
    wilders_with_kernel(input, Kernel::Auto)
}

pub fn wilders_with_kernel(
    input: &WildersInput,
    kernel: Kernel,
) -> Result<WildersOutput, WildersError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(WildersError::EmptyInputData);
    }
    let period = input.get_period();

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WildersError::AllValuesNaN)?;
    if period == 0 || period > len {
        return Err(WildersError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(WildersError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // Check that all values in the initial window are finite
    for i in 0..period {
        if !data[first + i].is_finite() {
            return Err(WildersError::NotEnoughValidData {
                needed: period,
                valid: i,
            });
        }
    }

    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warm);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => wilders_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wilders_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wilders_avx512(data, period, first, &mut out),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                wilders_scalar(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(WildersOutput { values: out })
}

/// Writes Wilder's Moving Average into the provided output slice without allocating.
///
/// - Preserves NaN warmups exactly like `wilders()`/`wilders_with_kernel()`.
/// - `out.len()` must equal the input length; returns `OutputLengthMismatch` otherwise.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn wilders_into(input: &WildersInput, out: &mut [f64]) -> Result<(), WildersError> {
    wilders_into_slice(out, input, Kernel::Auto)
}

// --- Scalar calculation (core logic) ---

#[inline]
pub fn wilders_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    debug_assert!(period > 0);
    debug_assert_eq!(data.len(), out.len());

    let len = data.len();
    if len == 0 {
        return;
    }

    let wma_start = first_valid + period - 1;

    // SAFETY: callers ensure indices are valid; pre-warmup prefix is already painted (NaN).
    // We only write from wma_start..len.
    unsafe {
        // Initial sum over the first `period` finite values starting at `first_valid`.
        let mut sum = 0.0f64;
        let mut p = data.as_ptr().add(first_valid);

        // Unroll by 4 for good codegen and fewer bounds checks.
        let chunks4 = period / 4;
        for _ in 0..chunks4 {
            sum += *p.add(0) + *p.add(1) + *p.add(2) + *p.add(3);
            p = p.add(4);
        }
        match period - (chunks4 * 4) {
            3 => sum += *p.add(0) + *p.add(1) + *p.add(2),
            2 => sum += *p.add(0) + *p.add(1),
            1 => sum += *p.add(0),
            0 => {}
            _ => core::hint::unreachable_unchecked(),
        }

        // First output after warmup: simple average
        let inv_n = 1.0 / (period as f64);
        let mut y = sum * inv_n;
        *out.get_unchecked_mut(wma_start) = y;

        // Fast path: period == 1 → identity after warmup
        if period == 1 {
            let mut i = wma_start + 1;
            while i < len {
                *out.get_unchecked_mut(i) = *data.get_unchecked(i);
                i += 1;
            }
            return;
        }

        let alpha = inv_n;
        // Slight unroll-by-2 for the dependency chain to trim loop overhead.
        let mut i = wma_start + 1;
        let end_even = wma_start + 1 + ((len - (wma_start + 1)) & !1);
        while i < end_even {
            let x0 = *data.get_unchecked(i);
            y = (x0 - y).mul_add(alpha, y);
            *out.get_unchecked_mut(i) = y;

            let x1 = *data.get_unchecked(i + 1);
            y = (x1 - y).mul_add(alpha, y);
            *out.get_unchecked_mut(i + 1) = y;
            i += 2;
        }
        if i < len {
            let x = *data.get_unchecked(i);
            y = (x - y).mul_add(alpha, y);
            *out.get_unchecked_mut(i) = y;
        }
    }
}

// --- Zero-copy helper for WASM ---

/// Write directly to output slice - no allocations
pub fn wilders_into_slice(
    dst: &mut [f64],
    input: &WildersInput,
    kern: Kernel,
) -> Result<(), WildersError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(WildersError::EmptyInputData);
    }
    let len = data.len();
    let period = input.get_period();

    if dst.len() != data.len() {
        return Err(WildersError::OutputLengthMismatch {
            output_len: dst.len(),
            data_len: data.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WildersError::AllValuesNaN)?;
    if period == 0 || period > len {
        return Err(WildersError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(WildersError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // Check that all values in the initial window are finite
    for i in 0..period {
        if !data[first + i].is_finite() {
            return Err(WildersError::NotEnoughValidData {
                needed: period,
                valid: i,
            });
        }
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => wilders_scalar(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wilders_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wilders_avx512(data, period, first, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                wilders_scalar(data, period, first, dst)
            }
            _ => unreachable!(),
        }
    }

    // Fill warmup with NaN
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

// --- AVX2 and AVX512 (vectorized warmup sum; scalar recurrence) ---

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn wilders_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    debug_assert!(period > 0);
    debug_assert_eq!(data.len(), out.len());
    let len = data.len();
    if len == 0 {
        return;
    }

    let wma_start = first_valid + period - 1;

    // 4‑lane accumulation for initial window
    let mut vacc = _mm256_setzero_pd();
    let mut p = data.as_ptr().add(first_valid);
    let chunks4 = period / 4;
    for _ in 0..chunks4 {
        let v = _mm256_loadu_pd(p);
        vacc = _mm256_add_pd(vacc, v);
        p = p.add(4);
    }
    // Reduce to scalar
    let hi = _mm256_extractf128_pd(vacc, 1);
    let lo = _mm256_castpd256_pd128(vacc);
    let v2 = _mm_add_pd(lo, hi);
    let sh = _mm_permute_pd(v2, 0b01);
    let v1 = _mm_add_sd(v2, sh);
    let mut sum = _mm_cvtsd_f64(v1);

    match period - (chunks4 * 4) {
        3 => sum += *p.add(0) + *p.add(1) + *p.add(2),
        2 => sum += *p.add(0) + *p.add(1),
        1 => sum += *p.add(0),
        0 => {}
        _ => core::hint::unreachable_unchecked(),
    }

    let inv_n = 1.0 / (period as f64);
    let mut y = sum * inv_n;
    *out.get_unchecked_mut(wma_start) = y;

    if period == 1 {
        let mut i = wma_start + 1;
        while i < len {
            *out.get_unchecked_mut(i) = *data.get_unchecked(i);
            i += 1;
        }
        return;
    }

    let alpha = inv_n;
    let mut i = wma_start + 1;
    let end_even = wma_start + 1 + ((len - (wma_start + 1)) & !1);
    while i < end_even {
        let x0 = *data.get_unchecked(i);
        y = (x0 - y).mul_add(alpha, y);
        *out.get_unchecked_mut(i) = y;

        let x1 = *data.get_unchecked(i + 1);
        y = (x1 - y).mul_add(alpha, y);
        *out.get_unchecked_mut(i + 1) = y;
        i += 2;
    }
    if i < len {
        let x = *data.get_unchecked(i);
        y = (x - y).mul_add(alpha, y);
        *out.get_unchecked_mut(i) = y;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn wilders_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    if period <= 32 {
        wilders_avx512_short(data, period, first_valid, out)
    } else {
        wilders_avx512_long(data, period, first_valid, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn wilders_avx512_short(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    debug_assert!(period > 0);
    debug_assert_eq!(data.len(), out.len());
    let len = data.len();
    if len == 0 {
        return;
    }

    let wma_start = first_valid + period - 1;

    // 8‑wide accumulation for initial window
    let mut vacc = _mm512_setzero_pd();
    let mut p = data.as_ptr().add(first_valid);
    let chunks8 = period / 8;
    for _ in 0..chunks8 {
        let v = _mm512_loadu_pd(p);
        vacc = _mm512_add_pd(vacc, v);
        p = p.add(8);
    }
    // Reduce 512→256→128→scalar
    let vhi256 = _mm512_extractf64x4_pd(vacc, 1);
    let vlo256 = _mm512_castpd512_pd256(vacc);
    let v256 = _mm256_add_pd(vlo256, vhi256);
    let hi = _mm256_extractf128_pd(v256, 1);
    let lo = _mm256_castpd256_pd128(v256);
    let v2 = _mm_add_pd(lo, hi);
    let sh = _mm_permute_pd(v2, 0b01);
    let v1 = _mm_add_sd(v2, sh);
    let mut sum = _mm_cvtsd_f64(v1);

    match period - (chunks8 * 8) {
        7 => {
            sum += *p.add(0) + *p.add(1) + *p.add(2) + *p.add(3) + *p.add(4) + *p.add(5) + *p.add(6)
        }
        6 => sum += *p.add(0) + *p.add(1) + *p.add(2) + *p.add(3) + *p.add(4) + *p.add(5),
        5 => sum += *p.add(0) + *p.add(1) + *p.add(2) + *p.add(3) + *p.add(4),
        4 => sum += *p.add(0) + *p.add(1) + *p.add(2) + *p.add(3),
        3 => sum += *p.add(0) + *p.add(1) + *p.add(2),
        2 => sum += *p.add(0) + *p.add(1),
        1 => sum += *p.add(0),
        0 => {}
        _ => core::hint::unreachable_unchecked(),
    }

    let inv_n = 1.0 / (period as f64);
    let mut y = sum * inv_n;
    *out.get_unchecked_mut(wma_start) = y;

    if period == 1 {
        let mut i = wma_start + 1;
        while i < len {
            *out.get_unchecked_mut(i) = *data.get_unchecked(i);
            i += 1;
        }
        return;
    }

    let alpha = inv_n;
    let mut i = wma_start + 1;
    let end_even = wma_start + 1 + ((len - (wma_start + 1)) & !1);
    while i < end_even {
        let x0 = *data.get_unchecked(i);
        y = (x0 - y).mul_add(alpha, y);
        *out.get_unchecked_mut(i) = y;

        let x1 = *data.get_unchecked(i + 1);
        y = (x1 - y).mul_add(alpha, y);
        *out.get_unchecked_mut(i + 1) = y;
        i += 2;
    }
    if i < len {
        let x = *data.get_unchecked(i);
        y = (x - y).mul_add(alpha, y);
        *out.get_unchecked_mut(i) = y;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn wilders_avx512_long(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    debug_assert!(period > 0);
    debug_assert_eq!(data.len(), out.len());
    let len = data.len();
    if len == 0 {
        return;
    }

    let wma_start = first_valid + period - 1;

    // For longer periods, unroll by 16 doubles per iter using 2 accumulators
    let mut vacc0 = _mm512_setzero_pd();
    let mut vacc1 = _mm512_setzero_pd();
    let mut p = data.as_ptr().add(first_valid);
    let chunks16 = period / 16;
    for _ in 0..chunks16 {
        let v0 = _mm512_loadu_pd(p);
        let v1 = _mm512_loadu_pd(p.add(8));
        vacc0 = _mm512_add_pd(vacc0, v0);
        vacc1 = _mm512_add_pd(vacc1, v1);
        p = p.add(16);
    }
    let mut vacc = _mm512_add_pd(vacc0, vacc1);

    // Handle leftover 8‑wide chunk
    let rem = period - (chunks16 * 16);
    if rem >= 8 {
        let v = _mm512_loadu_pd(p);
        vacc = _mm512_add_pd(vacc, v);
        p = p.add(8);
    }

    // Reduce to scalar
    let vhi256 = _mm512_extractf64x4_pd(vacc, 1);
    let vlo256 = _mm512_castpd512_pd256(vacc);
    let v256 = _mm256_add_pd(vlo256, vhi256);
    let hi = _mm256_extractf128_pd(v256, 1);
    let lo = _mm256_castpd256_pd128(v256);
    let v2 = _mm_add_pd(lo, hi);
    let sh = _mm_permute_pd(v2, 0b01);
    let v1 = _mm_add_sd(v2, sh);
    let mut sum = _mm_cvtsd_f64(v1);

    match period - (chunks16 * 16) - (rem / 8) * 8 {
        7 => {
            sum += *p.add(0) + *p.add(1) + *p.add(2) + *p.add(3) + *p.add(4) + *p.add(5) + *p.add(6)
        }
        6 => sum += *p.add(0) + *p.add(1) + *p.add(2) + *p.add(3) + *p.add(4) + *p.add(5),
        5 => sum += *p.add(0) + *p.add(1) + *p.add(2) + *p.add(3) + *p.add(4),
        4 => sum += *p.add(0) + *p.add(1) + *p.add(2) + *p.add(3),
        3 => sum += *p.add(0) + *p.add(1) + *p.add(2),
        2 => sum += *p.add(0) + *p.add(1),
        1 => sum += *p.add(0),
        0 => {}
        _ => core::hint::unreachable_unchecked(),
    }

    let inv_n = 1.0 / (period as f64);
    let mut y = sum * inv_n;
    *out.get_unchecked_mut(wma_start) = y;

    if period == 1 {
        let mut i = wma_start + 1;
        while i < len {
            *out.get_unchecked_mut(i) = *data.get_unchecked(i);
            i += 1;
        }
        return;
    }

    let alpha = inv_n;
    let mut i = wma_start + 1;
    let end_even = wma_start + 1 + ((len - (wma_start + 1)) & !1);
    while i < end_even {
        let x0 = *data.get_unchecked(i);
        y = (x0 - y).mul_add(alpha, y);
        *out.get_unchecked_mut(i) = y;

        let x1 = *data.get_unchecked(i + 1);
        y = (x1 - y).mul_add(alpha, y);
        *out.get_unchecked_mut(i + 1) = y;
        i += 2;
    }
    if i < len {
        let x = *data.get_unchecked(i);
        y = (x - y).mul_add(alpha, y);
        *out.get_unchecked_mut(i) = y;
    }
}

// --- Streaming (WildersStream) ---

/// Streaming Wilder's MA with O(1) warm-up and FMA recurrence.
/// Requires a contiguous run of `period` finite values before the first output.
#[derive(Debug, Clone)]
pub struct WildersStream {
    period: usize,
    alpha: f64,

    // O(1) warm-up state
    warm_sum: f64,
    warm_count: usize,

    // Steady-state
    last: f64,
    started: bool,
}

impl WildersStream {
    #[inline(always)]
    pub fn try_new(params: WildersParams) -> Result<Self, WildersError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(WildersError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let alpha = 1.0 / (period as f64);
        Ok(Self {
            period,
            alpha,
            warm_sum: 0.0,
            warm_count: 0,
            last: f64::NAN,
            started: false,
        })
    }

    /// Push one value. Returns `Some(y)` once initialized, else `None` during warm-up.
    /// Warm-up is O(1) per tick: we incrementally build the initial average instead of
    /// scanning the whole buffer when it fills.
    ///
    /// NaN/inf policy:
    /// - Before start: require a contiguous run of `period` finite values; any non-finite
    ///   resets the streak and keeps returning `None`.
    /// - After start: apply the Wilder recurrence; non-finite inputs propagate per IEEE‑754.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Warm-up path
        if !self.started {
            if !value.is_finite() {
                self.warm_sum = 0.0;
                self.warm_count = 0;
                return None;
            }

            self.warm_sum += value;
            self.warm_count += 1;

            if self.warm_count < self.period {
                return None;
            }

            // First output equals the simple average of the first `period` finite values
            let y0 = self.warm_sum / (self.period as f64);
            self.last = y0;
            self.started = true;
            return Some(self.last);
        }

        // Steady-state update with FMA
        self.last = (value - self.last).mul_add(self.alpha, self.last);
        Some(self.last)
    }
}

// --- Batch Ranges, Builder, Output, Batch Apply ---

#[derive(Clone, Debug)]
pub struct WildersBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for WildersBatchRange {
    fn default() -> Self {
        Self { period: (5, 24, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WildersBatchBuilder {
    range: WildersBatchRange,
    kernel: Kernel,
}

impl WildersBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<WildersBatchOutput, WildersError> {
        wilders_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<WildersBatchOutput, WildersError> {
        WildersBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<WildersBatchOutput, WildersError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WildersBatchOutput, WildersError> {
        WildersBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn wilders_batch_with_kernel(
    data: &[f64],
    sweep: &WildersBatchRange,
    k: Kernel,
) -> Result<WildersBatchOutput, WildersError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(WildersError::InvalidKernelForBatch(k))
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    wilders_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct WildersBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WildersParams>,
    pub rows: usize,
    pub cols: usize,
}
impl WildersBatchOutput {
    pub fn row_for_params(&self, p: &WildersParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &WildersParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

// --- Batch Internals & Grid Expansion ---

#[inline(always)]
fn expand_grid(r: &WildersBatchRange) -> Vec<WildersParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        // Zero step or equal bounds → static value
        if step == 0 || start == end {
            return vec![start];
        }
        let mut out = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                out.push(v);
                match v.checked_add(step) {
                    Some(n) => v = n,
                    None => break, // overflow guard
                }
            }
        } else {
            // Reversed bounds: walk downward by `step`
            let mut v = start;
            loop {
                if v < end {
                    break;
                }
                out.push(v);
                if v < end + step {
                    break;
                }
                v = v.saturating_sub(step);
                if v == 0 && end != 0 {
                    // Prevent infinite loop on extreme underflow scenarios
                    break;
                }
            }
        }
        out
    }

    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(WildersParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn wilders_batch_slice(
    data: &[f64],
    sweep: &WildersBatchRange,
    kern: Kernel,
) -> Result<WildersBatchOutput, WildersError> {
    wilders_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn wilders_batch_par_slice(
    data: &[f64],
    sweep: &WildersBatchRange,
    kern: Kernel,
) -> Result<WildersBatchOutput, WildersError> {
    wilders_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn wilders_batch_inner(
    data: &[f64],
    sweep: &WildersBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<WildersBatchOutput, WildersError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, st) = sweep.period;
        return Err(WildersError::InvalidRange { start: s, end: e, step: st });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WildersError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(WildersError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    // Checked arithmetic before allocation
    let _total = rows
        .checked_mul(cols)
        .ok_or(WildersError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // -----------------------------------------
    // 2. allocate rows×cols uninitialised
    //    and paint the NaN prefixes
    // -----------------------------------------
    let mut raw = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut raw, cols, &warm);

    // Precompute prefix sums once for all rows: P[i] = sum(data[first..first+i])
    let mut pref = Vec::with_capacity((cols - first) + 1);
    pref.push(0.0);
    let mut acc = 0.0f64;
    for &x in &data[first..] {
        acc += x;
        pref.push(acc);
    }

    // -----------------------------------------
    // 3. helper that fills a single row
    // -----------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let out_row = std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        // Initial average via prefix sums
        let wma_start = first + period - 1;
        let sum = pref[period] - pref[0];
        let inv_n = 1.0 / (period as f64);
        let mut y = sum * inv_n;
        *out_row.get_unchecked_mut(wma_start) = y;

        if period == 1 {
            let mut i = wma_start + 1;
            while i < cols {
                *out_row.get_unchecked_mut(i) = *data.get_unchecked(i);
                i += 1;
            }
            return;
        }

        let alpha = inv_n;
        let mut i = wma_start + 1;
        let end_even = wma_start + 1 + ((cols - (wma_start + 1)) & !1);
        while i < end_even {
            let x0 = *data.get_unchecked(i);
            y = (x0 - y).mul_add(alpha, y);
            *out_row.get_unchecked_mut(i) = y;

            let x1 = *data.get_unchecked(i + 1);
            y = (x1 - y).mul_add(alpha, y);
            *out_row.get_unchecked_mut(i + 1) = y;
            i += 2;
        }
        if i < cols {
            let x = *data.get_unchecked(i);
            y = (x - y).mul_add(alpha, y);
            *out_row.get_unchecked_mut(i) = y;
        }
    };

    // -----------------------------------------
    // 4. run every row
    // -----------------------------------------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in raw.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // -----------------------------------------
    // 5. convert to Vec<f64> now that everything
    //    has been fully initialised
    // -----------------------------------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
    Ok(WildersBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn wilders_batch_inner_into(
    data: &[f64],
    sweep: &WildersBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<WildersParams>, WildersError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        let (s, e, st) = sweep.period;
        return Err(WildersError::InvalidRange { start: s, end: e, step: st });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(WildersError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(WildersError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    // Guard potential overflow of rows * cols used by callers for preallocation
    let _total = rows
        .checked_mul(cols)
        .ok_or(WildersError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;

    // 1) Cast to MaybeUninit and paint warm prefixes via helper
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    // Precompute prefix sums once for all rows: P[i] = sum(data[first..first+i])
    let mut pref = Vec::with_capacity((cols - first) + 1);
    pref.push(0.0);
    let mut acc = 0.0f64;
    for &x in &data[first..] {
        acc += x;
        pref.push(acc);
    }

    // 2) Row writer that fills post-warm cells
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let dst: &mut [f64] =
            std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        // Initial average via prefix sums
        let wma_start = first + period - 1;
        let sum = pref[period] - pref[0];
        let inv_n = 1.0 / (period as f64);
        let mut y = sum * inv_n;
        *dst.get_unchecked_mut(wma_start) = y;

        if period == 1 {
            let mut i = wma_start + 1;
            while i < cols {
                *dst.get_unchecked_mut(i) = *data.get_unchecked(i);
                i += 1;
            }
            return;
        }

        let alpha = inv_n;
        let mut i = wma_start + 1;
        let end_even = wma_start + 1 + ((cols - (wma_start + 1)) & !1);
        while i < end_even {
            let x0 = *data.get_unchecked(i);
            y = (x0 - y).mul_add(alpha, y);
            *dst.get_unchecked_mut(i) = y;

            let x1 = *data.get_unchecked(i + 1);
            y = (x1 - y).mul_add(alpha, y);
            *dst.get_unchecked_mut(i + 1) = y;
            i += 2;
        }
        if i < cols {
            let x = *data.get_unchecked(i);
            y = (x - y).mul_add(alpha, y);
            *dst.get_unchecked_mut(i) = y;
        }
    };

    // 3) Iterate by rows
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, row)| do_row(r, row));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, row) in out_mu.chunks_mut(cols).enumerate() {
                do_row(r, row);
            }
        }
    } else {
        for (r, row) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    }

    Ok(combos)
}

// --- Row functions for batch (all just call scalar or AVX stubs) ---

#[inline(always)]
pub unsafe fn wilders_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    wilders_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn wilders_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    wilders_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn wilders_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        wilders_row_avx512_short(data, first, period, out)
    } else {
        wilders_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn wilders_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    wilders_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn wilders_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    wilders_avx512_long(data, period, first, out)
}

// --- Unit Tests (feature parity with alma.rs) ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_wilders_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Prepare input: two NaNs then a ramp of finite values
        let mut data = Vec::with_capacity(256);
        data.push(f64::NAN);
        data.push(f64::NAN);
        for i in 1..=254 {
            data.push(i as f64);
        }

        let params = WildersParams { period: Some(5) };
        let input = WildersInput::from_slice(&data, params);

        // Baseline via existing Vec-returning API
        let expected = wilders(&input)?.values;

        // Preallocate output and call new into API
        let mut out = vec![0.0; data.len()];

        #[cfg(not(feature = "wasm"))]
        {
            wilders_into(&input, &mut out)?;
        }

        #[cfg(feature = "wasm")]
        {
            // Fallback in wasm builds: parity still validated via into_slice helper
            wilders_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(out.len(), expected.len());
        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "NaN parity mismatch at index {}", i);
            } else {
                let diff = (a - b).abs();
                assert!(diff <= 1e-12, "Mismatch at {}: got {}, expected {}, diff {}", i, a, b, diff);
            }
        }
        Ok(())
    }

    fn check_wilders_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = WildersParams { period: None };
        let input = WildersInput::from_candles(&candles, "close", default_params);
        let output = wilders_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_wilders_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input =
            WildersInput::from_candles(&candles, "close", WildersParams { period: Some(5) });
        let result = wilders_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59302.18156619092,
            59277.94525295273,
            59230.15620236219,
            59215.12496188975,
            59103.0999695118,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] Wilders {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_wilders_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = WildersInput::with_default_candles(&candles);
        match input.data {
            WildersData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected WildersData::Candles"),
        }
        let output = wilders_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_wilders_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = WildersParams { period: Some(0) };
        let input = WildersInput::from_slice(&input_data, params);
        let res = wilders_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wilders should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_wilders_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = WildersParams { period: Some(10) };
        let input = WildersInput::from_slice(&data_small, params);
        let res = wilders_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wilders should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_wilders_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = WildersParams { period: Some(1) };
        let input = WildersInput::from_slice(&single_point, params);
        let res = wilders_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), single_point.len());
        Ok(())
    }

    fn check_wilders_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = WildersParams { period: Some(5) };
        let first_input = WildersInput::from_candles(&candles, "close", first_params);
        let first_result = wilders_with_kernel(&first_input, kernel)?;

        let second_params = WildersParams { period: Some(10) };
        let second_input = WildersInput::from_slice(&first_result.values, second_params);
        let second_result = wilders_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_wilders_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input =
            WildersInput::from_candles(&candles, "close", WildersParams { period: Some(5) });
        let res = wilders_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
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

    fn check_wilders_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = WildersInput::from_slice(&empty, WildersParams::default());
        let res = wilders_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(WildersError::EmptyInputData)),
            "[{}] Wilders should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_wilders_nan_in_initial_window(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        // Data with NaN in the initial window after first valid value
        let data = vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let params = WildersParams { period: Some(5) };
        let input = WildersInput::from_slice(&data, params);
        let res = wilders_with_kernel(&input, kernel);
        assert!(
            matches!(
                res,
                Err(WildersError::NotEnoughValidData {
                    needed: 5,
                    valid: 2
                })
            ),
            "[{}] Wilders should fail with NaN in initial window, got: {:?}",
            test_name,
            res
        );
        Ok(())
    }

    fn check_wilders_output_mismatch(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = WildersParams { period: Some(3) };
        let input = WildersInput::from_slice(&data, params);

        // Create a mismatched output buffer
        let mut out = vec![0.0; 10]; // Wrong size
        let res = wilders_into_slice(&mut out, &input, kernel);
        assert!(
            matches!(
                res,
                Err(WildersError::OutputLengthMismatch {
                    output_len: 10,
                    data_len: 5
                })
            ),
            "[{}] Wilders should fail with output length mismatch, got: {:?}",
            test_name,
            res
        );
        Ok(())
    }

    fn check_wilders_invalid_kernel_batch() -> Result<(), Box<dyn Error>> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let sweep = WildersBatchRange { period: (3, 5, 1) };

        // Try to use a non-batch kernel for batch operation
        let res = wilders_batch_with_kernel(&data, &sweep, Kernel::Scalar);
        assert!(
            matches!(res, Err(WildersError::InvalidKernelForBatch(_))),
            "Wilders batch should fail with non-batch kernel, got: {:?}",
            res
        );
        Ok(())
    }

    fn check_wilders_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 5;
        let input = WildersInput::from_candles(
            &candles,
            "close",
            WildersParams {
                period: Some(period),
            },
        );
        let batch_output = wilders_with_kernel(&input, kernel)?.values;

        let mut stream = WildersStream::try_new(WildersParams {
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
            assert!(
                diff < 1e-9,
                "[{}] Wilders streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_wilders_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase chance of catching bugs
        let test_periods = vec![
            2,   // Small period
            5,   // Default period
            10,  // Medium period
            14,  // Common RSI period
            20,  // Common period
            50,  // Large period
            100, // Very large period
            200, // Extra large period
        ];

        for &period in &test_periods {
            // Skip if period would be too large for the data
            if period > candles.close.len() {
                continue;
            }

            let input = WildersInput::from_candles(
                &candles,
                "close",
                WildersParams {
                    period: Some(period),
                },
            );
            let output = wilders_with_kernel(&input, kernel)?;

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
    fn check_wilders_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_wilders_tests {
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
    fn check_wilders_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=64).prop_flat_map(|period| {
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
                let params = WildersParams {
                    period: Some(period),
                };
                let input = WildersInput::from_slice(&data, params.clone());

                let WildersOutput { values: out } = wilders_with_kernel(&input, kernel)?;
                let WildersOutput { values: ref_out } =
                    wilders_with_kernel(&input, Kernel::Scalar)?;

                // Property 1: Output length matches input
                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                // Property 2: Warmup period handling - first period-1 values should be NaN
                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup = first_valid + period - 1;
                for i in 0..warmup.min(out.len()) {
                    prop_assert!(out[i].is_nan(), "Expected NaN at index {} during warmup", i);
                }

                // Property 3: Finite values after warmup
                for i in warmup..out.len() {
                    if data[i].is_finite() {
                        prop_assert!(
                            out[i].is_finite(),
                            "Expected finite value at index {} after warmup",
                            i
                        );
                    }
                }

                // Property 4: Bounded by min/max of input data
                // Wilder's MA is a weighted average, so output should be within data bounds
                if warmup < out.len() {
                    let data_min = data[first_valid..]
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, f64::min);
                    let data_max = data[first_valid..]
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);

                    for i in warmup..out.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                out[i] >= data_min - 1e-9 && out[i] <= data_max + 1e-9,
                                "Output {} at index {} outside bounds [{}, {}]",
                                out[i],
                                i,
                                data_min,
                                data_max
                            );
                        }
                    }
                }

                // Property 5: Period=1 should equal input values
                if period == 1 && warmup < out.len() {
                    for i in warmup..out.len() {
                        prop_assert!(
                            (out[i] - data[i]).abs() <= 1e-9,
                            "Period=1 output {} should equal input {} at index {}",
                            out[i],
                            data[i],
                            i
                        );
                    }
                }

                // Property 6: Constant input produces constant output
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && warmup < out.len() {
                    let constant_val = data[first_valid];
                    for i in warmup..out.len() {
                        prop_assert!(
                            (out[i] - constant_val).abs() <= 1e-9,
                            "Constant input should produce constant output {} at index {}",
                            out[i],
                            i
                        );
                    }
                }

                // Property 7: Exponential decay property
                // Wilder's uses formula: new_val = (data[i] - prev_val) * alpha + prev_val
                // where alpha = 1.0 / period
                if warmup + 1 < out.len() {
                    let alpha = 1.0 / (period as f64);
                    for i in (warmup + 1)..out.len() {
                        let expected = (data[i] - out[i - 1]) * alpha + out[i - 1];
                        prop_assert!(
                            (out[i] - expected).abs() <= 1e-9,
                            "Exponential decay formula mismatch at index {}: got {}, expected {}",
                            i,
                            out[i],
                            expected
                        );
                    }
                }

                // Property 8: First value after warmup equals simple average
                // Wilder's MA initializes with the simple average of the first period values
                if warmup < out.len() && warmup >= period - 1 {
                    let sum: f64 = data[first_valid..first_valid + period].iter().sum();
                    let expected_first = sum / (period as f64);
                    prop_assert!(
                        (out[warmup] - expected_first).abs() <= 1e-9,
                        "First output {} should equal simple average {} of first {} values",
                        out[warmup],
                        expected_first,
                        period
                    );
                }

                // Property 9: Kernel consistency - all kernels should produce same results
                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "NaN/infinite mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "Kernel mismatch at index {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                Ok(())
            })
            .map_err(|e| e.into())
    }

    generate_all_wilders_tests!(
        check_wilders_partial_params,
        check_wilders_accuracy,
        check_wilders_default_candles,
        check_wilders_zero_period,
        check_wilders_period_exceeds_length,
        check_wilders_very_small_dataset,
        check_wilders_reinput,
        check_wilders_nan_handling,
        check_wilders_empty_input,
        check_wilders_nan_in_initial_window,
        check_wilders_output_mismatch,
        check_wilders_streaming,
        check_wilders_no_poison,
        check_wilders_property
    );

    // Test invalid kernel batch separately (doesn't need kernel variants)
    #[test]
    fn test_wilders_invalid_kernel_batch() {
        let _ = check_wilders_invalid_kernel_batch();
    }

    // Test that NaN in initial window is caught and doesn't poison the series
    #[test]
    fn test_wilders_nan_poisoning_prevented() {
        // Create data with NaN in position 2 (within initial window for period=5)
        let data = vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let params = WildersParams { period: Some(5) };
        let input = WildersInput::from_slice(&data, params);

        // This should fail with NotEnoughValidData, not produce a poisoned series
        let result = wilders(&input);
        assert!(result.is_err(), "Should fail with NaN in initial window");

        // Now test with clean data to ensure normal operation works
        let clean_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let clean_params = WildersParams { period: Some(5) };
        let clean_input = WildersInput::from_slice(&clean_data, clean_params);
        let clean_result = wilders(&clean_input).unwrap();

        // Check that values after warmup are finite (not NaN-poisoned)
        for i in 5..clean_result.values.len() {
            assert!(
                clean_result.values[i].is_finite(),
                "Value at index {} should be finite, got {}",
                i,
                clean_result.values[i]
            );
        }
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = WildersBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = WildersParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Last five expected Wilder’s values for period = 5
        let expected = [
            59302.18156619092,
            59277.94525295273,
            59230.15620236219,
            59215.12496188975,
            59103.0999695118,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
                "[{}] default-row mismatch at idx {}: {} vs {:?}",
                test,
                i,
                v,
                expected
            );
        }

        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations to increase detection coverage
        let batch_configs = vec![
            (2, 10, 1),    // Small range with step 1
            (5, 25, 5),    // Default start with step 5
            (10, 30, 10),  // Medium range with larger step
            (14, 50, 7),   // RSI period range with step 7
            (20, 100, 20), // Large range with large step
            (50, 200, 50), // Very large periods
            (2, 6, 2),     // Very small range to test edge cases
        ];

        for (start, end, step) in batch_configs {
            // Skip configurations that would exceed data length
            if start > c.close.len() {
                continue;
            }

            let output = WildersBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
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
                let period = output.combos[row].period.unwrap_or(0);

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }
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

#[cfg(feature = "python")]
#[pyfunction(name = "wilders")]
#[pyo3(signature = (data, period, kernel=None))]

pub fn wilders_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = WildersParams {
        period: Some(period),
    };
    let wilders_in = WildersInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| wilders_with_kernel(&wilders_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "WildersStream")]
pub struct WildersStreamPy {
    stream: WildersStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl WildersStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = WildersParams {
            period: Some(period),
        };
        let stream =
            WildersStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(WildersStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated Wilder's MA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "wilders_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]

pub fn wilders_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = WildersBatchRange {
        period: period_range,
    };

    // Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate NumPy array (1-D, will reshape later)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Parse kernel string to enum with CPU feature validation
    let kern = validate_kernel(kernel, true)?;

    // Heavy work without the GIL
    let combos = py
        .allow_threads(|| {
            // Resolve Kernel::Auto to a specific kernel
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
            // Call the batch function with the pre-allocated buffer
            wilders_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build dict with the GIL
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
#[pyfunction(name = "wilders_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn wilders_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<WildersDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaWilders;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = WildersBatchRange {
        period: period_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaWilders::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.wilders_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(WildersDeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "wilders_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn wilders_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<WildersDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaWilders;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = WildersParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaWilders::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.wilders_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(WildersDeviceArrayF32Py { inner })
}

// Wilders-specific CUDA Array Interface v3 + DLPack stubs (context-guarded handle)
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct WildersDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32Wilders,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl WildersDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        // 2D row-major FP32: (rows, cols)
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Wrapper synchronizes its stream before returning, so omit 'stream' per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        // 2 == kDLCUDA
        (2, self.inner.device_id as i32)
    }

    fn __dlpack__<'py>(&self, py: Python<'py>, _stream: Option<i64>) -> PyResult<PyObject> {
        use std::ffi::c_void;

        #[repr(C)]
        struct DLDevice { device_type: i32, device_id: i32 }
        #[repr(C)]
        struct DLDataType { code: u8, bits: u8, lanes: u16 }
        #[repr(C)]
        struct DLTensor {
            data: *mut c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: u64,
        }
        #[repr(C)]
        struct DLManagedTensor {
            dl_tensor: DLTensor,
            manager_ctx: *mut c_void,
            deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
        }
        struct DlpGuard {
            // Own the shape/strides allocations and keep CUDA context alive
            _shape: Box<[i64; 2]>,
            _strides: Box<[i64; 2]>,
            _ctx: std::sync::Arc<cust::context::Context>,
        }

        extern "C" fn managed_deleter(p: *mut DLManagedTensor) {
            unsafe {
                if p.is_null() { return; }
                // Reclaim guard first (drops Arc + Boxed arrays)
                let guard_ptr = (*p).manager_ctx as *mut DlpGuard;
                if !guard_ptr.is_null() {
                    drop(Box::from_raw(guard_ptr));
                }
                // Finally free the DLManagedTensor itself
                drop(Box::from_raw(p));
            }
        }

        extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
            unsafe {
                let name = b"dltensor\0";
                let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr() as *const _) as *mut DLManagedTensor;
                if !ptr.is_null() {
                    if let Some(del) = (*ptr).deleter { del(ptr); }
                    // Rename capsule per convention to prevent double use
                    let used = b"used_dltensor\0";
                    pyo3::ffi::PyCapsule_SetName(capsule, used.as_ptr() as *const _);
                }
            }
        }

        // Build shape/strides (element-based strides per DLPack)
        let shape = Box::new([self.inner.rows as i64, self.inner.cols as i64]);
        let strides = Box::new([self.inner.cols as i64, 1i64]);
        let data_ptr = self.inner.device_ptr() as usize as *mut c_void;

        let guard = Box::new(DlpGuard { _shape: shape, _strides: strides, _ctx: self.inner.ctx.clone() });
        let guard_ptr = Box::into_raw(guard);

        // SAFETY: guard owns shape/strides; we borrow raw pointers into the tensor
        let guard_ref = unsafe { &*guard_ptr };
        let mt = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice { device_type: 2, device_id: self.inner.device_id as i32 },
                ndim: 2,
                dtype: DLDataType { code: 2 /* kDLFloat */, bits: 32, lanes: 1 },
                shape: guard_ref._shape.as_ptr() as *mut i64,
                strides: guard_ref._strides.as_ptr() as *mut i64,
                byte_offset: 0,
            },
            manager_ctx: guard_ptr as *mut c_void,
            deleter: Some(managed_deleter),
        });
        let mt_raw = Box::into_raw(mt);
        let name = b"dltensor\0";
        let capsule = unsafe {
            pyo3::ffi::PyCapsule_New(mt_raw as *mut c_void, name.as_ptr() as *const _, Some(capsule_destructor))
        };
        if capsule.is_null() {
            // If capsule creation failed, free the managed tensor explicitly
            unsafe { managed_deleter(mt_raw); }
            return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create DLPack capsule"));
        }
        Ok(unsafe { PyObject::from_owned_ptr(py, capsule) })
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wilders_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = WildersParams {
        period: Some(period),
    };
    let input = WildersInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    wilders_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wilders_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wilders_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wilders_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to wilders_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = WildersParams {
            period: Some(period),
        };
        let input = WildersInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // Handle aliasing - compute into temp buffer first
            let mut temp = vec![0.0; len];
            wilders_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            wilders_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WildersBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WildersBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WildersParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = wilders_batch)]
pub fn wilders_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: WildersBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = WildersBatchRange {
        period: config.period_range,
    };

    // Resolve Kernel::Auto to actual kernel before calling batch_inner
    let kernel = match Kernel::Auto {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    let output = wilders_batch_inner(data, &sweep, simd, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = WildersBatchJsOutput {
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
pub fn wilders_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = WildersBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    wilders_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wilders_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to wilders_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = WildersBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Resolve Kernel::Auto to actual kernel before calling batch_inner_into
        let kernel = match Kernel::Auto {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };

        wilders_batch_inner_into(data, &sweep, simd, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wilders_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = WildersBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}
