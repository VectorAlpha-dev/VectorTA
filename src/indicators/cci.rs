//! # Commodity Channel Index (CCI)
//! Decision log: SIMD present but short‑circuited to scalar for stability; CUDA wrappers available; Python interop via CAI v3 and DLPack; numerical outputs unchanged.
//!
//! CCI measures the variation of a security's price from its statistical mean.
//! It is calculated as the difference between typical price and its moving average,
//! divided by the mean absolute deviation multiplied by a constant (0.015).
//!
//! ## Parameters
//! - **period**: Window size for calculations (default: 14)
//!
//! ## Returns
//! - **`Ok(CciOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(CciError)`** on various error conditions.
//!
//! ## Developer Status
//! - **SIMD Kernels**: Implemented but disabled by default (runtime short-circuits to scalar). Compensated per-lane sums are present for future use; current versions are slower at 10k/100k/1M.
//! - **Streaming Performance**: O(log n) exact — maintains an order‑statistics treap to compute MAD without full rescans
//! - **Memory Optimization**: GOOD - properly uses alloc_with_nan_prefix helper for zero-copy allocation
//! - Note: CCI’s MAD requires a full window scan; SIMD accelerates that scan but cannot make it O(1).

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

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::CudaCci;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

// ---- Input/Output Structs ----

#[derive(Debug, Clone)]
pub enum CciData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CciOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CciParams {
    pub period: Option<usize>,
}

impl Default for CciParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct CciInput<'a> {
    pub data: CciData<'a>,
    pub params: CciParams,
}

impl<'a> AsRef<[f64]> for CciInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CciData::Slice(slice) => slice,
            CciData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> CciInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CciParams) -> Self {
        Self {
            data: CciData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CciParams) -> Self {
        Self {
            data: CciData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "hlc3", CciParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn data_len(&self) -> usize {
        match &self.data {
            CciData::Slice(slice) => slice.len(),
            CciData::Candles { candles, .. } => candles.close.len(),
        }
    }
}

// ---- Builder ----

#[derive(Copy, Clone, Debug)]
pub struct CciBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CciBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CciBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<CciOutput, CciError> {
        let p = CciParams {
            period: self.period,
        };
        let i = CciInput::from_candles(c, "hlc3", p);
        cci_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CciOutput, CciError> {
        let p = CciParams {
            period: self.period,
        };
        let i = CciInput::from_slice(d, p);
        cci_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<CciStream, CciError> {
        let p = CciParams {
            period: self.period,
        };
        CciStream::try_new(p)
    }
}

// ---- Error ----

#[derive(Debug, Error)]
pub enum CciError {
    #[error("cci: Input data slice is empty.")]
    EmptyInputData,
    #[error("cci: All values are NaN.")]
    AllValuesNaN,
    #[error("cci: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cci: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("cci: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("cci: invalid range expansion: start={start} end={end} step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("cci: invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
}

// ---- Indicator API ----

#[inline]
pub fn cci(input: &CciInput) -> Result<CciOutput, CciError> {
    cci_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn cci_prepare<'a>(
    input: &'a CciInput,
    kernel: Kernel,
) -> Result<
    (
        // data
        &'a [f64],
        // period
        usize,
        // first
        usize,
        // chosen
        Kernel,
    ),
    CciError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CciError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CciError::AllValuesNaN)?;
    let period = input.get_period();

    // ---- same guards as before ----
    if period == 0 || period > len {
        return Err(CciError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(CciError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // kernel auto-detection only once
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, period, first, chosen))
}

pub fn cci_with_kernel(input: &CciInput, kernel: Kernel) -> Result<CciOutput, CciError> {
    let (data, period, first, chosen) = cci_prepare(input, kernel)?;

    // Use alloc_with_nan_prefix helper like ALMA does
    let prefix = first + period - 1;
    let mut out = alloc_with_nan_prefix(data.len(), prefix);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => cci_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cci_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => cci_avx512(data, period, first, &mut out),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                cci_scalar(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(CciOutput { values: out })
}

#[inline]
pub fn cci_into_slice(dst: &mut [f64], input: &CciInput, kern: Kernel) -> Result<(), CciError> {
    let (data, period, first, chosen) = cci_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(CciError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    // Compute directly into the output slice
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => cci_scalar(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => cci_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => cci_avx512(data, period, first, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                cci_scalar(data, period, first, dst)
            }
            _ => unreachable!(),
        }
    }

    // Set warmup period to NaN
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Writes CCI values into a caller-provided output slice without allocating.
///
/// - Preserves the module's NaN warmup semantics (prefix up to `first + period - 1`).
/// - `out.len()` must equal `input` length; returns the module's length error on mismatch.
/// - Uses kernel auto-detection (same path as `cci()`), delegating to the existing compute kernels.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn cci_into(input: &CciInput, out: &mut [f64]) -> Result<(), CciError> {
    cci_into_slice(out, input, Kernel::Auto)
}

// ---- Scalar + AVX2/AVX512 ----

/// Optimized safe scalar implementation.
///
/// - Single-pass rolling sum for SMA
/// - Recomputes MAD per step (mathematically required), but uses chunked iterations
/// - Minimizes divisions by precomputing constants
#[inline]
pub fn cci_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    debug_assert_eq!(data.len(), out.len());
    let n = data.len();
    if n == 0 {
        return;
    }

    let inv_p = 1.0 / (period as f64);

    // ---- initial window sum ----
    let start0 = first_valid;
    let end0 = start0 + period;
    let mut sum: f64 = data[start0..end0].iter().sum();
    let mut sma = sum * inv_p;

    // ---- initial MAD ----
    let mut sum_abs = 0.0;
    for &v in &data[start0..end0] {
        sum_abs += (v - sma).abs();
    }

    let first_out = first_valid + period - 1;
    let price0 = data[first_out];
    out[first_out] = {
        let denom = 0.015 * (sum_abs * inv_p);
        if denom == 0.0 {
            0.0
        } else {
            (price0 - sma) / denom
        }
    };

    // ---- rolling steps ----
    for i in (first_out + 1)..n {
        let exiting = data[i - period];
        let entering = data[i];
        sum = sum - exiting + entering;
        sma = sum * inv_p;

        let wstart = i + 1 - period;
        let wend = i + 1; // exclusive
        let mut sabs = 0.0;
        for &v in &data[wstart..wend] {
            sabs += (v - sma).abs();
        }

        out[i] = {
            let denom = 0.015 * (sabs * inv_p);
            if denom == 0.0 {
                0.0
            } else {
                (entering - sma) / denom
            }
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cci_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    // Experimental SIMD slower on this workload; short-circuit to scalar
    cci_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cci_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    // Experimental SIMD slower on this workload; short-circuit to scalar
    cci_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn cci_avx2_impl(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    debug_assert!(data.len() == out.len());
    debug_assert!(period >= 1 && first_valid + period <= data.len());

    let n = data.len();
    let inv_p = 1.0 / (period as f64);
    let scale = (period as f64) * (1.0 / 0.015);

    // Initial sum (scalar)
    let base = data.as_ptr().add(first_valid);
    let mut sum = 0.0;
    {
        let mut k = 0usize;
        while k + 4 <= period {
            let x0 = *base.add(k + 0);
            let x1 = *base.add(k + 1);
            let x2 = *base.add(k + 2);
            let x3 = *base.add(k + 3);
            sum = sum + x0 + x1 + x2 + x3;
            k += 4;
        }
        while k < period {
            sum += *base.add(k);
            k += 1;
        }
    }

    let first_out = first_valid + period - 1;
    let mut sma = sum * inv_p;

    // Initial MAD (vectorized compute, scalar Kahan accumulation in element order)
    {
        let vmean = _mm256_set1_pd(sma);
        let vsgn = _mm256_set1_pd(-0.0f64);
        let mut k = 0usize;
        let mut sum_abs = 0.0f64;
        let mut comp = 0.0f64;
        while k + 4 <= period {
            let x = _mm256_loadu_pd(base.add(k));
            let d = _mm256_sub_pd(x, vmean);
            let a = _mm256_andnot_pd(vsgn, d);
            let mut lane = [0.0f64; 4];
            _mm256_storeu_pd(lane.as_mut_ptr(), a);
            // accumulate in element order
            for &val in &lane {
                let y = val - comp;
                let t = sum_abs + y;
                comp = (t - sum_abs) - y;
                sum_abs = t;
            }
            k += 4;
        }
        while k < period {
            let val = (*base.add(k) - sma).abs();
            let y = val - comp;
            let t = sum_abs + y;
            comp = (t - sum_abs) - y;
            sum_abs = t;
            k += 1;
        }
        let price0 = *data.get_unchecked(first_out);
        let denom = 0.015 * (sum_abs * inv_p);
        *out.get_unchecked_mut(first_out) = if denom == 0.0 {
            0.0
        } else {
            (price0 - sma) / denom
        };
    }

    // Rolling
    let mut i = first_out + 1;
    while i < n {
        let exiting = *data.get_unchecked(i - period);
        let entering = *data.get_unchecked(i);
        sum = sum - exiting + entering;
        sma = sum * inv_p;

        let start = i + 1 - period;
        let wptr = data.as_ptr().add(start);

        let vmean = _mm256_set1_pd(sma);
        let vsgn = _mm256_set1_pd(-0.0f64);
        let mut k = 0usize;
        let mut sum_abs = 0.0f64;
        let mut comp = 0.0f64;
        while k + 4 <= period {
            let x = _mm256_loadu_pd(wptr.add(k));
            let d = _mm256_sub_pd(x, vmean);
            let a = _mm256_andnot_pd(vsgn, d);
            let mut lane = [0.0f64; 4];
            _mm256_storeu_pd(lane.as_mut_ptr(), a);
            for &val in &lane {
                let y = val - comp;
                let t = sum_abs + y;
                comp = (t - sum_abs) - y;
                sum_abs = t;
            }
            k += 4;
        }
        while k < period {
            let val = (*wptr.add(k) - sma).abs();
            let y = val - comp;
            let t = sum_abs + y;
            comp = (t - sum_abs) - y;
            sum_abs = t;
            k += 1;
        }

        let denom = 0.015 * (sum_abs * inv_p);
        *out.get_unchecked_mut(i) = if denom == 0.0 {
            0.0
        } else {
            (entering - sma) / denom
        };
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cci_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    cci_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cci_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    cci_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn cci_avx512_impl(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    debug_assert!(data.len() == out.len());
    debug_assert!(period >= 1 && first_valid + period <= data.len());

    let n = data.len();
    let inv_p = 1.0 / (period as f64);
    let scale = (period as f64) * (1.0 / 0.015);

    // Initial sum (scalar)
    let base = data.as_ptr().add(first_valid);
    let mut sum = 0.0;
    {
        let mut k = 0usize;
        while k + 4 <= period {
            let x0 = *base.add(k + 0);
            let x1 = *base.add(k + 1);
            let x2 = *base.add(k + 2);
            let x3 = *base.add(k + 3);
            sum = sum + x0 + x1 + x2 + x3;
            k += 4;
        }
        while k < period {
            sum += *base.add(k);
            k += 1;
        }
    }

    let first_out = first_valid + period - 1;
    let mut sma = sum * inv_p;

    // abs mask (preserve NaNs)
    let pos_mask_i = _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64);
    let pos_mask = _mm512_castsi512_pd(pos_mask_i);

    // Initial MAD (vectorized compute, scalar Kahan accumulation in element order)
    {
        let vmean = _mm512_set1_pd(sma);
        let mut k = 0usize;
        let mut sum_abs = 0.0f64;
        let mut comp = 0.0f64;
        while k + 8 <= period {
            let x = _mm512_loadu_pd(base.add(k));
            let d = _mm512_sub_pd(x, vmean);
            let a = _mm512_and_pd(d, pos_mask); // abs
            let mut lane = [0.0f64; 8];
            _mm512_storeu_pd(lane.as_mut_ptr(), a);
            for &val in &lane {
                let y = val - comp;
                let t = sum_abs + y;
                comp = (t - sum_abs) - y;
                sum_abs = t;
            }
            k += 8;
        }
        while k < period {
            let val = (*base.add(k) - sma).abs();
            let y = val - comp;
            let t = sum_abs + y;
            comp = (t - sum_abs) - y;
            sum_abs = t;
            k += 1;
        }
        let price0 = *data.get_unchecked(first_out);
        let denom = 0.015 * (sum_abs * inv_p);
        *out.get_unchecked_mut(first_out) = if denom == 0.0 {
            0.0
        } else {
            (price0 - sma) / denom
        };
    }

    // Rolling
    let mut i = first_out + 1;
    while i < n {
        let exiting = *data.get_unchecked(i - period);
        let entering = *data.get_unchecked(i);
        sum = sum - exiting + entering;
        sma = sum * inv_p;

        let start = i + 1 - period;
        let wptr = data.as_ptr().add(start);

        let vmean = _mm512_set1_pd(sma);
        let mut k = 0usize;
        let mut sum_abs = 0.0f64;
        let mut comp = 0.0f64;
        while k + 8 <= period {
            let x = _mm512_loadu_pd(wptr.add(k));
            let d = _mm512_sub_pd(x, vmean);
            let a = _mm512_and_pd(d, pos_mask);
            let mut lane = [0.0f64; 8];
            _mm512_storeu_pd(lane.as_mut_ptr(), a);
            for &val in &lane {
                let y = val - comp;
                let t = sum_abs + y;
                comp = (t - sum_abs) - y;
                sum_abs = t;
            }
            k += 8;
        }
        while k < period {
            let val = (*wptr.add(k) - sma).abs();
            let y = val - comp;
            let t = sum_abs + y;
            comp = (t - sum_abs) - y;
            sum_abs = t;
            k += 1;
        }
        let denom = 0.015 * (sum_abs * inv_p);
        *out.get_unchecked_mut(i) = if denom == 0.0 {
            0.0
        } else {
            (entering - sma) / denom
        };
        i += 1;
    }
}

// ---- Row functions (all AVX variants point to correct level) ----

#[inline(always)]
pub unsafe fn cci_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    // Important: For batch processing, 'out' is a single row of the output matrix
    // with the same length as 'data'. We should write directly to the indices
    // corresponding to the data indices.
    cci_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cci_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    cci_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cci_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    cci_avx512(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cci_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    cci_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cci_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    cci_avx512_long(data, period, first, out)
}

// ---- Stream ----

// Decision: Streaming uses an order‑statistics treap for exact O(log n) MAD.
// Replaces prior O(n) scan per tick; API and warmup semantics unchanged.

#[derive(Debug, Clone)]
pub struct CciStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,

    // Rolling sum of the window (exact)
    sum: f64,

    // Precompute scale = period / 0.015 so CCI = (price - mean) * scale / sum_abs
    scale: f64,

    // Order statistics tree over the current window values
    ost: OrderStatsTreap,
}

// Simple deterministic SplitMix64 for treap priorities (fast, dependency-free)
#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[derive(Debug, Clone)]
struct OrderStatsTreap {
    root: Option<Box<Node>>,
    seed: u64,
}

#[derive(Debug, Clone)]
struct Node {
    key: f64,    // value
    prio: u64,   // heap priority
    cnt: u32,    // multiplicity of key
    size: usize, // total count in subtree (including multiplicities)
    sum: f64,    // sum of all values in subtree
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

#[inline(always)]
fn sz(t: &Option<Box<Node>>) -> usize {
    t.as_ref().map(|n| n.size).unwrap_or(0)
}
#[inline(always)]
fn sm(t: &Option<Box<Node>>) -> f64 {
    t.as_ref().map(|n| n.sum).unwrap_or(0.0)
}

impl Node {
    #[inline(always)]
    fn new(key: f64, prio: u64) -> Self {
        Self {
            key,
            prio,
            cnt: 1,
            size: 1,
            sum: key,
            left: None,
            right: None,
        }
    }
    #[inline(always)]
    fn recalc(&mut self) {
        self.size = self.cnt as usize + sz(&self.left) + sz(&self.right);
        self.sum = (self.key * self.cnt as f64) + sm(&self.left) + sm(&self.right);
    }
}

impl OrderStatsTreap {
    #[inline(always)]
    fn new() -> Self {
        // any non-zero seed; use address entropy
        let seed = splitmix64((&() as *const () as usize as u64) ^ 0xA5A5_A5A5_A5A5_A5A5);
        Self { root: None, seed }
    }

    #[inline(always)]
    fn next_prio(&mut self) -> u64 {
        self.seed = splitmix64(self.seed);
        self.seed
    }

    // Merge assumes all keys in 'a' <= all keys in 'b'
    fn merge(a: Option<Box<Node>>, b: Option<Box<Node>>) -> Option<Box<Node>> {
        match (a, b) {
            (None, t) | (t, None) => t,
            (Some(mut x), Some(mut y)) => {
                if x.prio > y.prio {
                    x.right = Self::merge(x.right.take(), Some(y));
                    x.recalc();
                    Some(x)
                } else {
                    y.left = Self::merge(Some(x), y.left.take());
                    y.recalc();
                    Some(y)
                }
            }
        }
    }

    // Split into (<= key, > key)
    fn split(mut t: Option<Box<Node>>, key: f64) -> (Option<Box<Node>>, Option<Box<Node>>) {
        match t.take() {
            None => (None, None),
            Some(mut n) => {
                if key < n.key {
                    let (l, r) = Self::split(n.left.take(), key);
                    n.left = r;
                    n.recalc();
                    (l, Some(n))
                } else {
                    let (l, r) = Self::split(n.right.take(), key);
                    n.right = l;
                    n.recalc();
                    (Some(n), r)
                }
            }
        }
    }

    // Insert one occurrence of key
    fn insert(&mut self, key: f64) {
        debug_assert!(key.is_finite());
        self.root = match self.root.take() {
            None => Some(Box::new(Node::new(key, self.next_prio()))),
            Some(mut _n) => {
                // put it back; recurse with helper
                self.root = Some(_n);
                Self::insert_into(self.root.take(), key, self.next_prio())
            }
        };
    }

    fn insert_into(t: Option<Box<Node>>, key: f64, prio: u64) -> Option<Box<Node>> {
        match t {
            None => Some(Box::new(Node::new(key, prio))),
            Some(mut n) => {
                if key == n.key {
                    n.cnt += 1;
                    n.recalc();
                    Some(n)
                } else if prio > n.prio {
                    let (l, r) = Self::split(Some(n), key);
                    let mut m = Box::new(Node::new(key, prio));
                    m.left = l;
                    m.right = r;
                    m.recalc();
                    Some(m)
                } else if key < n.key {
                    n.left = Self::insert_into(n.left.take(), key, prio);
                    n.recalc();
                    Some(n)
                } else {
                    n.right = Self::insert_into(n.right.take(), key, prio);
                    n.recalc();
                    Some(n)
                }
            }
        }
    }

    // Erase one occurrence of key (no-op if missing)
    fn erase(&mut self, key: f64) {
        debug_assert!(key.is_finite());
        self.root = Self::erase_from(self.root.take(), key);
    }

    fn erase_from(t: Option<Box<Node>>, key: f64) -> Option<Box<Node>> {
        match t {
            None => None,
            Some(mut n) => {
                if key == n.key {
                    if n.cnt > 1 {
                        n.cnt -= 1;
                        n.recalc();
                        Some(n)
                    } else {
                        Self::merge(n.left.take(), n.right.take())
                    }
                } else if key < n.key {
                    n.left = Self::erase_from(n.left.take(), key);
                    n.recalc();
                    Some(n)
                } else {
                    n.right = Self::erase_from(n.right.take(), key);
                    n.recalc();
                    Some(n)
                }
            }
        }
    }

    // Return (count_le, sum_le) for all keys <= 'key'
    fn prefix_le(&self, key: f64) -> (usize, f64) {
        debug_assert!(key.is_finite());
        let mut t = &self.root;
        let mut count = 0usize;
        let mut sum = 0.0f64;

        while let Some(n) = t.as_ref() {
            if key < n.key {
                t = &n.left;
            } else {
                // take left subtree + this node multiplicity
                count += sz(&n.left) + n.cnt as usize;
                sum += sm(&n.left) + (n.key * n.cnt as f64);
                t = &n.right;
            }
        }
        (count, sum)
    }

    #[inline(always)]
    fn size(&self) -> usize {
        sz(&self.root)
    }
}

impl CciStream {
    pub fn try_new(params: CciParams) -> Result<Self, CciError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(CciError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        // Initialize ring buffer with NaNs to match warmup behavior
        let buffer = alloc_with_nan_prefix(period, period);
        let scale = (period as f64) * (1.0 / 0.015);

        Ok(Self {
            period,
            buffer,
            head: 0,
            filled: false,
            sum: 0.0,
            scale,
            ost: OrderStatsTreap::new(),
        })
    }

    /// O(log n) exact update:
    /// - Maintain rolling sum and treap of window values.
    /// - Compute sum|x - mean| via prefix count/sum at 'mean'.
    /// - Return None until the buffer is filled (same semantics as before).
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        debug_assert!(value.is_finite(), "CCI stream expects finite inputs");

        // ring write
        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }

        if !self.filled {
            // warmup: accumulate partial sum and treap; no output yet
            self.sum += value;
            self.ost.insert(value);
            return None;
        }

        // rolling sum and treap maintenance
        if !old.is_nan() {
            self.sum -= old;
            self.ost.erase(old);
        }
        self.sum += value;
        self.ost.insert(value);

        // exact mean of the window
        let mean = self.sum / (self.period as f64);

        // prefix (<= mean): count & sum
        let (k_le, sum_le) = self.ost.prefix_le(mean);

        // Sum of absolute deviations using counts and sums:
        // sum_abs = mean*(2k - n) + (S - 2*sum_le)
        let n = self.period as f64;
        let sum_abs = mean.mul_add(2.0 * (k_le as f64) - n, self.sum - 2.0 * sum_le);

        if sum_abs == 0.0 {
            Some(0.0)
        } else {
            // CCI = (value - mean) / (0.015 * (sum_abs / n)) = (value - mean) * (n / (0.015 * sum_abs))
            Some((value - mean) * (self.scale / sum_abs))
        }
    }
}

// ---- Batch/Range ----

#[derive(Clone, Debug)]
pub struct CciBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CciBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 200, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CciBatchBuilder {
    range: CciBatchRange,
    kernel: Kernel,
}

impl CciBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<CciBatchOutput, CciError> {
        cci_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CciBatchOutput, CciError> {
        CciBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CciBatchOutput, CciError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CciBatchOutput, CciError> {
        CciBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "hlc3")
    }
}

#[derive(Clone, Debug)]
pub struct CciBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CciParams>,
    pub rows: usize,
    pub cols: usize,
}
impl CciBatchOutput {
    pub fn row_for_params(&self, p: &CciParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &CciParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CciBatchRange) -> Result<Vec<CciParams>, CciError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CciError> {
        // Zero step => static; start==end => singleton; support reversed bounds.
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                match cur.checked_add(step) {
                    Some(next) => {
                        if next == cur {
                            break;
                        }
                        cur = next;
                    }
                    None => break,
                }
            }
            if v.is_empty() { return Err(CciError::InvalidRange { start, end, step }); }
            Ok(v)
        } else {
            // reversed bounds
            let mut v = Vec::new();
            let mut cur = start;
            loop {
                v.push(cur);
                if cur <= end { break; }
                // prevent underflow
                if cur < step { break; }
                cur -= step;
                if cur == end { v.push(cur); break; }
                if cur < end { break; }
            }
            v.sort_unstable();
            v.dedup();
            if v.is_empty() { return Err(CciError::InvalidRange { start, end, step }); }
            Ok(v)
        }
    }
    let periods = axis_usize(r.period)?;
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(CciParams { period: Some(p) });
    }
    Ok(out)
}

pub fn cci_batch_with_kernel(
    data: &[f64],
    sweep: &CciBatchRange,
    k: Kernel,
) -> Result<CciBatchOutput, CciError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(CciError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    cci_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn cci_batch_slice(
    data: &[f64],
    sweep: &CciBatchRange,
    kern: Kernel,
) -> Result<CciBatchOutput, CciError> {
    cci_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn cci_batch_par_slice(
    data: &[f64],
    sweep: &CciBatchRange,
    kern: Kernel,
) -> Result<CciBatchOutput, CciError> {
    cci_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cci_batch_inner(
    data: &[f64],
    sweep: &CciBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CciBatchOutput, CciError> {
    if data.is_empty() {
        return Err(CciError::EmptyInputData);
    }
    let combos = expand_grid(sweep)?;
    let cols = data.len();
    let rows = combos.len();

    // Guard potential overflow in matrix allocation.
    rows.checked_mul(cols).ok_or(CciError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;

    // 1. Allocate *uninitialised* matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // 2. Fill the NaN warm-up prefix row-wise
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| data.iter().position(|x| !x.is_nan()).unwrap_or(0) + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // 3. Convert &[MaybeUninit<f64>] → &mut [f64]
    //    *after* the prefixes are written but *before*
    //    the compute kernels run – they will write the rest.
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // 4. Compute into it in place
    cci_batch_inner_into(data, sweep, kern, parallel, out)?;

    // 5. Reclaim the buffer as a normal Vec<f64> for the return value
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(CciBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn cci_batch_inner_into(
    data: &[f64],
    sweep: &CciBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CciParams>, CciError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(CciError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 });
    }

    if data.is_empty() {
        return Err(CciError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CciError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CciError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Checked matrix size and output slice length.
    let expected = rows
        .checked_mul(cols)
        .ok_or(CciError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    if out.len() != expected {
        return Err(CciError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    // Resolve kernel selection
    let kernel = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        Kernel::Scalar => Kernel::ScalarBatch,
        Kernel::Avx2 => Kernel::Avx2Batch,
        Kernel::Avx512 => Kernel::Avx512Batch,
        k => k,
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    // Initialize NaN prefixes for warmup periods
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Convert output slice to MaybeUninit for initialization
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    // Initialize the NaN prefixes row-wise
    for (row, &warmup) in warm.iter().enumerate() {
        let row_start = row * cols;
        for col in 0..warmup.min(cols) {
            out_uninit[row_start + col].write(f64::NAN);
        }
    }

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // Cast the row slice (which is definitely ours to mutate) to f64
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        // Ensure the output slice has the correct length
        assert_eq!(dst.len(), cols, "Output row length mismatch");

        match simd {
            Kernel::Scalar => cci_row_scalar(data, first, period, 0, std::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => cci_row_avx2(data, first, period, 0, std::ptr::null(), 0.0, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => cci_row_avx512(data, first, period, 0, std::ptr::null(), 0.0, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => {
                // Fallback to scalar when AVX is not available
                cci_row_scalar(data, first, period, 0, std::ptr::null(), 0.0, dst)
            }
            _ => unreachable!(),
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel && rows > 1 {
        out_uninit
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| do_row(row, out_row));
    } else {
        out_uninit
            .chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| do_row(row, out_row));
    }

    #[cfg(target_arch = "wasm32")]
    {
        out_uninit
            .chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| do_row(row, out_row));
    }

    Ok(combos)
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_cci_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = CciParams { period: None };
        let input_default = CciInput::from_candles(&candles, "close", default_params);
        let output_default = cci_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_20 = CciParams { period: Some(20) };
        let input_20 = CciInput::from_candles(&candles, "hl2", params_20);
        let output_20 = cci_with_kernel(&input_20, kernel)?;
        assert_eq!(output_20.values.len(), candles.close.len());

        let params_custom = CciParams { period: Some(9) };
        let input_custom = CciInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = cci_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cci_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CciInput::with_default_candles(&candles);
        let cci_result = cci_with_kernel(&input, kernel)?;
        assert_eq!(cci_result.values.len(), candles.close.len());

        let expected_last_five_cci = [
            -51.55252564125841,
            -43.50326506381541,
            -64.05117302269149,
            -39.05150631680948,
            -152.50523930896998,
        ];

        let start_idx = cci_result.values.len() - 5;
        let last_five_cci = &cci_result.values[start_idx..];
        for (i, &value) in last_five_cci.iter().enumerate() {
            let expected = expected_last_five_cci[i];
            assert!(
                (value - expected).abs() < 1e-6,
                "[{}] CCI mismatch at last five index {}: expected {}, got {}",
                test_name,
                i,
                expected,
                value
            );
        }
        let period: usize = input.get_period();
        for i in 0..(period - 1) {
            assert!(
                cci_result.values[i].is_nan(),
                "Expected NaN at index {} for initial period warm-up",
                i
            );
        }
        Ok(())
    }

    fn check_cci_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CciInput::with_default_candles(&candles);

        match input.data {
            CciData::Candles { source, .. } => {
                assert_eq!(source, "hlc3", "Expected default source to be 'hlc3'");
            }
            _ => panic!("Expected CciData::Candles variant"),
        }
        let output = cci_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cci_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CciParams { period: Some(0) };
        let input = CciInput::from_slice(&input_data, params);
        let res = cci_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cci_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CciParams { period: Some(10) };
        let input = CciInput::from_slice(&data_small, params);
        let res = cci_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cci_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CciParams { period: Some(9) };
        let input = CciInput::from_slice(&single_point, params);
        let res = cci_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cci_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = CciParams { period: Some(14) };
        let first_input = CciInput::from_candles(&candles, "close", first_params);
        let first_result = cci_with_kernel(&first_input, kernel)?;

        let second_params = CciParams { period: Some(14) };
        let second_input = CciInput::from_slice(&first_result.values, second_params);
        let second_result = cci_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 28 {
            for i in 28..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "Expected no NaN after index 28, found NaN at index {}",
                    i
                );
            }
        }
        Ok(())
    }

    fn check_cci_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CciInput::from_candles(&candles, "close", CciParams { period: Some(14) });
        let res = cci_with_kernel(&input, kernel)?;
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

    fn check_cci_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;
        let input = CciInput::from_candles(
            &candles,
            "close",
            CciParams {
                period: Some(period),
            },
        );
        let batch_output = cci_with_kernel(&input, kernel)?.values;

        let mut stream = CciStream::try_new(CciParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(cci_val) => stream_values.push(cci_val),
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
                "[{}] CCI streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_cci_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty_data: &[f64] = &[];
        let params = CciParams { period: Some(14) };
        let input = CciInput::from_slice(empty_data, params);
        let res = cci_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI should fail with empty input data",
            test_name
        );
        if let Err(e) = res {
            match e {
                CciError::EmptyInputData => {}
                other => panic!(
                    "[{}] Expected EmptyInputData error, got: {:?}",
                    test_name, other
                ),
            }
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_cci_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with default parameters
        let input = CciInput::from_candles(&candles, "close", CciParams::default());
        let output = cci_with_kernel(&input, kernel)?;

        // Also test with different parameters to increase coverage
        let params_20 = CciParams { period: Some(20) };
        let input_20 = CciInput::from_candles(&candles, "hlc3", params_20);
        let output_20 = cci_with_kernel(&input_20, kernel)?;

        // Check both outputs
        for output in [output, output_20] {
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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                        test_name, val, bits, i
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_cci_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_cci_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate test data with period and price data
        let strat = (1usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period)| {
            let params = CciParams {
                period: Some(period),
            };
            let input = CciInput::from_slice(&data, params);

            // Calculate CCI with the specified kernel
            let CciOutput { values: out } = cci_with_kernel(&input, kernel).unwrap();
            // Calculate reference output with scalar kernel for comparison
            let CciOutput { values: ref_out } = cci_with_kernel(&input, Kernel::Scalar).unwrap();

            // Test 1: Warmup period validation - first (period - 1) values should be NaN
            for i in 0..(period - 1) {
                prop_assert!(
                    out[i].is_nan(),
                    "[{}] Expected NaN at index {} during warmup period, got {}",
                    test_name,
                    i,
                    out[i]
                );
            }

            // Test 2: Non-NaN values after warmup
            for i in (period - 1)..data.len() {
                prop_assert!(
                    !out[i].is_nan(),
                    "[{}] Expected valid value at index {} after warmup, got NaN",
                    test_name,
                    i
                );
            }

            // Test 3: Kernel consistency - all kernels should produce identical results
            for i in 0..data.len() {
                let y = out[i];
                let r = ref_out[i];

                if y.is_nan() && r.is_nan() {
                    continue; // Both NaN is ok
                }

                // Use ULP comparison for floating point equality
                let y_bits = y.to_bits();
                let r_bits = r.to_bits();
                let ulp_diff = if y_bits > r_bits {
                    y_bits - r_bits
                } else {
                    r_bits - y_bits
                };

                prop_assert!(
                    ulp_diff <= 8,
                    "[{}] Kernel mismatch at index {}: {} != {} (ULP diff: {})",
                    test_name,
                    i,
                    y,
                    r,
                    ulp_diff
                );
            }

            // Test 4: Constant price property - when all prices are equal, CCI should be 0
            if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && data.len() >= period {
                for i in (period - 1)..data.len() {
                    prop_assert!(
                        out[i].abs() < 1e-9,
                        "[{}] CCI should be ~0 for constant prices, got {} at index {}",
                        test_name,
                        out[i],
                        i
                    );
                }
            }

            // Test 5: Mathematical correctness - verify CCI calculation
            for i in (period - 1)..data.len() {
                let window_start = i + 1 - period;
                let window = &data[window_start..=i];

                // Calculate SMA
                let sum: f64 = window.iter().sum();
                let sma = sum / period as f64;

                // Calculate Mean Absolute Deviation
                let mad: f64 = window.iter().map(|&x| (x - sma).abs()).sum::<f64>() / period as f64;

                // Calculate expected CCI
                let price = data[i];
                let expected_cci = if mad == 0.0 {
                    0.0
                } else {
                    (price - sma) / (0.015 * mad)
                };

                // Compare with actual output
                let actual_cci = out[i];
                let diff = (actual_cci - expected_cci).abs();

                prop_assert!(
                    diff < 1e-10,
                    "[{}] CCI calculation mismatch at index {}: expected {}, got {}, diff {}",
                    test_name,
                    i,
                    expected_cci,
                    actual_cci,
                    diff
                );
            }

            // Test 6: Edge case - period = 1
            if period == 1 {
                // With period=1, SMA is just the current value, MAD is 0, so CCI should be 0
                for i in 0..data.len() {
                    prop_assert!(
                        out[i].abs() < 1e-9,
                        "[{}] CCI should be ~0 for period=1, got {} at index {}",
                        test_name,
                        out[i],
                        i
                    );
                }
            }

            // Test 7: Reasonable bounds - CCI typically ranges between -300 and 300
            // (this is a soft check, extreme values are possible but rare)
            for i in (period - 1)..data.len() {
                if out[i].abs() > 500.0 {
                    // Just a warning, not a failure - extreme values are mathematically possible
                    eprintln!(
							"[{}] Warning: Extreme CCI value {} at index {} (typical range is -300 to 300)",
							test_name,
							out[i],
							i
						);
                }
            }

            // Test 8: Very small MAD values - ensure no overflow/underflow issues
            // When prices vary by tiny amounts, MAD can be very small, producing extreme CCIs
            for i in (period - 1)..data.len() {
                let window_start = i + 1 - period;
                let window = &data[window_start..=i];
                let sum: f64 = window.iter().sum();
                let sma = sum / period as f64;
                let mad: f64 = window.iter().map(|&x| (x - sma).abs()).sum::<f64>() / period as f64;

                // If MAD is extremely small but non-zero
                if mad > 0.0 && mad < 1e-12 {
                    let actual_cci = out[i];
                    // CCI should still be finite (not infinite or NaN)
                    prop_assert!(
							actual_cci.is_finite(),
							"[{}] CCI should be finite even with very small MAD ({}) at index {}, got {}",
							test_name,
							mad,
							i,
							actual_cci
						);

                    // The calculation should still be mathematically correct
                    let price = data[i];
                    let expected_cci = (price - sma) / (0.015 * mad);
                    let relative_error = ((actual_cci - expected_cci) / expected_cci).abs();

                    // Allow slightly more tolerance for extreme values due to floating point precision
                    prop_assert!(
							relative_error < 1e-8 || (actual_cci - expected_cci).abs() < 1e-10,
							"[{}] CCI calculation with small MAD at index {}: expected {}, got {}, relative error {}",
							test_name,
							i,
							expected_cci,
							actual_cci,
							relative_error
						);
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    macro_rules! generate_all_cci_tests {
        ($($test_fn:ident),*) => {
            paste! {
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

    generate_all_cci_tests!(
        check_cci_partial_params,
        check_cci_accuracy,
        check_cci_default_candles,
        check_cci_zero_period,
        check_cci_period_exceeds_length,
        check_cci_very_small_dataset,
        check_cci_reinput,
        check_cci_nan_handling,
        check_cci_streaming,
        check_cci_empty_input,
        check_cci_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_cci_tests!(check_cci_property);

    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_cci_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Build a small but non-trivial input with an initial NaN prefix.
        let mut data = Vec::with_capacity(256);
        data.extend_from_slice(&[f64::NAN, f64::NAN, f64::NAN]);
        for i in 0..253usize {
            let x = (i as f64 * 0.037).sin() * 5.0 + 100.0 + (i % 7) as f64 * 0.1;
            data.push(x);
        }

        let input = CciInput::from_slice(&data, CciParams::default());

        // Baseline via existing Vec-returning API.
        let baseline = cci(&input)?.values;

        // Preallocate output; function will write warmups as NaN.
        let mut out = vec![0.0; data.len()];
        cci_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for (i, (a, b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_nan(*a, *b),
                "cci_into mismatch at idx {}: baseline={}, into={}",
                i,
                a,
                b
            );
        }

        Ok(())
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = CciBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "hlc3")?;

        let def = CciParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            -51.55252564125841,
            -43.50326506381541,
            -64.05117302269149,
            -39.05150631680948,
            -152.50523930896998,
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations
        let output = CciBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10) // Test with periods 10, 20, 30
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
            paste! {
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
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "cci")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn cci_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = CciParams {
        period: Some(period),
    };
    let cci_in = CciInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function and convert to NumPy with zero-copy
    let result_vec: Vec<f64> = py
        .allow_threads(|| cci_with_kernel(&cci_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CciStream")]
pub struct CciStreamPy {
    stream: CciStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CciStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = CciParams {
            period: Some(period),
        };
        let stream =
            CciStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(CciStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated CCI value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cci_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn cci_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = CciBatchRange {
        period: period_range,
    };

    // Use the existing batch function that handles everything
    let output = py
        .allow_threads(|| cci_batch_with_kernel(slice_in, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Convert output to NumPy arrays
    let values_arr = output.values.into_pyarray(py);
    let reshaped = values_arr.reshape((output.rows, output.cols))?;

    // Build result dictionary
    let dict = PyDict::new(py);
    dict.set_item("values", reshaped)?;
    dict.set_item(
        "periods",
        output
            .combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// -------- Python CUDA bindings --------

// Dedicated CCI VRAM handle with CAI v3 + DLPack v1.x
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "CciDeviceArrayF32", unsendable)]
pub struct CciDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
    pub(crate) stream: usize,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl CciDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = &self.inner;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (inner.device_ptr() as usize, false))?;
        // CCI CUDA wrappers synchronize the producing stream before returning,
        // so CAI v3 permits omitting the "stream" key.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    pub fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__();
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
            DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "cci_cuda_batch_dev")]
#[pyo3(signature = (data, period_range, device_id=0))]
pub fn cci_cuda_batch_dev_py(
    py: Python<'_>,
    data: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<CciDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = data.as_slice()?;
    let sweep = CciBatchRange {
        period: period_range,
    };
    let (inner, dev_id, ctx, stream) = py.allow_threads(|| -> PyResult<_> {
        let cuda =
            CudaCci::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.context_arc();
        let out = cuda
            .cci_batch_dev(slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.stream()
            .synchronize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((out, dev_id, ctx, cuda.stream_handle_usize()))
    })?;
    Ok(CciDeviceArrayF32Py {
        inner,
        _ctx: ctx,
        device_id: dev_id,
        stream,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "cci_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm, cols, rows, period, device_id=0))]
pub fn cci_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    device_id: usize,
) -> PyResult<CciDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = data_tm.as_slice()?;
    let (inner, dev_id, ctx, stream) = py.allow_threads(|| -> PyResult<_> {
        let cuda =
            CudaCci::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.context_arc();
        let out = cuda
            .cci_many_series_one_param_time_major_dev(slice, cols, rows, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.stream()
            .synchronize()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((out, dev_id, ctx, cuda.stream_handle_usize()))
    })?;
    Ok(CciDeviceArrayF32Py {
        inner,
        _ctx: ctx,
        device_id: dev_id,
        stream,
    })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = CciParams {
        period: Some(period),
    };
    let input = CciInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    cci_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to cci_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = CciParams {
            period: Some(period),
        };
        let input = CciInput::from_slice(data, params);

        // Check for aliasing
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid aliasing issues
            let mut temp = vec![0.0; len];
            cci_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // Direct write to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            cci_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = CciBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    cci_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = CciBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CciBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CciBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CciParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cci_batch)]
pub fn cci_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    // 1. Deserialize the configuration object from JavaScript
    let config: CciBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = CciBatchRange {
        period: config.period_range,
    };

    // 2. Run the existing core logic
    let output = cci_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 3. Create the structured output
    let js_output = CciBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // 4. Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cci_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to cci_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = CciBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        cci_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
