//! # Time Series Forecast (TSF)
//!
//! TSF calculates linear regression over a rolling window and projects the next value,
//! providing a smoothed forecast of price direction based on recent trend.
//!
//! ## Parameters
//! - **period**: Linear regression window size. Defaults to 14.
//!
//! ## Returns
//! - **`Ok(TsfOutput)`** containing a `Vec<f64>` of forecasted values matching input length.
//! - **`Err(TsfError)`** on invalid parameters or insufficient data.
//!
//! ## Developer Notes
//! - **Scalar**: Optimized O(1) sliding window using S0=∑y and S1=∑(j·y);
//!   falls back to O(p) recompute only when NaNs enter/leave the window. Matches tests.
//! - **SIMD**: AVX2 is implemented (4-at-a-time vector algebra; sequential update retained).
//!   AVX512 is present but currently forwards to scalar for short/long variants. Runtime Auto
//!   short-circuits to `Scalar` because SIMD provides marginal/no gains for TSF’s sequential
//!   recurrence; explicit AVX2/AVX512 requests are honored for testing.
//! - **Batch**: Parallel per-row scalar path with warmup prefixes; SIMD selection short-circuits
//!   to scalar for the same reason as single-series.
//! - **Perf (reference)**: On `RUSTFLAGS="-C target-cpu=native"` single-series 100k:
//!   - Scalar: ~147–149 µs via `cargo bench --bench indicator_benchmark -- tsf/tsf_scalar/100k`
//!   - AVX2:   ~190–195 µs via `cargo +nightly bench --features nightly-avx --bench indicator_benchmark -- tsf/tsf_avx2/100k`
//!   These confirm that Auto should prefer the scalar path on this workload.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
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

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::{cuda::moving_averages::CudaTsf, indicators::moving_averages::alma::DeviceArrayF32Py};

impl<'a> AsRef<[f64]> for TsfInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TsfData::Slice(slice) => slice,
            TsfData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TsfData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TsfOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct TsfParams {
    pub period: Option<usize>,
}

impl Default for TsfParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct TsfInput<'a> {
    pub data: TsfData<'a>,
    pub params: TsfParams,
}

impl<'a> TsfInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: TsfParams) -> Self {
        Self {
            data: TsfData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TsfParams) -> Self {
        Self {
            data: TsfData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", TsfParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TsfBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for TsfBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TsfBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<TsfOutput, TsfError> {
        let p = TsfParams {
            period: self.period,
        };
        let i = TsfInput::from_candles(c, "close", p);
        tsf_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TsfOutput, TsfError> {
        let p = TsfParams {
            period: self.period,
        };
        let i = TsfInput::from_slice(d, p);
        tsf_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<TsfStream, TsfError> {
        let p = TsfParams {
            period: self.period,
        };
        TsfStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum TsfError {
    #[error("tsf: Input data slice is empty.")]
    EmptyInputData,
    #[error("tsf: All values are NaN.")]
    AllValuesNaN,
    #[error("tsf: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("tsf: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("tsf: Mismatched output length: expected = {expected}, actual = {actual}")]
    MismatchedOutputLen { expected: usize, actual: usize },
    #[error("tsf: Period must be at least 2 for linear regression, got {period}")]
    PeriodTooSmall { period: usize },
}

#[inline]
pub fn tsf(input: &TsfInput) -> Result<TsfOutput, TsfError> {
    tsf_with_kernel(input, Kernel::Auto)
}

pub fn tsf_with_kernel(input: &TsfInput, kernel: Kernel) -> Result<TsfOutput, TsfError> {
    let data: &[f64] = match &input.data {
        TsfData::Candles { candles, source } => source_type(candles, source),
        TsfData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(TsfError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TsfError::AllValuesNaN)?;
    let period = input.get_period();

    if period < 2 {
        return Err(TsfError::PeriodTooSmall { period });
    }
    if period > len {
        return Err(TsfError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(TsfError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // Short-circuit Auto to Scalar: AVX2/AVX512 underperform for TSF's sequential recurrence.
    // Explicit Avx2/Avx512 requests are still honored for testing.
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };
    let mut out = alloc_with_nan_prefix(len, first + period - 1);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => tsf_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => tsf_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => tsf_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(TsfOutput { values: out })
}

#[inline]
pub fn tsf_into_slice(dst: &mut [f64], input: &TsfInput, kern: Kernel) -> Result<(), TsfError> {
    let data: &[f64] = match &input.data {
        TsfData::Candles { candles, source } => source_type(candles, source),
        TsfData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(TsfError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TsfError::AllValuesNaN)?;
    let period = input.get_period();

    if period < 2 {
        return Err(TsfError::PeriodTooSmall { period });
    }
    if period > len {
        return Err(TsfError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(TsfError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if dst.len() != data.len() {
        return Err(TsfError::MismatchedOutputLen {
            expected: data.len(),
            actual: dst.len(),
        });
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => tsf_scalar(data, period, first, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => unsafe { tsf_avx2(data, period, first, dst) },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => unsafe { tsf_avx512(data, period, first, dst) },
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
            tsf_scalar(data, period, first, dst)
        }
        _ => unreachable!(),
    }

    // Fill warmup with NaN
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline(always)]
pub fn tsf_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    // Closed-form sums over x = 0..p-1
    let p = period;
    let n = data.len();
    if n == 0 {
        return;
    }

    let pf = p as f64;
    // Match streaming path numerics: compute ∑x and ∑x² via loops
    let mut sum_x = 0.0f64;
    let mut sum_x2 = 0.0f64;
    for x in 0..p {
        let xf = x as f64;
        sum_x += xf;
        sum_x2 += xf * xf;
    }
    let divisor = pf * sum_x2 - sum_x * sum_x;
    // Use direct divisions to mirror streaming path numerics

    // First index we are allowed to write (warmup prefix handled by caller)
    let mut base = first_val;
    let mut i = base + p - 1;
    if i >= n {
        return;
    }

    // --- initialize S0 = ∑y, S1 = ∑(j*y) for window [base .. base+p-1] ---
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut ok = true;
    for j in 0..p {
        let v = data[base + j];
        if v.is_nan() {
            s0 = f64::NAN;
            s1 = f64::NAN;
            ok = false;
            break; // propagate NaN just like the naive loop
        }
        s0 += v;
        s1 += (j as f64) * v;
    }

    // write forecast for the first full window
    if ok {
        let m = (pf * s1 - sum_x * s0) / divisor;
        let b = (s0 - m * sum_x) / pf;
        out[i] = b + m * pf;
    } else {
        out[i] = f64::NAN;
    }

    // --- slide window in O(1): S0' = S0 - y_old + y_new ; S1' = S1 + p*y_new - S0' ---
    while i + 1 < n {
        let y_old = data[base];
        let y_new = data[base + p];
        base += 1;
        i += 1;

        if s0.is_finite() && s1.is_finite() && y_old.is_finite() && y_new.is_finite() {
            // fast O(1) update
            let new_s0 = s0 + (y_new - y_old);
            let new_s1 = pf * y_new + s1 - new_s0;
            s0 = new_s0;
            s1 = new_s1;

            let m = (pf * s1 - sum_x * s0) / divisor;
            let b = (s0 - m * sum_x) / pf;
            out[i] = b + m * pf;
        } else {
            // Fallback: recompute sums for this window to recover after NaNs
            let mut r0 = 0.0f64;
            let mut r1 = 0.0f64;
            let mut clean = true;
            for j in 0..p {
                let v = data[base + j];
                if v.is_nan() {
                    r0 = f64::NAN;
                    r1 = f64::NAN;
                    clean = false;
                    break;
                }
                r0 += v;
                r1 += (j as f64) * v;
            }
            s0 = r0;
            s1 = r1;

            if clean {
                let m = (pf * s1 - sum_x * s0) / divisor;
                let b = (s0 - m * sum_x) / pf;
                out[i] = b + m * pf;
            } else {
                out[i] = f64::NAN;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn tsf_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe {
        if period <= 32 {
            tsf_avx512_short(data, period, first_valid, out);
        } else {
            tsf_avx512_long(data, period, first_valid, out);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub fn tsf_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe {
        let n = data.len();
        if n == 0 {
            return;
        }

        let p = period;
        let pf = p as f64;

        // Compute ∑x and ∑x² via loops to match scalar/streaming numerics
        let mut sum_x = 0.0f64;
        let mut sum_x2 = 0.0f64;
        for x in 0..p {
            let xf = x as f64;
            sum_x += xf;
            sum_x2 += xf * xf;
        }
        let divisor = pf * sum_x2 - sum_x * sum_x;

        let pfv = _mm256_set1_pd(pf);
        let sumxv = _mm256_set1_pd(sum_x);
        let divv = _mm256_set1_pd(divisor);

        let mut base = first_valid;
        let mut i = base + p - 1;
        if i >= n {
            return;
        }

        // Initialize S0, S1 for the first window [base .. base+p-1]
        let mut s0 = 0.0f64;
        let mut s1 = 0.0f64;
        let mut ok = true;
        for j in 0..p {
            let v = *data.get_unchecked(base + j);
            if v.is_nan() {
                s0 = f64::NAN;
                s1 = f64::NAN;
                ok = false;
                break;
            }
            s0 += v;
            s1 += (j as f64) * v;
        }

        // Emit first value (scalar arithmetic for exact parity)
        if ok {
            let m = (pf * s1 - sum_x * s0) / divisor;
            let b = (s0 - m * sum_x) / pf;
            *out.get_unchecked_mut(i) = b + m * pf;
        } else {
            *out.get_unchecked_mut(i) = f64::NAN;
        }

        // Main loop: process 4 outputs per iteration.
        // We advance the streaming state sequentially for 4 steps, then compute
        // the 4 results with AVX2 to accelerate the expensive divides.
        while i + 4 < n {
            // Precompute S0/S1 for the next 4 windows (i+1 .. i+4)
            let mut s0_buf = [0.0f64; 4];
            let mut s1_buf = [0.0f64; 4];
            let mut s0_k = s0;
            let mut s1_k = s1;

            // Step t = 0..3 produces window at i + (t+1)
            // t = 0
            let y_old0 = *data.get_unchecked(base);
            let y_new0 = *data.get_unchecked(base + p);
            if s0_k.is_finite() && s1_k.is_finite() && y_old0.is_finite() && y_new0.is_finite() {
                let ns0 = s0_k + (y_new0 - y_old0);
                let ns1 = pf * y_new0 + s1_k - ns0;
                s0_k = ns0;
                s1_k = ns1;
            } else {
                let mut r0 = 0.0f64;
                let mut r1 = 0.0f64;
                let mut clean = true;
                let b1 = base + 1;
                for j in 0..p {
                    let v = *data.get_unchecked(b1 + j);
                    if v.is_nan() {
                        r0 = f64::NAN;
                        r1 = f64::NAN;
                        clean = false;
                        break;
                    }
                    r0 += v;
                    r1 += (j as f64) * v;
                }
                s0_k = r0;
                s1_k = r1;
                let _ = clean;
            }
            s0_buf[0] = s0_k;
            s1_buf[0] = s1_k;

            // t = 1
            let y_old1 = *data.get_unchecked(base + 1);
            let y_new1 = *data.get_unchecked(base + p + 1);
            if s0_k.is_finite() && s1_k.is_finite() && y_old1.is_finite() && y_new1.is_finite() {
                let ns0 = s0_k + (y_new1 - y_old1);
                let ns1 = pf * y_new1 + s1_k - ns0;
                s0_k = ns0;
                s1_k = ns1;
            } else {
                let mut r0 = 0.0f64;
                let mut r1 = 0.0f64;
                let mut clean = true;
                let b2 = base + 2;
                for j in 0..p {
                    let v = *data.get_unchecked(b2 + j);
                    if v.is_nan() {
                        r0 = f64::NAN;
                        r1 = f64::NAN;
                        clean = false;
                        break;
                    }
                    r0 += v;
                    r1 += (j as f64) * v;
                }
                s0_k = r0;
                s1_k = r1;
                let _ = clean;
            }
            s0_buf[1] = s0_k;
            s1_buf[1] = s1_k;

            // t = 2
            let y_old2 = *data.get_unchecked(base + 2);
            let y_new2 = *data.get_unchecked(base + p + 2);
            if s0_k.is_finite() && s1_k.is_finite() && y_old2.is_finite() && y_new2.is_finite() {
                let ns0 = s0_k + (y_new2 - y_old2);
                let ns1 = pf * y_new2 + s1_k - ns0;
                s0_k = ns0;
                s1_k = ns1;
            } else {
                let mut r0 = 0.0f64;
                let mut r1 = 0.0f64;
                let mut clean = true;
                let b3 = base + 3;
                for j in 0..p {
                    let v = *data.get_unchecked(b3 + j);
                    if v.is_nan() {
                        r0 = f64::NAN;
                        r1 = f64::NAN;
                        clean = false;
                        break;
                    }
                    r0 += v;
                    r1 += (j as f64) * v;
                }
                s0_k = r0;
                s1_k = r1;
                let _ = clean;
            }
            s0_buf[2] = s0_k;
            s1_buf[2] = s1_k;

            // t = 3
            let y_old3 = *data.get_unchecked(base + 3);
            let y_new3 = *data.get_unchecked(base + p + 3);
            if s0_k.is_finite() && s1_k.is_finite() && y_old3.is_finite() && y_new3.is_finite() {
                let ns0 = s0_k + (y_new3 - y_old3);
                let ns1 = pf * y_new3 + s1_k - ns0;
                s0_k = ns0;
                s1_k = ns1;
            } else {
                let mut r0 = 0.0f64;
                let mut r1 = 0.0f64;
                let mut clean = true;
                let b4 = base + 4;
                for j in 0..p {
                    let v = *data.get_unchecked(b4 + j);
                    if v.is_nan() {
                        r0 = f64::NAN;
                        r1 = f64::NAN;
                        clean = false;
                        break;
                    }
                    r0 += v;
                    r1 += (j as f64) * v;
                }
                s0_k = r0;
                s1_k = r1;
                let _ = clean;
            }
            s0_buf[3] = s0_k;
            s1_buf[3] = s1_k;

            // Vector compute 4 outputs for i+1..i+4
            let s0v = _mm256_loadu_pd(s0_buf.as_ptr());
            let s1v = _mm256_loadu_pd(s1_buf.as_ptr());
            let num = _mm256_sub_pd(_mm256_mul_pd(pfv, s1v), _mm256_mul_pd(sumxv, s0v));
            let mv = _mm256_div_pd(num, divv);
            let bv = _mm256_div_pd(_mm256_sub_pd(s0v, _mm256_mul_pd(mv, sumxv)), pfv);
            let outv = _mm256_add_pd(bv, _mm256_mul_pd(mv, pfv));
            _mm256_storeu_pd(out.as_mut_ptr().add(i + 1), outv);

            // Advance base/i and set current state to window at i+4
            s0 = s0_k;
            s1 = s1_k;
            base += 4;
            i += 4;
        }

        // Tail: advance one-by-one (scalar O(1) update)
        while i + 1 < n {
            let y_old = *data.get_unchecked(base);
            let y_new = *data.get_unchecked(base + p);
            base += 1;
            i += 1;

            if s0.is_finite() && s1.is_finite() && y_old.is_finite() && y_new.is_finite() {
                let ns0 = s0 + (y_new - y_old);
                let ns1 = pf * y_new + s1 - ns0;
                s0 = ns0;
                s1 = ns1;
                let m = (pf * s1 - sum_x * s0) / divisor;
                let b = (s0 - m * sum_x) / pf;
                *out.get_unchecked_mut(i) = b + m * pf;
            } else {
                let mut r0 = 0.0f64;
                let mut r1 = 0.0f64;
                let mut clean = true;
                for j in 0..p {
                    let v = *data.get_unchecked(base + j - 1); // base already advanced
                    if v.is_nan() {
                        r0 = f64::NAN;
                        r1 = f64::NAN;
                        clean = false;
                        break;
                    }
                    r0 += v;
                    r1 += (j as f64) * v;
                }
                s0 = r0;
                s1 = r1;
                if clean {
                    let m = (pf * s1 - sum_x * s0) / divisor;
                    let b = (s0 - m * sum_x) / pf;
                    *out.get_unchecked_mut(i) = b + m * pf;
                } else {
                    *out.get_unchecked_mut(i) = f64::NAN;
                }
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsf_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    tsf_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsf_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    tsf_scalar(data, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct TsfStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,

    // precomputed invariants for x = 0..p-1
    pf: f64,    // p as f64
    sum_x: f64, // ∑ x
    sum_x_sqr: f64,
    divisor: f64, // p*∑x² - (∑x)²

    // sliding accumulators (when window is clean)
    s0: f64, // ∑ y
    s1: f64, // ∑ (j*y)

    // counts NaNs currently inside the window (valid only after filled==true)
    nan_count: usize,
}

impl TsfStream {
    pub fn try_new(params: TsfParams) -> Result<Self, TsfError> {
        let period = params.period.unwrap_or(14);
        if period < 2 {
            return Err(TsfError::PeriodTooSmall { period });
        }

        // Precompute ∑x and ∑x² for x = 0..period-1 (loop form to mirror scalar path numerics)
        let pf = period as f64;
        let mut sum_x = 0.0f64;
        let mut sum_x_sqr = 0.0f64;
        for x in 0..period {
            let xf = x as f64;
            sum_x += xf;
            sum_x_sqr += xf * xf;
        }
        let divisor = pf * sum_x_sqr - (sum_x * sum_x);

        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            pf,
            sum_x,
            sum_x_sqr,
            divisor,
            s0: 0.0,
            s1: 0.0,
            nan_count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // 1) Warmup: fill the ring; when full, initialize sums once and return first forecast
        if !self.filled {
            self.buffer[self.head] = value;
            self.advance_head();
            if self.head == 0 {
                self.filled = true;

                // Compute S0/S1 once from ring and initialize nan_count.
                let (s0, s1, nan_count) = self.recompute_from_ring_checked();
                self.s0 = s0;
                self.s1 = s1;
                self.nan_count = nan_count;

                if self.nan_count > 0 || !self.s0.is_finite() || !self.s1.is_finite() {
                    return Some(f64::NAN);
                }

                // First forecast after warmup: forecast = b + m*pf
                let m = (self.pf * self.s1 - self.sum_x * self.s0) / self.divisor;
                let b = (self.s0 - m * self.sum_x) / self.pf;
                return Some(b + m * self.pf);
            }
            return None;
        }

        // 2) Steady state: slide the window by one.
        let y_old = self.buffer[self.head];
        let y_new = value;
        self.buffer[self.head] = y_new;

        // Maintain NaN count in O(1).
        let prev_nan_count = self.nan_count;
        if y_old.is_nan() {
            self.nan_count = self.nan_count.saturating_sub(1);
        }
        if y_new.is_nan() {
            self.nan_count = self.nan_count.saturating_add(1);
        }

        let out = if self.nan_count == 0 {
            // Window is clean now.
            if prev_nan_count == 0
                && self.s0.is_finite()
                && self.s1.is_finite()
                && y_old.is_finite()
                && y_new.is_finite()
            {
                // O(1) update for S0 and S1
                let new_s0 = self.s0 + (y_new - y_old);
                // S1' = S1 + p*y_new - S0'   (with j=0..p-1 indexing)
                let new_s1 = self.pf * y_new + self.s1 - new_s0;
                self.s0 = new_s0;
                self.s1 = new_s1;
            } else {
                // Window just became clean (or state was contaminated): one O(p) recovery.
                let (s0, s1) = self.recompute_from_ring_clean();
                self.s0 = s0;
                self.s1 = s1;
            }

            // Forecast = b + m*pf with standard OLS closed form
            let m = (self.pf * self.s1 - self.sum_x * self.s0) / self.divisor;
            let b = (self.s0 - m * self.sum_x) / self.pf;
            b + m * self.pf
        } else {
            // Window contains at least one NaN → forecast is NaN; keep S0/S1 in a poisoned state.
            self.s0 = f64::NAN;
            self.s1 = f64::NAN;
            f64::NAN
        };

        self.advance_head();
        Some(out)
    }

    #[inline(always)]
    fn advance_head(&mut self) {
        // Branchy wrap avoids a modulo in the hot path.
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
    }

    // Recompute assuming the window is clean (no NaNs) – faster than the checked version.
    #[inline(always)]
    fn recompute_from_ring_clean(&self) -> (f64, f64) {
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        let mut idx = self.head;
        for j in 0..self.period {
            let v = self.buffer[idx];
            // under contract: no NaNs here
            s0 += v;
            s1 += (j as f64) * v;
            idx += 1;
            if idx == self.period {
                idx = 0;
            }
        }
        (s0, s1)
    }

    // Recompute with NaN detection; returns (S0,S1,nan_count).
    #[inline(always)]
    fn recompute_from_ring_checked(&self) -> (f64, f64, usize) {
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        let mut cnt = 0usize;
        let mut idx = self.head;
        for j in 0..self.period {
            let v = self.buffer[idx];
            if v.is_nan() {
                cnt += 1;
            } else {
                s0 += v;
                s1 += (j as f64) * v;
            }
            idx += 1;
            if idx == self.period {
                idx = 0;
            }
        }
        if cnt > 0 {
            (f64::NAN, f64::NAN, cnt)
        } else {
            (s0, s1, 0)
        }
    }
}

#[derive(Clone, Debug)]
pub struct TsfBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for TsfBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TsfBatchBuilder {
    range: TsfBatchRange,
    kernel: Kernel,
}

impl TsfBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<TsfBatchOutput, TsfError> {
        tsf_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TsfBatchOutput, TsfError> {
        TsfBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TsfBatchOutput, TsfError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<TsfBatchOutput, TsfError> {
        TsfBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn tsf_batch_with_kernel(
    data: &[f64],
    sweep: &TsfBatchRange,
    k: Kernel,
) -> Result<TsfBatchOutput, TsfError> {
    let kernel = match k {
        // Short-circuit to ScalarBatch for TSF: SIMD underperforms for the sequential recurrence.
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => {
            return Err(TsfError::InvalidPeriod {
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
    tsf_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TsfBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TsfParams>,
    pub rows: usize,
    pub cols: usize,
}
impl TsfBatchOutput {
    pub fn row_for_params(&self, p: &TsfParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }

    pub fn values_for(&self, p: &TsfParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TsfBatchRange) -> Vec<TsfParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let periods = axis_usize(r.period);

    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(TsfParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn tsf_batch_slice(
    data: &[f64],
    sweep: &TsfBatchRange,
    kern: Kernel,
) -> Result<TsfBatchOutput, TsfError> {
    tsf_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn tsf_batch_par_slice(
    data: &[f64],
    sweep: &TsfBatchRange,
    kern: Kernel,
) -> Result<TsfBatchOutput, TsfError> {
    tsf_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn tsf_batch_inner(
    data: &[f64],
    sweep: &TsfBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TsfBatchOutput, TsfError> {
    // Build the list of TsfParams to run over
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TsfError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    // Find first non‐NaN index
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TsfError::AllValuesNaN)?;
    // Compute the maximum period required by any combo
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(TsfError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut sum_xs = vec![0.0; rows];
    let mut divisors = vec![0.0; rows];

    // Precompute ∑x and divisor for each "period = combos[row].period"
    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let sum_x = (0..period).map(|x| x as f64).sum::<f64>();
        let sum_x2 = (0..period).map(|x| (x as f64) * (x as f64)).sum::<f64>();
        let divisor = (period as f64 * sum_x2) - (sum_x * sum_x);
        sum_xs[row] = sum_x;
        divisors[row] = divisor;
    }

    // Use uninitialized memory like ALMA
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Compute warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Use ManuallyDrop pattern for safe conversion
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Closure that computes one row into out_row
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let sum_x = sum_xs[row];
        let divisor = divisors[row];

        match kern {
            Kernel::Scalar => tsf_row_scalar(data, first, period, sum_x, divisor, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => tsf_row_avx2(data, first, period, sum_x, divisor, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => tsf_row_avx512(data, first, period, sum_x, divisor, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => {
                tsf_row_scalar(data, first, period, sum_x, divisor, out_row)
            }
            _ => unreachable!(),
        }
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

    // Convert back to Vec using ManuallyDrop pattern
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(TsfBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// Helper function for Python batch bindings - writes directly to the provided buffer
#[inline(always)]
fn tsf_batch_inner_into(
    data: &[f64],
    sweep: &TsfBatchRange,
    kern: Kernel,
    parallel: bool,
    output: &mut [f64],
) -> Result<Vec<TsfParams>, TsfError> {
    // Build combos
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TsfError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    // First valid index and max window
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TsfError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(TsfError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Precompute ∑x and divisor per row
    let mut sum_xs = vec![0.0; rows];
    let mut divisors = vec![0.0; rows];
    for (row, prm) in combos.iter().enumerate() {
        let p = prm.period.unwrap();
        let sum_x = (0..p).map(|x| x as f64).sum::<f64>();
        let sum_x2 = (0..p).map(|x| (x as f64) * (x as f64)).sum::<f64>();
        sum_xs[row] = sum_x;
        divisors[row] = (p as f64 * sum_x2) - (sum_x * sum_x);
    }

    // Initialize only warmup prefixes to NaN using helper, zero extra writes
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut MaybeUninit<f64>, output.len())
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    // Row kernel
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = combos[row].period.unwrap();
        let sum_x = sum_xs[row];
        let div = divisors[row];
        match kern {
            Kernel::Scalar => tsf_row_scalar(data, first, p, sum_x, div, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => tsf_row_avx2(data, first, p, sum_x, div, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => tsf_row_avx512(data, first, p, sum_x, div, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => tsf_row_scalar(data, first, p, sum_x, div, out_row),
            _ => unreachable!(),
        }
    };

    // Compute rows into output
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            output
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, sl)| do_row(r, sl));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, sl) in output.chunks_mut(cols).enumerate() {
                do_row(r, sl);
            }
        }
    } else {
        for (r, sl) in output.chunks_mut(cols).enumerate() {
            do_row(r, sl);
        }
    }

    Ok(combos)
}

// 2) tsf_row_scalar (fixed indexing so j=0→oldest, j=period−1→newest)
#[inline(always)]
unsafe fn tsf_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    sum_x: f64,
    divisor: f64,
    out: &mut [f64],
) {
    // Keep signature unsafe (module contract), but use safe indexing internally.
    let n = data.len();
    if n == 0 {
        return;
    }

    let p = period;
    let pf = p as f64;
    // Use direct divisions to mirror streaming path numerics

    // First full window index
    let mut base = first;
    let mut i = base + p - 1;
    if i >= n {
        return;
    }

    // Initialize S0, S1 on the first full window
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut ok = true;
    for j in 0..p {
        let v = data[base + j];
        if v.is_nan() {
            s0 = f64::NAN;
            s1 = f64::NAN;
            ok = false;
            break;
        }
        s0 += v;
        s1 += (j as f64) * v;
    }

    if ok {
        let m = (pf * s1 - sum_x * s0) / divisor;
        let b = (s0 - m * sum_x) / pf;
        out[i] = b + m * pf;
    } else {
        out[i] = f64::NAN;
    }

    // Slide across the row
    while i + 1 < n {
        let y_old = data[base];
        let y_new = data[base + p];
        base += 1;
        i += 1;

        if s0.is_finite() && s1.is_finite() && y_old.is_finite() && y_new.is_finite() {
            let new_s0 = s0 + (y_new - y_old);
            let new_s1 = pf * y_new + s1 - new_s0;
            s0 = new_s0;
            s1 = new_s1;

            let m = (pf * s1 - sum_x * s0) / divisor;
            let b = (s0 - m * sum_x) / pf;
            out[i] = b + m * pf;
        } else {
            // Recompute to recover from NaNs
            let mut r0 = 0.0f64;
            let mut r1 = 0.0f64;
            let mut clean = true;
            for j in 0..p {
                let v = data[base + j];
                if v.is_nan() {
                    r0 = f64::NAN;
                    r1 = f64::NAN;
                    clean = false;
                    break;
                }
                r0 += v;
                r1 += (j as f64) * v;
            }
            s0 = r0;
            s1 = r1;

            if clean {
                let m = (pf * s1 - sum_x * s0) / divisor;
                let b = (s0 - m * sum_x) / pf;
                out[i] = b + m * pf;
            } else {
                out[i] = f64::NAN;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn tsf_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    sum_x: f64,
    divisor: f64,
    out: &mut [f64],
) {
    // Vectorized arithmetic for 4-at-a-time outputs; sliding updates remain sequential for correctness.
    let n = data.len();
    if n == 0 {
        return;
    }
    let p = period;
    let pf = p as f64;

    let pfv = _mm256_set1_pd(pf);
    let sumxv = _mm256_set1_pd(sum_x);
    let divv = _mm256_set1_pd(divisor);

    let mut base = first;
    let mut i = base + p - 1;
    if i >= n {
        return;
    }

    // Init sums for the first window
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut ok = true;
    for j in 0..p {
        let v = *data.get_unchecked(base + j);
        if v.is_nan() {
            s0 = f64::NAN;
            s1 = f64::NAN;
            ok = false;
            break;
        }
        s0 += v;
        s1 += (j as f64) * v;
    }
    if ok {
        let m = (pf * s1 - sum_x * s0) / divisor;
        let b = (s0 - m * sum_x) / pf;
        *out.get_unchecked_mut(i) = b + m * pf;
    } else {
        *out.get_unchecked_mut(i) = f64::NAN;
    }

    while i + 4 < n {
        let mut s0_buf = [0.0f64; 4];
        let mut s1_buf = [0.0f64; 4];
        let mut s0_k = s0;
        let mut s1_k = s1;

        // t = 0
        let y_old0 = *data.get_unchecked(base);
        let y_new0 = *data.get_unchecked(base + p);
        if s0_k.is_finite() && s1_k.is_finite() && y_old0.is_finite() && y_new0.is_finite() {
            let ns0 = s0_k + (y_new0 - y_old0);
            let ns1 = pf * y_new0 + s1_k - ns0;
            s0_k = ns0;
            s1_k = ns1;
        } else {
            let mut r0 = 0.0f64;
            let mut r1 = 0.0f64;
            let mut clean = true;
            let b1 = base + 1;
            for j in 0..p {
                let v = *data.get_unchecked(b1 + j);
                if v.is_nan() {
                    r0 = f64::NAN;
                    r1 = f64::NAN;
                    clean = false;
                    break;
                }
                r0 += v;
                r1 += (j as f64) * v;
            }
            s0_k = r0;
            s1_k = r1;
            let _ = clean;
        }
        s0_buf[0] = s0_k;
        s1_buf[0] = s1_k;

        // t = 1
        let y_old1 = *data.get_unchecked(base + 1);
        let y_new1 = *data.get_unchecked(base + p + 1);
        if s0_k.is_finite() && s1_k.is_finite() && y_old1.is_finite() && y_new1.is_finite() {
            let ns0 = s0_k + (y_new1 - y_old1);
            let ns1 = pf * y_new1 + s1_k - ns0;
            s0_k = ns0;
            s1_k = ns1;
        } else {
            let mut r0 = 0.0f64;
            let mut r1 = 0.0f64;
            let mut clean = true;
            let b2 = base + 2;
            for j in 0..p {
                let v = *data.get_unchecked(b2 + j);
                if v.is_nan() {
                    r0 = f64::NAN;
                    r1 = f64::NAN;
                    clean = false;
                    break;
                }
                r0 += v;
                r1 += (j as f64) * v;
            }
            s0_k = r0;
            s1_k = r1;
            let _ = clean;
        }
        s0_buf[1] = s0_k;
        s1_buf[1] = s1_k;

        // t = 2
        let y_old2 = *data.get_unchecked(base + 2);
        let y_new2 = *data.get_unchecked(base + p + 2);
        if s0_k.is_finite() && s1_k.is_finite() && y_old2.is_finite() && y_new2.is_finite() {
            let ns0 = s0_k + (y_new2 - y_old2);
            let ns1 = pf * y_new2 + s1_k - ns0;
            s0_k = ns0;
            s1_k = ns1;
        } else {
            let mut r0 = 0.0f64;
            let mut r1 = 0.0f64;
            let mut clean = true;
            let b3 = base + 3;
            for j in 0..p {
                let v = *data.get_unchecked(b3 + j);
                if v.is_nan() {
                    r0 = f64::NAN;
                    r1 = f64::NAN;
                    clean = false;
                    break;
                }
                r0 += v;
                r1 += (j as f64) * v;
            }
            s0_k = r0;
            s1_k = r1;
            let _ = clean;
        }
        s0_buf[2] = s0_k;
        s1_buf[2] = s1_k;

        // t = 3
        let y_old3 = *data.get_unchecked(base + 3);
        let y_new3 = *data.get_unchecked(base + p + 3);
        if s0_k.is_finite() && s1_k.is_finite() && y_old3.is_finite() && y_new3.is_finite() {
            let ns0 = s0_k + (y_new3 - y_old3);
            let ns1 = pf * y_new3 + s1_k - ns0;
            s0_k = ns0;
            s1_k = ns1;
        } else {
            let mut r0 = 0.0f64;
            let mut r1 = 0.0f64;
            let mut clean = true;
            let b4 = base + 4;
            for j in 0..p {
                let v = *data.get_unchecked(b4 + j);
                if v.is_nan() {
                    r0 = f64::NAN;
                    r1 = f64::NAN;
                    clean = false;
                    break;
                }
                r0 += v;
                r1 += (j as f64) * v;
            }
            s0_k = r0;
            s1_k = r1;
            let _ = clean;
        }
        s0_buf[3] = s0_k;
        s1_buf[3] = s1_k;

        // Vector compute and store for i+1..i+4
        let s0v = _mm256_loadu_pd(s0_buf.as_ptr());
        let s1v = _mm256_loadu_pd(s1_buf.as_ptr());
        let num = _mm256_sub_pd(_mm256_mul_pd(pfv, s1v), _mm256_mul_pd(sumxv, s0v));
        let mv = _mm256_div_pd(num, divv);
        let bv = _mm256_div_pd(_mm256_sub_pd(s0v, _mm256_mul_pd(mv, sumxv)), pfv);
        let outv = _mm256_add_pd(bv, _mm256_mul_pd(mv, pfv));
        _mm256_storeu_pd(out.as_mut_ptr().add(i + 1), outv);

        s0 = s0_k;
        s1 = s1_k;
        base += 4;
        i += 4;
    }

    // Tail
    while i + 1 < n {
        let y_old = *data.get_unchecked(base);
        let y_new = *data.get_unchecked(base + p);
        base += 1;
        i += 1;

        if s0.is_finite() && s1.is_finite() && y_old.is_finite() && y_new.is_finite() {
            let ns0 = s0 + (y_new - y_old);
            let ns1 = pf * y_new + s1 - ns0;
            s0 = ns0;
            s1 = ns1;
            let m = (pf * s1 - sum_x * s0) / divisor;
            let b = (s0 - m * sum_x) / pf;
            *out.get_unchecked_mut(i) = b + m * pf;
        } else {
            let mut r0 = 0.0f64;
            let mut r1 = 0.0f64;
            let mut clean = true;
            for j in 0..p {
                let v = *data.get_unchecked(base + j - 1);
                if v.is_nan() {
                    r0 = f64::NAN;
                    r1 = f64::NAN;
                    clean = false;
                    break;
                }
                r0 += v;
                r1 += (j as f64) * v;
            }
            s0 = r0;
            s1 = r1;
            if clean {
                let m = (pf * s1 - sum_x * s0) / divisor;
                let b = (s0 - m * sum_x) / pf;
                *out.get_unchecked_mut(i) = b + m * pf;
            } else {
                *out.get_unchecked_mut(i) = f64::NAN;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn tsf_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    sum_x: f64,
    divisor: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        tsf_row_avx512_short(data, first, period, sum_x, divisor, out);
    } else {
        tsf_row_avx512_long(data, first, period, sum_x, divisor, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn tsf_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    sum_x: f64,
    divisor: f64,
    out: &mut [f64],
) {
    tsf_row_scalar(data, first, period, sum_x, divisor, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn tsf_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    sum_x: f64,
    divisor: f64,
    out: &mut [f64],
) {
    tsf_row_scalar(data, first, period, sum_x, divisor, out)
}

// ================================
// Python Bindings
// ================================

#[cfg(feature = "python")]
#[pyfunction(name = "tsf")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn tsf_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = TsfParams {
        period: Some(period),
    };
    let tsf_in = TsfInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| tsf_with_kernel(&tsf_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TsfStream")]
pub struct TsfStreamPy {
    stream: TsfStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TsfStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = TsfParams {
            period: Some(period),
        };
        let stream =
            TsfStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TsfStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "tsf_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn tsf_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = TsfBatchRange {
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
            tsf_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

// ================================
// WASM Bindings
// ================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsf_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = TsfParams {
        period: Some(period),
    };
    let input = TsfInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    tsf_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsf_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsf_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tsf_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = TsfParams {
            period: Some(period),
        };
        let input = TsfInput::from_slice(data, params);

        let kernel = detect_best_kernel();
        if in_ptr == out_ptr {
            // Handle aliasing case
            let mut temp = vec![0.0; len];
            tsf_into_slice(&mut temp, &input, kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            tsf_into_slice(out, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    Ok(())
}

// ================= Python CUDA Bindings =================

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "tsf_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn tsf_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
)
-> PyResult<(DeviceArrayF32Py, Bound<'py, pyo3::types::PyDict>)> {
    use crate::cuda::cuda_available;
    use numpy::IntoPyArray;
    use pyo3::types::PyDict;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = TsfBatchRange { period: period_range };

    let (inner, combos) = py.allow_threads(|| {
        let cuda = CudaTsf::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.tsf_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    let periods: Vec<u64> = combos.iter().map(|c| c.period.unwrap() as u64).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;

    Ok((DeviceArrayF32Py { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "tsf_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn tsf_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
)
-> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = TsfParams { period: Some(period) };

    let inner = py.allow_threads(|| {
        let cuda = CudaTsf::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.tsf_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TsfBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TsfBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TsfParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tsf_batch)]
pub fn tsf_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: TsfBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = TsfBatchRange {
        period: config.period_range,
    };

    let kernel = detect_best_batch_kernel();
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar, // Fallback for WASM
    };
    let output = tsf_batch_inner(data, &sweep, simd, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = TsfBatchJsOutput {
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
pub fn tsf_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to tsf_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = TsfBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        if rows == 0 {
            return Err(JsValue::from_str("No valid parameter combinations"));
        }

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        let kernel = detect_best_batch_kernel();
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };

        tsf_batch_inner_into(data, &sweep, simd, true, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_tsf_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = TsfParams { period: None };
        let input = TsfInput::from_candles(&candles, "close", default_params);
        let output = tsf_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_tsf_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TsfInput::from_candles(&candles, "close", TsfParams::default());
        let result = tsf_with_kernel(&input, kernel)?;
        let expected_last_five = [
            58846.945054945056,
            58818.83516483516,
            58854.57142857143,
            59083.846153846156,
            58962.25274725275,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] TSF {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_tsf_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TsfInput::with_default_candles(&candles);
        match input.data {
            TsfData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TsfData::Candles"),
        }
        let output = tsf_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_tsf_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TsfParams { period: Some(0) };
        let input = TsfInput::from_slice(&input_data, params);
        let res = tsf_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TSF should fail with zero period",
            test_name
        );
        if let Err(e) = res {
            assert!(matches!(e, TsfError::PeriodTooSmall { period: 0 }));
        }
        Ok(())
    }

    fn check_tsf_period_one(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let params = TsfParams { period: Some(1) };
        let input = TsfInput::from_slice(&input_data, params);
        let res = tsf_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TSF should fail with period=1",
            test_name
        );
        if let Err(e) = res {
            assert!(matches!(e, TsfError::PeriodTooSmall { period: 1 }));
        }
        Ok(())
    }

    fn check_tsf_mismatched_output_len(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let params = TsfParams { period: Some(3) };
        let input = TsfInput::from_slice(&input_data, params);

        // Destination slice has wrong length
        let mut dst = vec![0.0; 10]; // Wrong size (should be 5)
        let res = tsf_into_slice(&mut dst, &input, kernel);

        assert!(
            res.is_err(),
            "[{}] TSF should fail with mismatched output length",
            test_name
        );
        if let Err(e) = res {
            assert!(matches!(
                e,
                TsfError::MismatchedOutputLen {
                    expected: 5,
                    actual: 10
                }
            ));
        }
        Ok(())
    }

    fn check_tsf_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TsfParams { period: Some(10) };
        let input = TsfInput::from_slice(&data_small, params);
        let res = tsf_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TSF should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_tsf_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TsfParams { period: Some(9) };
        let input = TsfInput::from_slice(&single_point, params);
        let res = tsf_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TSF should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_tsf_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = TsfParams { period: Some(14) };
        let first_input = TsfInput::from_candles(&candles, "close", first_params);
        let first_result = tsf_with_kernel(&first_input, kernel)?;
        let second_params = TsfParams { period: Some(14) };
        let second_input = TsfInput::from_slice(&first_result.values, second_params);
        let second_result = tsf_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_tsf_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TsfInput::from_candles(&candles, "close", TsfParams { period: Some(14) });
        let res = tsf_with_kernel(&input, kernel)?;
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

    fn check_tsf_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;
        let input = TsfInput::from_candles(
            &candles,
            "close",
            TsfParams {
                period: Some(period),
            },
        );
        let batch_output = tsf_with_kernel(&input, kernel)?.values;

        let mut stream = TsfStream::try_new(TsfParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(tsf_val) => stream_values.push(tsf_val),
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
                "[{}] TSF streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_tsf_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            TsfParams::default(), // period: 14
            TsfParams { period: Some(1) },
            TsfParams { period: Some(2) },
            TsfParams { period: Some(5) },
            TsfParams { period: Some(7) },
            TsfParams { period: Some(10) },
            TsfParams { period: Some(20) },
            TsfParams { period: Some(30) },
            TsfParams { period: Some(50) },
            TsfParams { period: Some(100) },
            TsfParams { period: Some(200) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = TsfInput::from_candles(&candles, "close", params.clone());
            let output = tsf_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_tsf_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_tsf_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate random parameter combinations and corresponding data vectors
        // Period must be at least 2 for meaningful regression (period=1 would have no regression)
        let strat = (2usize..=30).prop_flat_map(|period| {
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
                let params = TsfParams {
                    period: Some(period),
                };
                let input = TsfInput::from_slice(&data, params);

                // Get outputs from the kernel under test and scalar reference
                let TsfOutput { values: out } = tsf_with_kernel(&input, kernel).unwrap();
                let TsfOutput { values: ref_out } =
                    tsf_with_kernel(&input, Kernel::Scalar).unwrap();

                // Find first non-NaN index (warmup period)
                let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup_end = first + period - 1;

                // Property 1: Warmup period validation - first period-1 values should be NaN
                for i in 0..warmup_end {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN during warmup at index {}, got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }

                // Property 2: Kernel consistency - compare with scalar reference
                for i in warmup_end..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Handle NaN/infinite values
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "[{}] finite/NaN mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    // ULP comparison for finite values
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "[{}] Kernel mismatch at idx {}: {} vs {} (ULP={})",
                        test_name,
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                // Property 3: Constant data property - TSF should predict the same constant
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && !data.is_empty() {
                    for i in warmup_end..data.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                (out[i] - data[0]).abs() <= 1e-6,
                                "[{}] Constant data: TSF at {} = {}, expected {}",
                                test_name,
                                i,
                                out[i],
                                data[0]
                            );
                        }
                    }
                }

                // Property 4: Bounded output range
                // TSF predictions should be within reasonable bounds of the input data
                // Since TSF extrapolates, we allow for some extension beyond the data range
                let data_min = data[first..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let data_max = data[first..]
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let data_range = data_max - data_min;

                // Allow TSF to extrapolate up to 2x the data range beyond min/max
                let bound_factor = 2.0;
                let lower_bound = data_min - bound_factor * data_range;
                let upper_bound = data_max + bound_factor * data_range;

                for i in warmup_end..data.len() {
                    if out[i].is_finite() && data_range > 1e-10 {
                        prop_assert!(
                            out[i] >= lower_bound && out[i] <= upper_bound,
                            "[{}] TSF at {} = {} is outside bounds [{}, {}]",
                            test_name,
                            i,
                            out[i],
                            lower_bound,
                            upper_bound
                        );
                    }
                }

                // Property 5: Monotonic trend validation (improved)
                let is_monotonic_inc = data.windows(2).all(|w| w[1] >= w[0] - 1e-10);
                let is_monotonic_dec = data.windows(2).all(|w| w[1] <= w[0] + 1e-10);

                if is_monotonic_inc || is_monotonic_dec {
                    // For monotonic data, check trend direction at a few points
                    let test_points =
                        vec![warmup_end, (warmup_end + data.len()) / 2, data.len() - 1];

                    for &i in test_points.iter() {
                        if i < data.len() && out[i].is_finite() {
                            let window_end = i;
                            let window_start = i.saturating_sub(period - 1);

                            if is_monotonic_inc {
                                // For increasing data, TSF should predict >= last value
                                prop_assert!(
                                    out[i] >= data[window_end] - 1e-6,
                                    "[{}] Monotonic increasing: TSF at {} = {} < last value {}",
                                    test_name,
                                    i,
                                    out[i],
                                    data[window_end]
                                );
                            } else if is_monotonic_dec {
                                // For decreasing data, TSF should predict <= last value
                                prop_assert!(
                                    out[i] <= data[window_end] + 1e-6,
                                    "[{}] Monotonic decreasing: TSF at {} = {} > last value {}",
                                    test_name,
                                    i,
                                    out[i],
                                    data[window_end]
                                );
                            }
                        }
                    }
                }

                // Property 6: Edge case - period = 2 (minimum valid period)
                // Only test once at the first valid output position
                if period == 2 && warmup_end < data.len() {
                    let i = warmup_end;
                    if out[i].is_finite() {
                        // With period=2, TSF fits a line through the last 2 points and extrapolates
                        let x0 = data[i - 1]; // older point at x=0
                        let x1 = data[i]; // newer point at x=1
                                          // Linear extrapolation to x=2: 2*x1 - x0
                        let expected = 2.0 * x1 - x0;
                        prop_assert!(
                            (out[i] - expected).abs() <= 1e-6,
                            "[{}] Period=2: TSF at {} = {}, expected {}",
                            test_name,
                            i,
                            out[i],
                            expected
                        );
                    }
                }

                // Property 7: Clean input produces clean output (no NaN injection)
                // After warmup, all outputs should be finite for finite inputs
                for i in warmup_end..data.len() {
                    if data[i.saturating_sub(period - 1)..=i]
                        .iter()
                        .all(|x| x.is_finite())
                    {
                        prop_assert!(
                            out[i].is_finite(),
                            "[{}] TSF produced NaN/Inf at {} despite finite input window",
                            test_name,
                            i
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_tsf_tests {
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

    generate_all_tsf_tests!(
        check_tsf_partial_params,
        check_tsf_accuracy,
        check_tsf_default_candles,
        check_tsf_zero_period,
        check_tsf_period_one,
        check_tsf_mismatched_output_len,
        check_tsf_period_exceeds_length,
        check_tsf_very_small_dataset,
        check_tsf_reinput,
        check_tsf_nan_handling,
        check_tsf_streaming,
        check_tsf_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_tsf_tests!(check_tsf_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = TsfBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = TsfParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            58846.945054945056,
            58818.83516483516,
            58854.57142857143,
            59083.846153846156,
            58962.25274725275,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (period_start, period_end, period_step)
            (2, 10, 2),     // Small periods
            (5, 25, 5),     // Medium periods
            (20, 50, 10),   // Large periods
            (2, 5, 1),      // Dense small range
            (14, 14, 0),    // Single value (default period)
            (30, 60, 15),   // Mixed range
            (50, 100, 25),  // Very large periods
            (100, 200, 50), // Extra large periods
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = TsfBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
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
