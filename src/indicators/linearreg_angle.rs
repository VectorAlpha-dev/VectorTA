//! # Linear Regression Angle (LRA)
//!
//! Computes the angle (in degrees) of the linear regression line for a given period.
//! The angle represents the slope of the best-fit line converted to degrees, useful for
//! identifying trend strength and direction.
//!
//! ## Parameters
//! - **period**: Window size (number of data points), defaults to 14
//!
//! ## Inputs
//! - Single data slice (typically close prices)
//!
//! ## Returns
//! - **`Ok(Linearreg_angleOutput)`** containing values (Vec<f64>) representing regression angle in degrees
//! - Output length matches input data length with NaN padding for warmup period
//!
//! ## Developer Notes
//! - SIMD: Implemented but disabled by default; AVX2/AVX512 kernels are stubs delegating to the scalar path because the optimized scalar is faster at realistic sizes.
//! - Batch: Row-specific path shares S/K across rows when NaN-free; SIMD batch variants are stubs delegating to the scalar prefix path (no SIMD).
//! - Streaming: ✅ O(1) per update via ring buffer maintaining Σy and Σ(k*y); O(period) rebuild only on boundary NaNs (mirrors scalar semantics).
//! - Memory: ✅ Uses zero-copy warmup allocation helpers (`alloc_with_nan_prefix`, `init_matrix_prefixes`, `make_uninit_matrix`).
//! - Style: Matches alma.rs patterns for API shape, warmup handling, and feature-gated SIMD.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaLinearregAngle};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;
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
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum Linearreg_angleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for Linearreg_angleInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            Linearreg_angleData::Slice(slice) => slice,
            Linearreg_angleData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct Linearreg_angleParams {
    pub period: Option<usize>,
}

impl Default for Linearreg_angleParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleInput<'a> {
    pub data: Linearreg_angleData<'a>,
    pub params: Linearreg_angleParams,
}

impl<'a> Linearreg_angleInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: Linearreg_angleParams) -> Self {
        Self {
            data: Linearreg_angleData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: Linearreg_angleParams) -> Self {
        Self {
            data: Linearreg_angleData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", Linearreg_angleParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Linearreg_angleBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for Linearreg_angleBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl Linearreg_angleBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
        let p = Linearreg_angleParams {
            period: self.period,
        };
        let i = Linearreg_angleInput::from_candles(c, "close", p);
        linearreg_angle_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
        let p = Linearreg_angleParams {
            period: self.period,
        };
        let i = Linearreg_angleInput::from_slice(d, p);
        linearreg_angle_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<Linearreg_angleStream, Linearreg_angleError> {
        let p = Linearreg_angleParams {
            period: self.period,
        };
        Linearreg_angleStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum Linearreg_angleError {
    #[error("linearreg_angle: Empty data slice.")]
    EmptyInputData,
    #[error("linearreg_angle: All values are NaN.")]
    AllValuesNaN,
    #[error("linearreg_angle: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("linearreg_angle: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("linearreg_angle: Output length mismatch: expected = {expected}, actual = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("linearreg_angle: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("linearreg_angle: Invalid kernel type for batch operation: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[cfg(feature = "wasm")]
impl From<Linearreg_angleError> for JsValue {
    fn from(err: Linearreg_angleError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

#[inline]
pub fn linearreg_angle(
    input: &Linearreg_angleInput,
) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
    linearreg_angle_with_kernel(input, Kernel::Auto)
}

/// Write Linear Regression Angle outputs into a caller-provided buffer with no allocations.
///
/// - Preserves NaN warmups exactly like the Vec-returning API: the first
///   `first_valid + period - 1` elements are set to NaN.
/// - The output slice length must match the input length.
///
/// This is the native (non-WASM) entry point. For the WASM build, a separate
/// `linearreg_angle_into` with pointers is exposed via wasm-bindgen.
#[cfg(not(feature = "wasm"))]
pub fn linearreg_angle_into(
    input: &Linearreg_angleInput,
    out: &mut [f64],
) -> Result<(), Linearreg_angleError> {
    // Delegate to the existing into-slice kernel dispatcher using Auto.
    linearreg_angle_into_slice(out, input, Kernel::Auto)
}

pub fn linearreg_angle_with_kernel(
    input: &Linearreg_angleInput,
    kernel: Kernel,
) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(Linearreg_angleError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(Linearreg_angleError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period < 2 || period > len {
        return Err(Linearreg_angleError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(Linearreg_angleError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // For this indicator, scalar outperforms current SIMD variants in typical sizes.
    // Keep SIMD implementations available for explicit selection, but default Auto → Scalar.
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let mut out = alloc_with_nan_prefix(len, first + period - 1);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                linearreg_angle_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => linearreg_angle_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_angle_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(Linearreg_angleOutput { values: out })
}

#[inline]
pub fn linearreg_angle_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    // Optimized sliding window with O(1) per step, NaN-safe rebuild on boundary NaNs
    let p = period as f64;
    let sum_x = (period * (period - 1)) as f64 * 0.5; // Σx
    let sum_x_sqr = (period * (period - 1) * (2 * period - 1)) as f64 / 6.0; // Σx^2
    let divisor = sum_x * sum_x - p * sum_x_sqr; // reversed-x denominator
    let inv_div = 1.0 / divisor;
    let rad2deg = 180.0 / PI;

    let n = data.len();
    let mut i = first_valid + period - 1;
    if i >= n {
        return;
    }

    // Initial window [start..=i]
    let mut start = i + 1 - period;
    let mut sum_y = 0.0;
    let mut sum_kd = 0.0;

    // Fast path if there are no NaNs after first_valid: no per-step NaN checks
    let has_nan = data[first_valid..].iter().any(|v| v.is_nan());

    unsafe {
        // Build initial sums (unrolled)
        let mut j = start;
        let end = i + 1;
        while j + 3 < end {
            let y0 = *data.get_unchecked(j);
            let y1 = *data.get_unchecked(j + 1);
            let y2 = *data.get_unchecked(j + 2);
            let y3 = *data.get_unchecked(j + 3);

            sum_y += y0 + y1 + y2 + y3;
            let jf = j as f64;
            sum_kd += jf * y0 + (jf + 1.0) * y1 + (jf + 2.0) * y2 + (jf + 3.0) * y3;

            j += 4;
        }
        while j < end {
            let y = *data.get_unchecked(j);
            sum_y += y;
            sum_kd += (j as f64) * y;
            j += 1;
        }

        if !has_nan {
            // No-NaN fast path: O(1) slide with no NaN checks
            loop {
                let i_f = i as f64;
                let sum_xy = i_f * sum_y - sum_kd;
                let num = p.mul_add(sum_xy, -sum_x * sum_y);
                let slope = num * inv_div;
                *out.get_unchecked_mut(i) = slope.atan() * rad2deg;

                i += 1;
                if i >= n {
                    break;
                }

                let enter = *data.get_unchecked(i);
                let leave = *data.get_unchecked(start);
                start += 1;

                sum_y += enter - leave;
                sum_kd += (i as f64) * enter - ((i - period) as f64) * leave;
            }
        } else {
            // NaN-safe path: rebuild on boundary-NaN
            loop {
                let i_f = i as f64;
                let sum_xy = i_f * sum_y - sum_kd;
                let num = p.mul_add(sum_xy, -sum_x * sum_y);
                let slope = num * inv_div;
                *out.get_unchecked_mut(i) = slope.atan() * rad2deg;

                i += 1;
                if i >= n {
                    break;
                }

                let enter = *data.get_unchecked(i);
                let leave = *data.get_unchecked(start);
                start += 1;

                if enter.is_nan() | leave.is_nan() {
                    sum_y = 0.0;
                    sum_kd = 0.0;
                    let ws = i + 1 - period;
                    let mut jj = ws;
                    let ee = i + 1;
                    while jj + 3 < ee {
                        let y0 = *data.get_unchecked(jj);
                        let y1 = *data.get_unchecked(jj + 1);
                        let y2 = *data.get_unchecked(jj + 2);
                        let y3 = *data.get_unchecked(jj + 3);

                        sum_y += y0 + y1 + y2 + y3;
                        let jf = jj as f64;
                        sum_kd += jf * y0 + (jf + 1.0) * y1 + (jf + 2.0) * y2 + (jf + 3.0) * y3;
                        jj += 4;
                    }
                    while jj < ee {
                        let y = *data.get_unchecked(jj);
                        sum_y += y;
                        sum_kd += (jj as f64) * y;
                        jj += 1;
                    }
                } else {
                    sum_y += enter - leave;
                    sum_kd += (i as f64) * enter - ((i - period) as f64) * leave;
                }
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_angle_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { linearreg_angle_avx512_short(data, period, first_valid, out) }
    } else {
        unsafe { linearreg_angle_avx512_long(data, period, first_valid, out) }
    }
}

#[inline]
pub fn linearreg_angle_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    // SIMD stub: delegate to scalar for accuracy/performance reasons
    linearreg_angle_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_angle_avx512_short(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    // SIMD stub: delegate to scalar
    linearreg_angle_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_angle_avx512_long(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    // SIMD stub: delegate to scalar
    linearreg_angle_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn linearreg_angle_avx512_impl(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    // Fallback to scalar if NaNs present (prefix sums would poison later windows).
    if data[first_valid..].iter().any(|v| v.is_nan()) {
        return linearreg_angle_scalar(data, period, first_valid, out);
    }

    let n = data.len();
    let start_i = first_valid + period - 1;
    if start_i >= n {
        return;
    }

    // Constants
    let p = period as f64;
    let sum_x = (period * (period - 1)) as f64 * 0.5;
    let sum_x_sqr = (period * (period - 1) * (2 * period - 1)) as f64 / 6.0;
    let divisor = sum_x * sum_x - p * sum_x_sqr;
    let inv_div = 1.0 / divisor;
    let rad2deg = 180.0 / PI;

    // Prefix sums (S and K)
    let mut s = vec![0.0f64; n + 1];
    let mut k = vec![0.0f64; n + 1];
    let mut acc_s = 0.0f64;
    let mut acc_k = 0.0f64;

    let mut idx = 0usize;
    while idx + 3 < n {
        let y0 = *data.get_unchecked(idx);
        let y1 = *data.get_unchecked(idx + 1);
        let y2 = *data.get_unchecked(idx + 2);
        let y3 = *data.get_unchecked(idx + 3);

        acc_s += y0;
        s[idx + 1] = acc_s;
        acc_k += (idx as f64) * y0;
        k[idx + 1] = acc_k;
        acc_s += y1;
        s[idx + 2] = acc_s;
        acc_k += ((idx + 1) as f64) * y1;
        k[idx + 2] = acc_k;
        acc_s += y2;
        s[idx + 3] = acc_s;
        acc_k += ((idx + 2) as f64) * y2;
        k[idx + 3] = acc_k;
        acc_s += y3;
        s[idx + 4] = acc_s;
        acc_k += ((idx + 3) as f64) * y3;
        k[idx + 4] = acc_k;

        idx += 4;
    }
    while idx < n {
        let y = *data.get_unchecked(idx);
        acc_s += y;
        s[idx + 1] = acc_s;
        acc_k += (idx as f64) * y;
        k[idx + 1] = acc_k;
        idx += 1;
    }

    // SIMD constants
    let v_p = _mm512_set1_pd(p);
    let v_nsumx = _mm512_set1_pd(-sum_x);
    let v_invdiv = _mm512_set1_pd(inv_div);

    let mut i = start_i;
    let width = 8usize;

    // Bulk by 8 lanes
    while i + width <= n {
        let s_hi = _mm512_loadu_pd(s.as_ptr().add(i + 1));
        let s_lo = _mm512_loadu_pd(s.as_ptr().add(i + 1 - period));
        let sum_y = _mm512_sub_pd(s_hi, s_lo);

        let k_hi = _mm512_loadu_pd(k.as_ptr().add(i + 1));
        let k_lo = _mm512_loadu_pd(k.as_ptr().add(i + 1 - period));
        let sum_kd = _mm512_sub_pd(k_hi, k_lo);

        let base = i as f64;
        let v_i = _mm512_setr_pd(
            base,
            base + 1.0,
            base + 2.0,
            base + 3.0,
            base + 4.0,
            base + 5.0,
            base + 6.0,
            base + 7.0,
        );

        let sum_xy = _mm512_fnmadd_pd(v_i, sum_y, sum_kd);
        let sum_xy = _mm512_sub_pd(_mm512_setzero_pd(), sum_xy);

        let num = _mm512_fmadd_pd(v_p, sum_xy, _mm512_mul_pd(v_nsumx, sum_y));
        let slope = _mm512_mul_pd(num, v_invdiv);

        let mut tmp: [f64; 8] = core::mem::zeroed();
        _mm512_storeu_pd(tmp.as_mut_ptr(), slope);

        *out.get_unchecked_mut(i) = tmp[0].atan() * rad2deg;
        *out.get_unchecked_mut(i + 1) = tmp[1].atan() * rad2deg;
        *out.get_unchecked_mut(i + 2) = tmp[2].atan() * rad2deg;
        *out.get_unchecked_mut(i + 3) = tmp[3].atan() * rad2deg;
        *out.get_unchecked_mut(i + 4) = tmp[4].atan() * rad2deg;
        *out.get_unchecked_mut(i + 5) = tmp[5].atan() * rad2deg;
        *out.get_unchecked_mut(i + 6) = tmp[6].atan() * rad2deg;
        *out.get_unchecked_mut(i + 7) = tmp[7].atan() * rad2deg;

        i += width;
    }

    while i < n {
        let sum_y = *s.get_unchecked(i + 1) - *s.get_unchecked(i + 1 - period);
        let sum_kd = *k.get_unchecked(i + 1) - *k.get_unchecked(i + 1 - period);
        let sum_xy = (i as f64) * sum_y - sum_kd;
        let num = p.mul_add(sum_xy, -sum_x * sum_y);
        let slope = num * (1.0 / divisor);
        *out.get_unchecked_mut(i) = slope.atan() * rad2deg;
        i += 1;
    }
}

/// Decision: Streaming uses an O(1) ring-buffer kernel keeping Σy and Σ(k*y);
/// rebuilds O(period) only on boundary NaNs to mirror scalar semantics.
#[derive(Debug, Clone)]
pub struct Linearreg_angleStream {
    period: usize,

    // Ring buffer of the last `period` samples (values only)
    ring: Vec<f64>,
    head: usize, // next write position (also oldest when len == period)
    len: usize,  // number of items currently in the ring (<= period)

    // Running accumulators for the current window [i-(len-1) .. i]
    // sum_y  = Σ y[k]
    // sum_kd = Σ (k_abs * y[k]) with absolute index k_abs
    sum_y: f64,
    sum_kd: f64,

    // Absolute index of the next incoming sample
    idx: usize,

    // Precomputed constants for fixed x = 0..(period-1)
    p: f64,
    sum_x: f64,   // Σ x = period*(period-1)/2
    inv_div: f64, // 1 / (sum_x^2 - p * Σx^2)
    rad2deg: f64, // 180 / π

    // Preserve params for API consistency (builders/bindings)
    params: Linearreg_angleParams,
}

impl Linearreg_angleStream {
    pub fn try_new(params: Linearreg_angleParams) -> Result<Self, Linearreg_angleError> {
        let period = params.period.unwrap_or(14);
        if period < 2 {
            return Err(Linearreg_angleError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        // Precompute constants for fixed x = 0..N-1
        let p = period as f64;
        let sum_x = (period * (period - 1)) as f64 * 0.5;
        let sum_x_sqr = (period * (period - 1) * (2 * period - 1)) as f64 / 6.0;
        let divisor = sum_x * sum_x - p * sum_x_sqr;
        let inv_div = 1.0 / divisor;

        Ok(Self {
            period,
            ring: vec![f64::NAN; period],
            head: 0,
            len: 0,
            sum_y: 0.0,
            sum_kd: 0.0,
            idx: 0,
            p,
            sum_x,
            inv_div,
            rad2deg: 180.0 / std::f64::consts::PI,
            params,
        })
    }

    /// O(1) update on normal steps; O(period) rebuild only if the entering
    /// or leaving sample is NaN (mirrors scalar semantics). Returns None
    /// until at least `period` samples have been seen.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let i = self.idx; // absolute index of this sample
        let had_full = self.len == self.period;
        let leave = if had_full { self.ring[self.head] } else { 0.0 };

        // Write into ring and advance head
        self.ring[self.head] = value;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }

        if self.len < self.period {
            // Warmup accumulation for Σy and Σ(k*y)
            self.len += 1;
            self.sum_y += value;
            self.sum_kd += (i as f64) * value;
        } else if value.is_nan() | leave.is_nan() {
            // Boundary NaN → rebuild window sums from ring (O(period))
            self.rebuild_window_sums(i);
        } else {
            // Steady slide in O(1)
            self.sum_y += value - leave;
            self.sum_kd += (i as f64) * value - ((i - self.period) as f64) * leave;
        }

        let out = if self.len < self.period {
            None
        } else {
            // reversed-x identity: sum_xy = i*sum_y - sum_kd
            let sum_xy = (i as f64) * self.sum_y - self.sum_kd;
            let num = self.p.mul_add(sum_xy, -self.sum_x * self.sum_y);
            let slope = num * self.inv_div;
            Some(slope.atan() * self.rad2deg)
        };

        self.idx = i + 1;
        out
    }

    /// Rebuild Σy and Σ(k*y) for the current window ending at absolute index `i`.
    /// The oldest element sits at `head` after the write step.
    #[inline(always)]
    fn rebuild_window_sums(&mut self, i: usize) {
        let win_len = self.len;
        let mut s_y = 0.0f64;
        let mut s_kd = 0.0f64;

        // Window absolute indices: [i - (win_len - 1) .. i]
        let start_abs = i + 1 - win_len;

        for j in 0..win_len {
            let pos = self.head + j;
            let rix = if pos >= self.period {
                pos - self.period
            } else {
                pos
            };
            let y = self.ring[rix];
            let k = (start_abs + j) as f64;
            s_y += y;
            s_kd += k * y;
        }
        self.sum_y = s_y;
        self.sum_kd = s_kd;
    }
}

#[derive(Clone, Debug)]
pub struct Linearreg_angleBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for Linearreg_angleBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 60, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Linearreg_angleBatchBuilder {
    range: Linearreg_angleBatchRange,
    kernel: Kernel,
}

impl Linearreg_angleBatchBuilder {
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
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
        linearreg_angle_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
        Linearreg_angleBatchBuilder::new()
            .kernel(k)
            .apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
        Linearreg_angleBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn linearreg_angle_batch_with_kernel(
    data: &[f64],
    sweep: &Linearreg_angleBatchRange,
    k: Kernel,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(Linearreg_angleError::InvalidKernelForBatch(k))
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    linearreg_angle_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct Linearreg_angleBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<Linearreg_angleParams>,
    pub rows: usize,
    pub cols: usize,
}

impl Linearreg_angleBatchOutput {
    pub fn row_for_params(&self, p: &Linearreg_angleParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &Linearreg_angleParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &Linearreg_angleBatchRange) -> Vec<Linearreg_angleParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let mut vals = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end {
                vals.push(x);
                let next = x.saturating_add(step);
                if next == x {
                    break;
                }
                x = next;
            }
        } else {
            let mut x = start;
            loop {
                vals.push(x);
                if x <= end {
                    break;
                }
                let next = x.saturating_sub(step);
                if next >= x {
                    break;
                }
                x = next;
            }
        }
        vals
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(Linearreg_angleParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn linearreg_angle_batch_slice(
    data: &[f64],
    sweep: &Linearreg_angleBatchRange,
    kern: Kernel,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
    linearreg_angle_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn linearreg_angle_batch_par_slice(
    data: &[f64],
    sweep: &Linearreg_angleBatchRange,
    kern: Kernel,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
    linearreg_angle_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn linearreg_angle_batch_inner(
    data: &[f64],
    sweep: &Linearreg_angleBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<Linearreg_angleBatchOutput, Linearreg_angleError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(Linearreg_angleError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    // Validate all periods are >= 2 (linear regression needs at least 2 points)
    for combo in &combos {
        let period = combo.period.unwrap();
        if period < 2 {
            return Err(Linearreg_angleError::InvalidPeriod {
                period,
                data_len: data.len(),
            });
        }
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(Linearreg_angleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let _ = combos
        .len()
        .checked_mul(max_p)
        .ok_or(Linearreg_angleError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    if data.len() - first < max_p {
        return Err(Linearreg_angleError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let _ = rows
        .checked_mul(cols)
        .ok_or(Linearreg_angleError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;

    // Use uninitialized memory helpers to avoid allocation
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // If input contains NaNs post-first, avoid prefix-sum path to preserve semantics
    let has_nan = data[first..].iter().any(|v| v.is_nan());

    // Optional shared prefixes for all rows (S: Σy, K: Σ(k*y)) when NaN-free
    let (s_pref, k_pref): (Option<Vec<f64>>, Option<Vec<f64>>) = if !has_nan {
        // Precompute S and K once for reuse across rows
        let n = data.len();
        let mut s = vec![0.0f64; n + 1];
        let mut k = vec![0.0f64; n + 1];
        let mut acc_s = 0.0f64;
        let mut acc_k = 0.0f64;
        let mut idx = 0usize;
        unsafe {
            while idx + 3 < n {
                let y0 = *data.get_unchecked(idx);
                let y1 = *data.get_unchecked(idx + 1);
                let y2 = *data.get_unchecked(idx + 2);
                let y3 = *data.get_unchecked(idx + 3);

                acc_s += y0;
                s[idx + 1] = acc_s;
                acc_k += (idx as f64) * y0;
                k[idx + 1] = acc_k;
                acc_s += y1;
                s[idx + 2] = acc_s;
                acc_k += ((idx + 1) as f64) * y1;
                k[idx + 2] = acc_k;
                acc_s += y2;
                s[idx + 3] = acc_s;
                acc_k += ((idx + 2) as f64) * y2;
                k[idx + 3] = acc_k;
                acc_s += y3;
                s[idx + 4] = acc_s;
                acc_k += ((idx + 3) as f64) * y3;
                k[idx + 4] = acc_k;

                idx += 4;
            }
            while idx < n {
                let y = *data.get_unchecked(idx);
                acc_s += y;
                s[idx + 1] = acc_s;
                acc_k += (idx as f64) * y;
                k[idx + 1] = acc_k;
                idx += 1;
            }
        }
        (Some(s), Some(k))
    } else {
        (None, None)
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        if has_nan {
            // Always use scalar sliding path when NaNs present
            linearreg_angle_row_scalar(data, first, period, out_row)
        } else {
            // Hoist per-row constants once
            let p = period as f64;
            let sum_x = (period * (period - 1)) as f64 * 0.5;
            let sum_x_sqr = (period * (period - 1) * (2 * period - 1)) as f64 / 6.0;
            let inv_div = 1.0 / (sum_x * sum_x - p * sum_x_sqr);
            let rad2deg = 180.0 / PI;

            match kern {
                Kernel::Scalar => linearreg_angle_row_scalar_with_prefixes(
                    data,
                    first,
                    out_row,
                    s_pref.as_ref().unwrap(),
                    k_pref.as_ref().unwrap(),
                    p,
                    sum_x,
                    inv_div,
                    rad2deg,
                    period,
                ),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => linearreg_angle_row_avx2_with_prefixes(
                    data,
                    first,
                    out_row,
                    s_pref.as_ref().unwrap(),
                    k_pref.as_ref().unwrap(),
                    p,
                    sum_x,
                    inv_div,
                    rad2deg,
                    period,
                ),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => linearreg_angle_row_avx512_with_prefixes(
                    data,
                    first,
                    out_row,
                    s_pref.as_ref().unwrap(),
                    k_pref.as_ref().unwrap(),
                    p,
                    sum_x,
                    inv_div,
                    rad2deg,
                    period,
                ),
                _ => unreachable!(),
            }
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

    // Convert uninitialized memory to Vec
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };
    core::mem::forget(buf_guard);

    Ok(Linearreg_angleBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn linearreg_angle_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    linearreg_angle_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    // SIMD stub: delegate to scalar
    linearreg_angle_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    // SIMD stub: delegate to scalar
    linearreg_angle_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    // SIMD stub: delegate to scalar
    linearreg_angle_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    // SIMD stub: delegate to scalar
    linearreg_angle_row_scalar(data, first, period, out)
}

// Row-specific helpers using shared prefix sums (S, K)
#[inline(always)]
unsafe fn linearreg_angle_row_scalar_with_prefixes(
    data: &[f64],
    first: usize,
    out: &mut [f64],
    s: &[f64],
    k: &[f64],
    p: f64,
    sum_x: f64,
    inv_div: f64,
    rad2deg: f64,
    period: usize,
) {
    let n = data.len();
    let start_i = first + period - 1;
    if start_i >= n {
        return;
    }

    let mut i = start_i;
    while i < n {
        let sum_y = *s.get_unchecked(i + 1) - *s.get_unchecked(i + 1 - period);
        let sum_kd = *k.get_unchecked(i + 1) - *k.get_unchecked(i + 1 - period);
        let sum_xy = (i as f64) * sum_y - sum_kd;
        let num = p.mul_add(sum_xy, -sum_x * sum_y);
        let slope = num * inv_div;
        *out.get_unchecked_mut(i) = slope.atan() * rad2deg;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx2_with_prefixes(
    data: &[f64],
    first: usize,
    out: &mut [f64],
    s: &[f64],
    k: &[f64],
    p: f64,
    sum_x: f64,
    inv_div: f64,
    rad2deg: f64,
    period: usize,
) {
    // SIMD stub with prefixes: call scalar prefix path
    linearreg_angle_row_scalar_with_prefixes(
        data, first, out, s, k, p, sum_x, inv_div, rad2deg, period,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_angle_row_avx512_with_prefixes(
    data: &[f64],
    first: usize,
    out: &mut [f64],
    s: &[f64],
    k: &[f64],
    p: f64,
    sum_x: f64,
    inv_div: f64,
    rad2deg: f64,
    period: usize,
) {
    // SIMD stub with prefixes: call scalar prefix path
    linearreg_angle_row_scalar_with_prefixes(
        data, first, out, s, k, p, sum_x, inv_div, rad2deg, period,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_lra_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = Linearreg_angleParams { period: None };
        let input = Linearreg_angleInput::from_candles(&candles, "close", default_params);
        let output = linearreg_angle_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_lra_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = Linearreg_angleParams { period: Some(14) };
        let input = Linearreg_angleInput::from_candles(&candles, "close", params);
        let result = linearreg_angle_with_kernel(&input, kernel)?;

        let expected_last_five = [
            -89.30491945492733,
            -89.28911257342405,
            -89.1088041965075,
            -86.58419429159467,
            -87.77085937059316,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-5,
                "[{}] LRA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_lra_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = Linearreg_angleParams { period: Some(0) };
        let input = Linearreg_angleInput::from_slice(&input_data, params);
        let res = linearreg_angle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LRA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_lra_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = Linearreg_angleParams { period: Some(10) };
        let input = Linearreg_angleInput::from_slice(&data_small, params);
        let res = linearreg_angle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LRA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_lra_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = Linearreg_angleParams { period: Some(14) };
        let input = Linearreg_angleInput::from_slice(&single_point, params);
        let res = linearreg_angle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LRA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_lra_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = Linearreg_angleParams { period: Some(14) };
        let first_input = Linearreg_angleInput::from_candles(&candles, "close", first_params);
        let first_result = linearreg_angle_with_kernel(&first_input, kernel)?;

        let second_params = Linearreg_angleParams { period: Some(14) };
        let second_input = Linearreg_angleInput::from_slice(&first_result.values, second_params);
        let second_result = linearreg_angle_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_lra_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations for linearreg_angle
        let test_params = vec![
            Linearreg_angleParams::default(),            // period: 14 (default)
            Linearreg_angleParams { period: Some(2) },   // minimum viable period
            Linearreg_angleParams { period: Some(3) },   // very small period
            Linearreg_angleParams { period: Some(5) },   // small period
            Linearreg_angleParams { period: Some(7) },   // small period
            Linearreg_angleParams { period: Some(10) },  // small-medium period
            Linearreg_angleParams { period: Some(14) },  // default period
            Linearreg_angleParams { period: Some(20) },  // medium period
            Linearreg_angleParams { period: Some(30) },  // medium period
            Linearreg_angleParams { period: Some(50) },  // medium-large period
            Linearreg_angleParams { period: Some(100) }, // large period
            Linearreg_angleParams { period: Some(200) }, // very large period
            Linearreg_angleParams { period: Some(500) }, // extremely large period
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = Linearreg_angleInput::from_candles(&candles, "close", params.clone());
            let output = linearreg_angle_with_kernel(&input, kernel)?;

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
    fn check_lra_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_lra_tests {
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

    #[test]
    fn test_linearreg_angle_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Build a small but non-trivial input slice with a few leading NaNs
        let len = 256usize;
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            let x = i as f64;
            // Smoothly varying series with trend + curvature
            let v = (0.05 * x).sin() * 3.0 + 0.01 * x + (0.001 * x * x);
            data.push(v);
        }
        // Introduce a couple of NaNs at the start to exercise warmup handling
        data[0] = f64::NAN;
        data[1] = f64::NAN;

        let params = Linearreg_angleParams::default();
        let input = Linearreg_angleInput::from_slice(&data, params);

        // Baseline via Vec-returning API
        let baseline = linearreg_angle(&input)?.values;

        // Preallocate output buffer and invoke the new into API
        let mut into_out = vec![0.0f64; len];
        #[cfg(not(feature = "wasm"))]
        {
            linearreg_angle_into(&input, &mut into_out)?;
        }
        #[cfg(feature = "wasm")]
        {
            // In WASM builds the native into is not available; use into_slice directly
            linearreg_angle_into_slice(&mut into_out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), into_out.len());

        // Helper: NaN == NaN, else exact equality
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(baseline[i], into_out[i]),
                "Mismatch at index {}: api={} into={}",
                i,
                baseline[i],
                into_out[i]
            );
        }

        Ok(())
    }
    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_linearreg_angle_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic data and parameters
        let strat = (2usize..=50) // Period from 2-50 (1 is invalid for linear regression)
            .prop_flat_map(|period| {
                (
                    // Generate base price and volatility
                    (100f64..10000f64, 0.0001f64..0.1f64, period + 10..400)
                        .prop_flat_map(move |(base_price, volatility, data_len)| {
                            // Generate price movements with controlled volatility
                            let price_changes = prop::collection::vec(
                                (-volatility..volatility).prop_filter("finite", |x| x.is_finite()),
                                data_len,
                            );

                            // Generate scenarios with different data patterns
                            let scenario = prop::strategy::Union::new(vec![
                                // 60% normal price movements
                                (0u8..60u8).boxed(),
                                // 10% constant prices
                                (60u8..70u8).boxed(),
                                // 10% linear uptrend
                                (70u8..80u8).boxed(),
                                // 10% linear downtrend
                                (80u8..90u8).boxed(),
                                // 5% near-horizontal (very small slope)
                                (90u8..95u8).boxed(),
                                // 5% steep trend (large slope)
                                (95u8..100u8).boxed(),
                            ]);

                            (
                                Just(base_price),
                                Just(volatility),
                                Just(data_len),
                                price_changes,
                                scenario,
                            )
                        })
                        .prop_map(
                            move |(base_price, volatility, data_len, price_changes, scenario)| {
                                let mut data = Vec::with_capacity(data_len);

                                match scenario {
                                    0..=59 => {
                                        // Normal price movements
                                        let mut current_price = base_price;
                                        for change in price_changes {
                                            current_price *= 1.0 + change;
                                            data.push(current_price);
                                        }
                                    }
                                    60..=69 => {
                                        // Constant prices
                                        data.resize(data_len, base_price);
                                    }
                                    70..=79 => {
                                        // Linear uptrend
                                        let slope = base_price * 0.001; // 0.1% per period
                                        for i in 0..data_len {
                                            data.push(base_price + slope * i as f64);
                                        }
                                    }
                                    80..=89 => {
                                        // Linear downtrend
                                        let slope = base_price * 0.001;
                                        for i in 0..data_len {
                                            data.push(base_price - slope * i as f64);
                                        }
                                    }
                                    90..=94 => {
                                        // Near-horizontal (very small slope)
                                        let slope = base_price * 0.00001; // 0.001% per period
                                        for i in 0..data_len {
                                            data.push(base_price + slope * i as f64);
                                        }
                                    }
                                    95..=99 => {
                                        // Steep trend (large slope)
                                        let slope = base_price * 0.1; // 10% per period
                                        for i in 0..data_len {
                                            data.push(base_price + slope * i as f64);
                                        }
                                    }
                                    _ => unreachable!(),
                                }
                                data
                            },
                        ),
                    Just(period),
                )
            });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period)| {
            let params = Linearreg_angleParams {
                period: Some(period),
            };
            let input = Linearreg_angleInput::from_slice(&data, params);

            // Get output from the kernel being tested
            let result = linearreg_angle_with_kernel(&input, kernel)?;
            let out = &result.values;

            // Get reference output from scalar kernel
            let ref_result = linearreg_angle_with_kernel(&input, Kernel::Scalar)?;
            let ref_out = &ref_result.values;

            // Property 1: Output length matches input
            prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

            // Property 2: Warmup period values are NaN
            // Since we generate clean data without NaN, first = 0
            let warmup_end = period - 1;
            for i in 0..warmup_end {
                prop_assert!(
                    out[i].is_nan(),
                    "Expected NaN during warmup at index {}, got {}",
                    i,
                    out[i]
                );
            }

            // Property 3: Valid values are within [-90, 90] degrees
            for (i, &val) in out.iter().enumerate().skip(warmup_end) {
                if !val.is_nan() {
                    prop_assert!(
                        val >= -90.0 && val <= 90.0,
                        "Angle out of bounds at index {}: {} degrees",
                        i,
                        val
                    );
                }
            }

            // Property 4: Constant data produces ~0 degree angle (relaxed tolerance)
            if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                // All values are essentially the same
                for (i, &val) in out.iter().enumerate().skip(warmup_end) {
                    if !val.is_nan() {
                        prop_assert!(
                            val.abs() < 1e-3, // Relaxed tolerance for floating-point
                            "Expected ~0° for constant data at index {}, got {}°",
                            i,
                            val
                        );
                    }
                }
            }

            // Property 5: Enhanced linear trend validation
            // Check if data is perfectly linear
            let is_perfectly_linear = if data.len() >= 3 {
                let mut deltas = Vec::new();
                for i in 1..data.len() {
                    deltas.push(data[i] - data[i - 1]);
                }
                // Check if all deltas are approximately equal
                deltas.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
            } else {
                false
            };

            if is_perfectly_linear && data.len() > period {
                // For perfectly linear data, all angles should be identical
                let valid_angles: Vec<f64> = out
                    .iter()
                    .skip(warmup_end)
                    .filter(|&&v| !v.is_nan())
                    .copied()
                    .collect();

                if valid_angles.len() >= 2 {
                    let first_angle = valid_angles[0];
                    for (i, &angle) in valid_angles.iter().enumerate().skip(1) {
                        prop_assert!(
								(angle - first_angle).abs() < 1e-6,
								"Linear data should produce consistent angles. Index {} has {} vs first {}",
								i, angle, first_angle
							);
                    }
                }
            }

            // Check trend direction for linear data
            let is_linear_up = data.windows(2).all(|w| w[1] > w[0]);
            let is_linear_down = data.windows(2).all(|w| w[1] < w[0]);

            if is_linear_up {
                for (i, &val) in out.iter().enumerate().skip(warmup_end) {
                    if !val.is_nan() {
                        prop_assert!(
                            val > 0.0,
                            "Expected positive angle for uptrend at index {}, got {}°",
                            i,
                            val
                        );
                    }
                }
            }

            if is_linear_down {
                for (i, &val) in out.iter().enumerate().skip(warmup_end) {
                    if !val.is_nan() {
                        prop_assert!(
                            val < 0.0,
                            "Expected negative angle for downtrend at index {}, got {}°",
                            i,
                            val
                        );
                    }
                }
            }

            // Property 6: Mathematical correctness for simple cases
            // For a simple linear sequence, verify the angle is correct
            if period <= 10 && data.len() >= period * 2 {
                // Create a simple test case: y = x (45-degree line in data space)
                let test_data: Vec<f64> = (0..period).map(|i| i as f64).collect();
                let test_params = Linearreg_angleParams {
                    period: Some(period),
                };
                let test_input = Linearreg_angleInput::from_slice(&test_data, test_params);

                if let Ok(test_result) = linearreg_angle_with_kernel(&test_input, kernel) {
                    if test_result.values.len() >= period {
                        let test_angle = test_result.values[period - 1];
                        if !test_angle.is_nan() {
                            // For y = x with unit spacing, slope = 1, angle = atan(1) = 45°
                            let expected_angle = 45.0;
                            prop_assert!(
                                (test_angle - expected_angle).abs() < 1.0,
                                "Mathematical test failed: expected ~45°, got {}°",
                                test_angle
                            );
                        }
                    }
                }
            }

            // Property 7: Edge cases - extreme angles
            // Near-horizontal should produce small angles
            let base_price = data[0];
            if data.windows(2).all(|w| {
                let delta = (w[1] - w[0]).abs();
                delta < base_price * 0.00001 && delta > 0.0
            }) {
                for &val in out.iter().skip(warmup_end) {
                    if !val.is_nan() {
                        prop_assert!(
                            val.abs() < 1.0, // Should be very small angle
                            "Near-horizontal data should produce small angle, got {}°",
                            val
                        );
                    }
                }
            }

            // Property 8: Kernel consistency
            for i in warmup_end..data.len() {
                let y = out[i];
                let r = ref_out[i];

                // Handle NaN/infinity cases
                if !y.is_finite() || !r.is_finite() {
                    prop_assert_eq!(
                        y.to_bits(),
                        r.to_bits(),
                        "NaN/infinity mismatch at index {}: {} vs {}",
                        i,
                        y,
                        r
                    );
                    continue;
                }

                // Check ULP difference for finite values
                let y_bits = y.to_bits();
                let r_bits = r.to_bits();
                let ulp_diff = y_bits.abs_diff(r_bits);

                prop_assert!(
                    (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                    "Kernel mismatch at index {}: {} vs {} (ULP diff: {})",
                    i,
                    y,
                    r,
                    ulp_diff
                );
            }

            Ok(())
        })?;

        Ok(())
    }
    #[cfg(feature = "proptest")]
    generate_all_lra_tests!(check_linearreg_angle_property);

    generate_all_lra_tests!(
        check_lra_partial_params,
        check_lra_accuracy,
        check_lra_zero_period,
        check_lra_period_exceeds_length,
        check_lra_very_small_dataset,
        check_lra_reinput,
        check_lra_no_poison
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = Linearreg_angleBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = Linearreg_angleParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -89.30491945492733,
            -89.28911257342405,
            -89.1088041965075,
            -86.58419429159467,
            -87.77085937059316,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    fn check_batch_grid_search(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let batch = Linearreg_angleBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 16, 2)
            .apply_candles(&c, "close")?;

        // Should have periods: 10, 12, 14, 16
        let periods = [10, 12, 14, 16];
        assert_eq!(batch.rows, 4);

        for (ix, p) in periods.iter().enumerate() {
            let param = Linearreg_angleParams { period: Some(*p) };
            let row_idx = batch.row_for_params(&param);
            assert_eq!(row_idx, Some(ix), "Batch grid missing period {p}");
            let row = batch.values_for(&param).expect("Missing row for period");
            assert_eq!(row.len(), batch.cols, "Row len mismatch for period {p}");
        }
        Ok(())
    }

    fn check_batch_period_static(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let batch = Linearreg_angleBatchBuilder::new()
            .kernel(kernel)
            .period_static(14)
            .apply_candles(&c, "close")?;

        assert_eq!(batch.rows, 1);
        let param = Linearreg_angleParams { period: Some(14) };
        let row = batch.values_for(&param).expect("Missing static row");
        assert_eq!(row.len(), batch.cols);

        // Check a value
        let last = *row.last().unwrap();
        let expected = -87.77085937059316;
        assert!(
            (last - expected).abs() < 1e-5,
            "Static period row last val mismatch: got {last}, want {expected}"
        );

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations for linearreg_angle
        let test_configs = vec![
            // (period_start, period_end, period_step)
            (2, 10, 2),      // Small periods, step 2
            (5, 15, 1),      // Dense small range
            (10, 50, 10),    // Medium periods, step 10
            (20, 100, 20),   // Large periods, step 20
            (50, 200, 50),   // Very large periods
            (14, 14, 0),     // Single period (default)
            (2, 5, 1),       // Very small dense range
            (100, 500, 100), // Extremely large periods
            (7, 21, 7),      // Weekly-based periods
            (30, 90, 30),    // Monthly-based periods
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let mut builder = Linearreg_angleBatchBuilder::new().kernel(kernel);

            // Configure period range
            if p_step > 0 {
                builder = builder.period_range(p_start, p_end, p_step);
            } else {
                builder = builder.period_static(p_start);
            }

            let output = builder.apply_candles(&c, "close")?;

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
    gen_batch_tests!(check_batch_grid_search);
    gen_batch_tests!(check_batch_period_static);
    gen_batch_tests!(check_batch_no_poison);
}

/// Write linearreg_angle directly to output slice - no allocations
pub fn linearreg_angle_into_slice(
    dst: &mut [f64],
    input: &Linearreg_angleInput,
    kern: Kernel,
) -> Result<(), Linearreg_angleError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(Linearreg_angleError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(Linearreg_angleError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period < 2 || period > len {
        return Err(Linearreg_angleError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(Linearreg_angleError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    if dst.len() != len {
        return Err(Linearreg_angleError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Initialize warmup period with NaN before computation
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                linearreg_angle_scalar(data, period, first, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => linearreg_angle_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_angle_avx512(data, period, first, dst)
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_angle")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn linearreg_angle_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = Linearreg_angleParams {
        period: Some(period),
    };
    let linearreg_angle_in = Linearreg_angleInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| linearreg_angle_with_kernel(&linearreg_angle_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "Linearreg_angleStream")]
pub struct Linearreg_angleStreamPy {
    stream: Linearreg_angleStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl Linearreg_angleStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = Linearreg_angleParams {
            period: Some(period),
        };
        let stream = Linearreg_angleStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Linearreg_angleStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_angle_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn linearreg_angle_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;
    use std::mem::MaybeUninit;

    let slice_in = data.as_slice()?;

    let sweep = Linearreg_angleBatchRange {
        period: period_range,
    };
    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(PyValueError::new_err("linearreg_angle_batch: empty grid"));
    }
    let rows = combos.len();
    let cols = slice_in.len();

    // Validate all periods are >= 2 before computing warmups/allocating.
    for combo in &combos {
        let period = combo.period.unwrap();
        if period < 2 {
            return Err(PyValueError::new_err(
                Linearreg_angleError::InvalidPeriod {
                    period,
                    data_len: cols,
                }
                .to_string(),
            ));
        }
    }

    let total = rows.checked_mul(cols).ok_or_else(|| {
        PyValueError::new_err(
            Linearreg_angleError::InvalidRange {
                start: sweep.period.0,
                end: sweep.period.1,
                step: sweep.period.2,
            }
            .to_string(),
        )
    })?;

    // Allocate uninitialized flat output (rows*cols), like alma.rs
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Warmup prefixes via helper, zero-copy, before compute
    let first = slice_in
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| PyValueError::new_err("AllValuesNaN"))?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Reinterpret NumPy memory as MaybeUninit<f64> and initialize prefixes with NaN
    let mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(
            slice_out.as_mut_ptr() as *mut MaybeUninit<f64>,
            slice_out.len(),
        )
    };
    init_matrix_prefixes(mu, cols, &warm);

    // Resolve kernel once, map batch→simd
    let kern = validate_kernel(kernel, true)?;
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

    // Compute into pre-initialized buffer
    py.allow_threads(|| linearreg_angle_batch_inner_into(slice_in, &sweep, simd, true, slice_out))
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

// ---- CUDA Python bindings (device handles) ----
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct LinearregAngleDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl LinearregAngleDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.rows, self.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?
            .as_device_ptr()
            .as_raw() as usize;
        d.set_item("data", (ptr, false))?;
        // Producing stream is synchronized before return; omit explicit stream key.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
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

        // Move ownership of the underlying CUDA buffer into the shared DLPack helper.
        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = self.rows;
        let cols = self.cols;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "linearreg_angle_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn linearreg_angle_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<LinearregAngleDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = Linearreg_angleBatchRange {
        period: period_range,
    };
    let (buf, rows, cols, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaLinearregAngle::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = cuda
            .linearreg_angle_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let crate::cuda::moving_averages::DeviceArrayF32 { buf, rows, cols } = out;
        let ctx = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((buf, rows, cols, ctx, cuda.device_id()))
    })?;
    Ok(LinearregAngleDeviceArrayF32Py {
        buf: Some(buf),
        rows,
        cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "linearreg_angle_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, cols, rows, period, device_id=0))]
pub fn linearreg_angle_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    device_id: usize,
) -> PyResult<LinearregAngleDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let params = Linearreg_angleParams {
        period: Some(period),
    };
    let slice_in = data_tm_f32.as_slice()?;
    let (buf, r_out, c_out, ctx, dev_id) = py.allow_threads(|| {
        let cuda =
            CudaLinearregAngle::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = cuda
            .linearreg_angle_many_series_one_param_time_major_dev(slice_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let crate::cuda::moving_averages::DeviceArrayF32 { buf, rows, cols } = out;
        let ctx = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((buf, rows, cols, ctx, cuda.device_id()))
    })?;
    Ok(LinearregAngleDeviceArrayF32Py {
        buf: Some(buf),
        rows: r_out,
        cols: c_out,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(feature = "python")]
pub fn register_linearreg_angle_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(linearreg_angle_py, m)?)?;
    m.add_function(wrap_pyfunction!(linearreg_angle_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(linearreg_angle_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            linearreg_angle_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<LinearregAngleDeviceArrayF32Py>()?;
        m.add_class::<crate::indicators::moving_averages::alma::DeviceArrayF32Py>()?;
    }
    Ok(())
}

#[inline(always)]
fn linearreg_angle_batch_inner_into(
    data: &[f64],
    sweep: &Linearreg_angleBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<Linearreg_angleParams>, Linearreg_angleError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(Linearreg_angleError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    // Validate all periods are >= 2 (linear regression needs at least 2 points)
    for combo in &combos {
        let period = combo.period.unwrap();
        if period < 2 {
            return Err(Linearreg_angleError::InvalidPeriod {
                period,
                data_len: data.len(),
            });
        }
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(Linearreg_angleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let _ = combos
        .len()
        .checked_mul(max_p)
        .ok_or(Linearreg_angleError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    if data.len() - first < max_p {
        return Err(Linearreg_angleError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let cols = data.len();
    let expected = combos
        .len()
        .checked_mul(cols)
        .ok_or(Linearreg_angleError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        })?;
    if out.len() != expected {
        return Err(Linearreg_angleError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    // Treat out as uninitialized to avoid UB and copies, like alma.rs
    let out_uninit: &mut [core::mem::MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut core::mem::MaybeUninit<f64>,
            out.len(),
        )
    };

    let do_row = |row: usize, dst_mu: &mut [core::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        // Temporarily view as f64 for kernel write
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                linearreg_angle_row_scalar(data, first, period, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => linearreg_angle_row_avx2(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_angle_row_avx512(data, first, period, dst)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_angle_row_scalar(data, first, period, dst)
            }
            Kernel::Auto => unreachable!("resolve kernel before calling inner_into"),
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

// ============================================================================
// WASM Bindings
// ============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_angle_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = Linearreg_angleParams {
        period: Some(period),
    };
    let input = Linearreg_angleInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];
    linearreg_angle_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_angle_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_angle_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linearreg_angle_into(
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

        if period < 2 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = Linearreg_angleParams {
            period: Some(period),
        };
        let input = Linearreg_angleInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // Aliasing detected - use temp buffer
            let mut temp = vec![0.0; len];
            linearreg_angle_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing - write directly
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            linearreg_angle_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct Linearreg_angleBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct Linearreg_angleBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<Linearreg_angleParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = linearreg_angle_batch)]
pub fn linearreg_angle_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: Linearreg_angleBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = Linearreg_angleBatchRange {
        period: config.period_range,
    };

    // Use same kernel resolution as Python binding for consistency
    let kernel = detect_best_batch_kernel();
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };

    // Use same inner that does make_uninit_matrix + init_matrix_prefixes
    let output = linearreg_angle_batch_inner(data, &sweep, simd, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = Linearreg_angleBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
