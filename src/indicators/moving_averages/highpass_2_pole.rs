//! # 2-Pole High-Pass Filter
//!
//! A 2-pole high-pass filter using a user-specified cutoff frequency (`k`). This filter
//! removes or attenuates lower-frequency components from the input data.
//!
//! ## Parameters
//! - **period**: Window size (must be ≥ 2).
//! - **k**: Cutoff frequency (commonly in [0.0, 1.0]) controlling the filter's
//!          attenuation of low-frequency components (defaults to 0.707).
//!
//! ## Returns
//! - **`Ok(HighPass2Output)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(HighPass2Error)`** otherwise.
//!
//! ## Developer Status / Decision Log
//! - Scalar path optimized: 4x software pipeline + FMA via `mul_add`, `sin_cos` coefficients.
//!   On 100k candles: ~312µs -> ~78µs (~4x), measured with `-C target-cpu=native`.
//! - AVX2/AVX512: same pipeline; modest gains vs optimized scalar (~0–3%). AVX512 currently
//!   routes to the AVX2 body. Keep both enabled; scalar remains reference.
//! - Batch: adds optional precomputed second-difference shared across rows; engaged when rows ≥ 2.
//!   Helps large parameter grids; neutral overhead for single-row default benches.
//! - Streaming update: O(1) state rotation retained; semantics unchanged.
//! - Memory: follows `alma.rs` patterns (`alloc_with_nan_prefix`, `make_uninit_matrix`).

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, moving_averages::CudaHighPass2};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
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

#[derive(Debug, Clone)]
pub enum HighPass2Data<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for HighPass2Input<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HighPass2Data::Slice(slice) => slice,
            HighPass2Data::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HighPass2Output {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct HighPass2Params {
    pub period: Option<usize>,
    pub k: Option<f64>,
}

impl Default for HighPass2Params {
    fn default() -> Self {
        Self {
            period: Some(48),
            k: Some(0.707),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HighPass2Input<'a> {
    pub data: HighPass2Data<'a>,
    pub params: HighPass2Params,
}

impl<'a> HighPass2Input<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: HighPass2Params) -> Self {
        Self {
            data: HighPass2Data::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: HighPass2Params) -> Self {
        Self {
            data: HighPass2Data::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", HighPass2Params::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(48)
    }
    #[inline]
    pub fn get_k(&self) -> f64 {
        self.params.k.unwrap_or(0.707)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HighPass2Builder {
    period: Option<usize>,
    k: Option<f64>,
    kernel: Kernel,
}

impl Default for HighPass2Builder {
    fn default() -> Self {
        Self {
            period: None,
            k: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HighPass2Builder {
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
    pub fn k(mut self, val: f64) -> Self {
        self.k = Some(val);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<HighPass2Output, HighPass2Error> {
        let p = HighPass2Params {
            period: self.period,
            k: self.k,
        };
        let i = HighPass2Input::from_candles(c, "close", p);
        highpass_2_pole_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<HighPass2Output, HighPass2Error> {
        let p = HighPass2Params {
            period: self.period,
            k: self.k,
        };
        let i = HighPass2Input::from_slice(d, p);
        highpass_2_pole_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<HighPass2Stream, HighPass2Error> {
        let p = HighPass2Params {
            period: self.period,
            k: self.k,
        };
        HighPass2Stream::try_new(p)
    }
}

#[inline(always)]
fn highpass2_prepare<'a>(
    input: &'a HighPass2Input,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, f64, usize, Kernel), HighPass2Error> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HighPass2Error::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPass2Error::AllValuesNaN)?;
    let period = input.get_period();
    let k = input.get_k();

    if period < 2 || period > len {
        return Err(HighPass2Error::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if !(k > 0.0) || k.is_nan() || k.is_infinite() {
        return Err(HighPass2Error::InvalidK { k });
    }
    if len - first < period {
        return Err(HighPass2Error::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    Ok((data, period, k, first, chosen))
}

#[inline(always)]
fn warmup_end(first: usize, period: usize) -> usize {
    first + period - 1
}

#[derive(Debug, Error)]
pub enum HighPass2Error {
    #[error("highpass_2_pole: All values are NaN.")]
    AllValuesNaN,
    #[error("highpass_2_pole: Input data slice is empty.")]
    EmptyInputData,
    #[error("highpass_2_pole: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("highpass_2_pole: Invalid k value: {k}")]
    InvalidK { k: f64 },
    #[error("highpass_2_pole: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("highpass_2_pole: Output buffer length mismatch: out_len = {out_len}, expected = {expected}")]
    OutputLengthMismatch { out_len: usize, expected: usize },
}

/// Computes a two-pole high-pass filter using the best available kernel.
///
/// # Errors
/// Returns [`HighPass2Error`] when parameters are invalid.
#[inline]
pub fn highpass_2_pole(input: &HighPass2Input) -> Result<HighPass2Output, HighPass2Error> {
    highpass_2_pole_with_kernel(input, Kernel::Auto)
}

/// Computes a two-pole high-pass filter directly into the provided output buffer.
///
/// # Errors
/// Returns [`HighPass2Error`] when parameters are invalid.
#[inline]
pub fn highpass_2_pole_into(input: &HighPass2Input, out: &mut [f64]) -> Result<(), HighPass2Error> {
    highpass_2_pole_with_kernel_into(input, Kernel::Auto, out)
}

/// Computes a two-pole high-pass filter with a specific kernel.
///
/// # Errors
/// See [`HighPass2Error`] for details.
pub fn highpass_2_pole_with_kernel(
    input: &HighPass2Input,
    kernel: Kernel,
) -> Result<HighPass2Output, HighPass2Error> {
    let (data, period, k, first, chosen) = highpass2_prepare(input, kernel)?;
    let warm = warmup_end(first, period);
    let mut out = alloc_with_nan_prefix(data.len(), first); // Only NaN prefix up to first, not warm

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                highpass_2_pole_scalar_(data, period, k, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                highpass_2_pole_avx2(data, period, k, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                highpass_2_pole_avx512(data, period, k, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    // No NaN restoration needed - the kernels properly handle first
    // and alloc_with_nan_prefix already set the correct NaN prefix

    Ok(HighPass2Output { values: out })
}

/// Computes a two-pole high-pass filter with a specific kernel directly into the provided output buffer.
///
/// # Errors
/// See [`HighPass2Error`] for details.
fn highpass_2_pole_with_kernel_into(
    input: &HighPass2Input,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), HighPass2Error> {
    let (data, period, k, first, chosen) = highpass2_prepare(input, kernel)?;
    if out.len() != data.len() {
        return Err(HighPass2Error::OutputLengthMismatch {
            out_len: out.len(),
            expected: data.len(),
        });
    }
    let warm = warmup_end(first, period);
    // Only fill [0..first) with NaN, not the entire warmup range
    // The kernel will seed from first and compute the rest
    if first > 0 {
        out[..first].fill(f64::NAN);
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                highpass_2_pole_scalar_(data, period, k, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => highpass_2_pole_avx2(data, period, k, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                highpass_2_pole_avx512(data, period, k, first, out)
            }
            _ => unreachable!(),
        }
    }
    // No NaN restoration needed - the kernels properly handle first
    Ok(())
}

/// Scalar fallback implementation.
///
/// Optimized with precomputed coefficients, fused multiply-add chains
/// and a 4x software pipeline to reduce recurrence latency. Matches
/// the same seeding/warmup semantics as ALMA-style implementations:
/// - Preserve NaN prefix [0..first)
/// - Seed out[first..=first+1] from inputs when available
/// - Start recurrence from first+2
#[inline(always)]
pub unsafe fn highpass_2_pole_scalar_(
    data: &[f64],
    period: usize,
    k: f64,
    first: usize,
    out: &mut [f64],
) {
    use core::f64::consts::PI;
    let n = data.len();
    debug_assert!(out.len() >= n);
    debug_assert!(period >= 2 && period <= n);
    debug_assert!(first <= n);

    if n == 0 || first >= n {
        return;
    }

    // Coefficients: compute trig once using sin_cos to reduce overhead
    let theta = 2.0 * PI * k / (period as f64);
    let (s, c0) = theta.sin_cos();
    let alpha = 1.0 + ((s - 1.0) / c0);

    let one_minus_alpha = 1.0 - alpha;
    let c = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);

    // Precompute fused-friendly constants
    let cm2 = -2.0 * c;
    let two_1m = 2.0 * one_minus_alpha;
    let neg_oma_sq = -(one_minus_alpha * one_minus_alpha);

    // Seed (preserve NaN prefix [0..first))
    out[first] = data[first];
    if first + 1 >= n {
        return;
    }
    out[first + 1] = data[first + 1];
    if first + 2 >= n {
        return;
    }

    // Rolling state
    let mut x_im2 = data[first];
    let mut x_im1 = data[first + 1];
    let mut y_im2 = out[first];
    let mut y_im1 = out[first + 1];

    // Pointer walk with 4x unrolled software pipeline
    let mut src = data.as_ptr().add(first + 2);
    let mut dst = out.as_mut_ptr().add(first + 2);
    let mut rem = n - (first + 2);

    while rem >= 4 {
        // y[i+0]
        let x0 = *src;
        let t0 = cm2.mul_add(x_im1, c * x0);
        let t0 = c.mul_add(x_im2, t0);
        let y0 = two_1m.mul_add(y_im1, neg_oma_sq.mul_add(y_im2, t0));
        *dst = y0;

        // y[i+1]
        let x1 = *src.add(1);
        let t1 = cm2.mul_add(x0, c * x1);
        let t1 = c.mul_add(x_im1, t1);
        let y1 = two_1m.mul_add(y0, neg_oma_sq.mul_add(y_im1, t1));
        *dst.add(1) = y1;

        // y[i+2]
        let x2 = *src.add(2);
        let t2 = cm2.mul_add(x1, c * x2);
        let t2 = c.mul_add(x0, t2);
        let y2 = two_1m.mul_add(y1, neg_oma_sq.mul_add(y0, t2));
        *dst.add(2) = y2;

        // y[i+3]
        let x3 = *src.add(3);
        let t3 = cm2.mul_add(x2, c * x3);
        let t3 = c.mul_add(x1, t3);
        let y3 = two_1m.mul_add(y2, neg_oma_sq.mul_add(y1, t3));
        *dst.add(3) = y3;

        // rotate state for next block
        x_im2 = x2;
        x_im1 = x3;
        y_im2 = y2;
        y_im1 = y3;

        src = src.add(4);
        dst = dst.add(4);
        rem -= 4;
    }

    // Tail
    while rem > 0 {
        let xi = *src;
        let y = two_1m.mul_add(
            y_im1,
            neg_oma_sq.mul_add(
                y_im2,
                c.mul_add(x_im2, cm2.mul_add(x_im1, c * xi)),
            ),
        );
        *dst = y;

        x_im2 = x_im1;
        x_im1 = xi;
        y_im2 = y_im1;
        y_im1 = y;

        src = src.add(1);
        dst = dst.add(1);
        rem -= 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
/// AVX2 implementation of the filter.
#[inline]
pub unsafe fn highpass_2_pole_avx2(
    data: &[f64],
    period: usize,
    k: f64,
    first: usize,
    out: &mut [f64],
) {
    use core::f64::consts::PI;

    let n = data.len();
    if n == 0 || first >= n {
        return;
    }
    debug_assert!(out.len() >= n);
    debug_assert!(period >= 2 && period <= n);

    // ---- pre-compute coefficients ----
    let theta = 2.0 * PI * k / period as f64;
    let (s, c0) = theta.sin_cos();
    let alpha = 1.0 + ((s - 1.0) / c0);
    let c = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    let cm2 = -2.0 * c;
    let two_1m = 2.0 * (1.0 - alpha);
    let neg_oma_sq = -(1.0 - alpha) * (1.0 - alpha);

    // ---- seed starting from first ----
    // Leave [0..first) untouched (NaN)
    out[first] = data[first];
    if first + 1 >= n {
        return;
    }
    out[first + 1] = data[first + 1];
    if first + 2 >= n {
        return;
    }

    // ---- pointer iteration starting from first+2 ----
    let mut rem = n - (first + 2); // how many samples left
    let mut src = data.as_ptr().add(first + 2);
    let mut dst = out.as_mut_ptr().add(first + 2);

    // state registers initialized from first values
    let mut x_im2 = data[first];
    let mut x_im1 = data[first + 1];
    let mut y_im2 = out[first];
    let mut y_im1 = out[first + 1];

    while rem >= 4 {
        // y[i+0]
        let x0 = *src;
        let t0 = cm2.mul_add(x_im1, c * x0);
        let t0 = c.mul_add(x_im2, t0);
        let y0 = two_1m.mul_add(y_im1, neg_oma_sq.mul_add(y_im2, t0));
        *dst = y0;

        // y[i+1]
        let x1 = *src.add(1);
        let t1 = cm2.mul_add(x0, c * x1);
        let t1 = c.mul_add(x_im1, t1);
        let y1 = two_1m.mul_add(y0, neg_oma_sq.mul_add(y_im1, t1));
        *dst.add(1) = y1;

        // y[i+2]
        let x2 = *src.add(2);
        let t2 = cm2.mul_add(x1, c * x2);
        let t2 = c.mul_add(x0, t2);
        let y2 = two_1m.mul_add(y1, neg_oma_sq.mul_add(y0, t2));
        *dst.add(2) = y2;

        // y[i+3]
        let x3 = *src.add(3);
        let t3 = cm2.mul_add(x2, c * x3);
        let t3 = c.mul_add(x1, t3);
        let y3 = two_1m.mul_add(y2, neg_oma_sq.mul_add(y1, t3));
        *dst.add(3) = y3;

        // rotate state
        x_im2 = x2;
        x_im1 = x3;
        y_im2 = y2;
        y_im1 = y3;

        src = src.add(4);
        dst = dst.add(4);
        rem -= 4;
    }

    while rem > 0 {
        let xi = *src;
        let y = two_1m.mul_add(
            y_im1,
            neg_oma_sq.mul_add(
                y_im2,
                c.mul_add(x_im2, cm2.mul_add(x_im1, c * xi)),
            ),
        );
        *dst = y;

        x_im2 = x_im1;
        x_im1 = xi;
        y_im2 = y_im1;
        y_im1 = y;

        src = src.add(1);
        dst = dst.add(1);
        rem -= 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
/// AVX-512 implementation. Internally uses the AVX2 routine.
#[inline(always)]
pub fn highpass_2_pole_avx512(data: &[f64], period: usize, k: f64, first: usize, out: &mut [f64]) {
    unsafe { highpass_2_pole_avx2(data, period, k, first, out) }
}

#[derive(Debug, Clone)]
pub struct HighPass2Stream {
    period: usize,
    k: f64,
    // precomputed
    c: f64,
    two_1m: f64,
    neg_oma_sq: f64,
    // state
    x_im2: f64,
    x_im1: f64,
    y_im2: f64,
    y_im1: f64,
    idx: usize,
}

impl HighPass2Stream {
    pub fn try_new(params: HighPass2Params) -> Result<Self, HighPass2Error> {
        let period = params.period.unwrap_or(48);
        let k = params.k.unwrap_or(0.707);
        if period < 2 {
            return Err(HighPass2Error::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if !(k > 0.0) || !k.is_finite() {
            return Err(HighPass2Error::InvalidK { k });
        }

        use std::f64::consts::PI;
        let angle = 2.0 * PI * k / (period as f64);
        let sin_val = angle.sin();
        let cos_val = angle.cos();
        let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
        let c = (1.0 - 0.5 * alpha).powi(2);
        let two_1m = 2.0 * (1.0 - alpha);
        let neg_oma_sq = -(1.0 - alpha).powi(2);

        Ok(Self {
            period,
            k,
            c,
            two_1m,
            neg_oma_sq,
            x_im2: f64::NAN,
            x_im1: f64::NAN,
            y_im2: f64::NAN,
            y_im1: f64::NAN,
            idx: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x_i: f64) -> Option<f64> {
        // seed first two samples as pass-through like batch kernel, but suppress output until warmup_end
        let y_i = if self.idx == 0 {
            self.x_im2 = x_i;
            self.y_im2 = x_i;
            x_i
        } else if self.idx == 1 {
            self.x_im1 = x_i;
            self.y_im1 = x_i;
            x_i
        } else {
            // recurrence
            let t = self.c * x_i - 2.0 * self.c * self.x_im1
                + self.c * self.x_im2
                + self.two_1m * self.y_im1
                + self.neg_oma_sq * self.y_im2;

            // rotate state
            self.x_im2 = self.x_im1;
            self.x_im1 = x_i;
            self.y_im2 = self.y_im1;
            self.y_im1 = t;
            t
        };

        let out = if self.idx + 1 >= (self.period) {
            Some(y_i)
        } else {
            None
        };
        self.idx += 1;
        out
    }
}

#[derive(Clone, Debug)]
pub struct HighPass2BatchRange {
    pub period: (usize, usize, usize),
    pub k: (f64, f64, f64),
}

impl Default for HighPass2BatchRange {
    fn default() -> Self {
        Self {
            period: (48, 120, 0),
            k: (0.707, 0.707, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HighPass2BatchBuilder {
    range: HighPass2BatchRange,
    kernel: Kernel,
}

impl HighPass2BatchBuilder {
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
    pub fn k_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.k = (start, end, step);
        self
    }
    pub fn k_static(mut self, val: f64) -> Self {
        self.range.k = (val, val, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<HighPass2BatchOutput, HighPass2Error> {
        highpass_2_pole_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<HighPass2BatchOutput, HighPass2Error> {
        HighPass2BatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<HighPass2BatchOutput, HighPass2Error> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HighPass2BatchOutput, HighPass2Error> {
        HighPass2BatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn highpass_2_pole_batch_with_kernel(
    data: &[f64],
    sweep: &HighPass2BatchRange,
    k: Kernel,
) -> Result<HighPass2BatchOutput, HighPass2Error> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(HighPass2Error::InvalidPeriod {
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
    highpass_2_pole_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct HighPass2BatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HighPass2Params>,
    pub rows: usize,
    pub cols: usize,
}
impl HighPass2BatchOutput {
    pub fn row_for_params(&self, p: &HighPass2Params) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(48) == p.period.unwrap_or(48)
                && (c.k.unwrap_or(0.707) - p.k.unwrap_or(0.707)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &HighPass2Params) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &HighPass2BatchRange) -> Vec<HighPass2Params> {
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
    let ks = axis_f64(r.k);
    let mut out = Vec::with_capacity(periods.len() * ks.len());
    for &p in &periods {
        for &k in &ks {
            out.push(HighPass2Params {
                period: Some(p),
                k: Some(k),
            });
        }
    }
    out
}

#[inline(always)]
pub fn highpass_2_pole_batch_slice(
    data: &[f64],
    sweep: &HighPass2BatchRange,
    kern: Kernel,
) -> Result<HighPass2BatchOutput, HighPass2Error> {
    highpass_2_pole_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn highpass_2_pole_batch_par_slice(
    data: &[f64],
    sweep: &HighPass2BatchRange,
    kern: Kernel,
) -> Result<HighPass2BatchOutput, HighPass2Error> {
    highpass_2_pole_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn highpass_2_pole_batch_inner(
    data: &[f64],
    sweep: &HighPass2BatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<HighPass2BatchOutput, HighPass2Error> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(HighPass2Error::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPass2Error::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(HighPass2Error::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // per-row warmups: first + period - 1
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| warmup_end(first, c.period.unwrap()))
        .collect();

    // 1) allocate and seed prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Optionally precompute second-difference once and reuse across rows
    // Δ²x[i] = x[i] - 2x[i-1] + x[i-2]
    let dd: Option<Vec<f64>> = if rows >= 2 {
        let mut v = vec![0.0_f64; cols];
        if first + 2 < cols {
            // compute only where used; prefix doesn't matter
            for i in (first + 2)..cols {
                // no need for mul_add here; done once per batch
                v[i] = data[i] - 2.0 * data[i - 1] + data[i - 2];
            }
        }
        Some(v)
    } else {
        None
    };

    // 2) compute rows
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let k = combos[row].k.unwrap();
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => highpass_2_pole_row_avx512(data, first, period, k, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => highpass_2_pole_row_avx2(data, first, period, k, out_row),
            _ => {
                if let Some(ref d) = dd {
                    highpass_2_pole_row_scalar_dd(data, d, first, period, k, out_row)
                } else {
                    highpass_2_pole_row_scalar(data, first, period, k, out_row)
                }
            }
        }
        // The kernels now properly handle first, no NaN restoration needed
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            buf_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, s)| do_row(r, s));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, s) in buf_mu.chunks_mut(cols).enumerate() {
                do_row(r, s);
            }
        }
    } else {
        for (r, s) in buf_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    // 3) move out safely
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(HighPass2BatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn highpass_2_pole_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    k: f64,
    out: &mut [f64],
) {
    highpass_2_pole_scalar_(data, period, k, first, out);
}

/// Scalar per-row kernel using a precomputed second-difference `dd` shared across rows.
/// dd[i] = x[i] - 2x[i-1] + x[i-2]; only used for indices >= first+2.
#[inline(always)]
pub unsafe fn highpass_2_pole_row_scalar_dd(
    data: &[f64],
    dd: &[f64],
    first: usize,
    period: usize,
    k: f64,
    out: &mut [f64],
) {
    use core::f64::consts::PI;
    let n = data.len();
    debug_assert_eq!(dd.len(), n);
    debug_assert!(out.len() >= n);
    debug_assert!(period >= 2 && period <= n);
    debug_assert!(first <= n);
    if n == 0 || first >= n {
        return;
    }

    let theta = 2.0 * PI * k / (period as f64);
    let (s, c0) = theta.sin_cos();
    let alpha = 1.0 + ((s - 1.0) / c0);
    let one_minus_alpha = 1.0 - alpha;
    let c = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    let two_1m = 2.0 * one_minus_alpha;
    let neg_oma_sq = -(one_minus_alpha * one_minus_alpha);

    // Seed identical to scalar path
    out[first] = data[first];
    if first + 1 >= n {
        return;
    }
    out[first + 1] = data[first + 1];
    if first + 2 >= n {
        return;
    }

    // Rolling state
    let mut y_im2 = out[first];
    let mut y_im1 = out[first + 1];

    // Pointer walk over dd, starting at first+2
    let mut src = dd.as_ptr().add(first + 2);
    let mut dst = out.as_mut_ptr().add(first + 2);
    let mut rem = n - (first + 2);

    while rem >= 4 {
        let d0 = *src;
        let y0 = two_1m.mul_add(y_im1, neg_oma_sq.mul_add(y_im2, c * d0));
        *dst = y0;

        let d1 = *src.add(1);
        let y1 = two_1m.mul_add(y0, neg_oma_sq.mul_add(y_im1, c * d1));
        *dst.add(1) = y1;

        let d2 = *src.add(2);
        let y2 = two_1m.mul_add(y1, neg_oma_sq.mul_add(y0, c * d2));
        *dst.add(2) = y2;

        let d3 = *src.add(3);
        let y3 = two_1m.mul_add(y2, neg_oma_sq.mul_add(y1, c * d3));
        *dst.add(3) = y3;

        y_im2 = y2;
        y_im1 = y3;

        src = src.add(4);
        dst = dst.add(4);
        rem -= 4;
    }

    while rem > 0 {
        let di = *src;
        let y = two_1m.mul_add(y_im1, neg_oma_sq.mul_add(y_im2, c * di));
        *dst = y;
        y_im2 = y_im1;
        y_im1 = y;
        src = src.add(1);
        dst = dst.add(1);
        rem -= 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_2_pole_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    k: f64,
    out: &mut [f64],
) {
    highpass_2_pole_avx2(data, period, k, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_2_pole_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    k: f64,
    out: &mut [f64],
) {
    highpass_2_pole_row_avx2(data, first, period, k, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn check_highpass2_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = HighPass2Params {
            period: None,
            k: None,
        };
        let input = HighPass2Input::from_candles(&candles, "close", default_params);
        let output = highpass_2_pole_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_highpass2_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPass2Input::from_candles(&candles, "close", HighPass2Params::default());
        let result = highpass_2_pole_with_kernel(&input, kernel)?;
        let expected_last_five = [
            445.29073821108943,
            359.51467478973296,
            250.7236793408186,
            394.04381266217234,
            -52.65414073315134,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] HighPass2 {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_highpass2_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPass2Input::with_default_candles(&candles);
        match input.data {
            HighPass2Data::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected HighPass2Data::Candles"),
        }
        let output = highpass_2_pole_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_highpass2_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HighPass2Params {
            period: Some(0),
            k: None,
        };
        let input = HighPass2Input::from_slice(&input_data, params);
        let res = highpass_2_pole_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] HighPass2 should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_highpass2_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = HighPass2Params {
            period: Some(10),
            k: None,
        };
        let input = HighPass2Input::from_slice(&data_small, params);
        let res = highpass_2_pole_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] HighPass2 should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_highpass2_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = HighPass2Params {
            period: Some(2),
            k: None,
        };
        let input = HighPass2Input::from_slice(&single_point, params);
        let res = highpass_2_pole_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), single_point.len());
        assert_eq!(res.values[0], single_point[0]);
        Ok(())
    }
    fn check_highpass2_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = HighPass2Params {
            period: Some(48),
            k: None,
        };
        let first_input = HighPass2Input::from_candles(&candles, "close", first_params);
        let first_result = highpass_2_pole_with_kernel(&first_input, kernel)?;
        let second_params = HighPass2Params {
            period: Some(32),
            k: None,
        };
        let second_input = HighPass2Input::from_slice(&first_result.values, second_params);
        let second_result = highpass_2_pole_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(!second_result.values[i].is_nan());
        }
        Ok(())
    }
    fn check_highpass2_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPass2Input::from_candles(&candles, "close", HighPass2Params::default());
        let res = highpass_2_pole_with_kernel(&input, kernel)?;
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

    fn check_highpass2_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = HighPass2Input::from_slice(&empty, HighPass2Params::default());
        let res = highpass_2_pole_with_kernel(&input, kernel);
        assert!(matches!(res, Err(HighPass2Error::EmptyInputData)));
        Ok(())
    }

    fn check_highpass2_invalid_k(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = HighPass2Params {
            period: Some(2),
            k: Some(-0.5),
        };
        let input = HighPass2Input::from_slice(&data, params);
        let res = highpass_2_pole_with_kernel(&input, kernel);
        assert!(matches!(res, Err(HighPass2Error::InvalidK { .. })));
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_highpass2_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        use proptest::prelude::*;
        use std::f64::consts::PI;

        // Strategy for generating test parameters and data
        let strat = (2usize..=50) // period range
            .prop_flat_map(|period| {
                (
                    // Generate realistic data with varying patterns
                    prop::collection::vec(
                        (-1000f64..1000f64).prop_filter("finite", |x| x.is_finite()),
                        period.max(10)..400, // data length from max(period,10) to 400
                    ),
                    Just(period),
                    0.01f64..0.99f64, // k parameter range (extended to test boundaries)
                )
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, k)| {
                let params = HighPass2Params {
                    period: Some(period),
                    k: Some(k),
                };
                let input = HighPass2Input::from_slice(&data, params);

                // Compute with specified kernel
                let HighPass2Output { values: out } =
                    highpass_2_pole_with_kernel(&input, kernel).unwrap();

                // Also compute with scalar kernel as reference
                let HighPass2Output { values: ref_out } =
                    highpass_2_pole_with_kernel(&input, Kernel::Scalar).unwrap();

                // Basic property: output length equals input length
                prop_assert_eq!(out.len(), data.len());

                // Property 1: Verify initial seeding behavior
                // Implementation seeds first two values directly from input
                if data.len() > 0 && data[0].is_finite() {
                    prop_assert!(
                        (out[0] - data[0]).abs() < 1e-10,
                        "First output {} should match first input {}",
                        out[0],
                        data[0]
                    );
                }
                if data.len() > 1 && data[1].is_finite() {
                    prop_assert!(
                        (out[1] - data[1]).abs() < 1e-10,
                        "Second output {} should match second input {}",
                        out[1],
                        data[1]
                    );
                }

                // Property 2: Verify the recursive filter equation for i >= 2
                // y[i] = c*(x[i] - 2*x[i-1] + x[i-2]) + 2*(1-α)*y[i-1] - (1-α)²*y[i-2]
                if data.len() >= 3 {
                    // Calculate filter coefficients
                    let angle = 2.0 * PI * k / (period as f64);
                    let sin_val = angle.sin();
                    let cos_val = angle.cos();
                    let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
                    let c = (1.0 - alpha / 2.0).powi(2);
                    let one_minus_alpha = 1.0 - alpha;
                    let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

                    // Verify a few points of the filter equation
                    for i in 2..(10.min(data.len())) {
                        if data[i].is_finite() && data[i - 1].is_finite() && data[i - 2].is_finite()
                        {
                            let expected = c * data[i] - 2.0 * c * data[i - 1]
                                + c * data[i - 2]
                                + 2.0 * one_minus_alpha * out[i - 1]
                                - one_minus_alpha_sq * out[i - 2];
                            prop_assert!(
                                (out[i] - expected).abs() < 1e-10,
                                "Filter equation mismatch at index {}: got {} expected {}",
                                i,
                                out[i],
                                expected
                            );
                        }
                    }
                }

                // Property 3: DC removal test (fixed - no permissive OR condition)
                // For constant input, output should converge toward zero
                let const_start = data.len().saturating_sub(period * 3);
                let const_end = data.len();
                if const_start < const_end && const_end - const_start >= period {
                    let window = &data[const_start..const_end];
                    if window.iter().all(|&x| x.is_finite()) {
                        let mean = window.iter().sum::<f64>() / window.len() as f64;
                        let is_constant = window.iter().all(|&x| (x - mean).abs() < 1e-6);

                        if is_constant && mean.abs() > 1e-6 {
                            // For constant input, high-pass output should be near zero at the end
                            let final_out = out[const_end - 1];
                            let relative_output = final_out.abs() / mean.abs();
                            prop_assert!(relative_output < 0.01,
								"DC not removed: output {} is {:.2}% of constant input {} at index {}",
								final_out, relative_output * 100.0, mean, const_end - 1);
                        }
                    }
                }

                // Property 4: Step response test
                // For a significant step change in input, high-pass should produce a transient response
                // Note: Due to the recursive nature and period, decay may not be monotonic
                if data.len() > period * 3 {
                    // Look for large changes in the data
                    for i in period..(data.len() - period * 2) {
                        if i > 0 && data[i].is_finite() && data[i - 1].is_finite() {
                            let change = (data[i] - data[i - 1]).abs();
                            if change > 200.0 {
                                // Significant change
                                // High-pass filter responds to changes
                                // Check that output shows response to the change
                                if out[i].is_finite() {
                                    // Output should show some response to the change
                                    prop_assert!(
                                        out[i].abs() > 0.01,
                                        "High-pass should respond to change of {} at index {}",
                                        change,
                                        i
                                    );
                                }
                                break; // Only test one step per data set
                            }
                        }
                    }
                }

                // Property 5: Frequency response - alternating signal should pass through
                // Test with a section of alternating values
                if data.len() > 10 {
                    let alt_start = 5;
                    let alt_end = (alt_start + 8).min(data.len());
                    let mut is_alternating = true;
                    for i in (alt_start + 1)..alt_end {
                        if data[i].is_finite() && data[i - 1].is_finite() {
                            // Check if signs alternate
                            if data[i] * data[i - 1] > 0.0 {
                                is_alternating = false;
                                break;
                            }
                        } else {
                            is_alternating = false;
                            break;
                        }
                    }

                    if is_alternating && alt_end > alt_start + 4 {
                        // High frequency (alternating) signal should pass through
                        // Output amplitude should be significant relative to input
                        let input_amp = data[alt_start..alt_end]
                            .iter()
                            .filter(|x| x.is_finite())
                            .map(|x| x.abs())
                            .fold(0.0, f64::max);
                        let output_amp = out[alt_start..alt_end]
                            .iter()
                            .filter(|x| x.is_finite())
                            .map(|x| x.abs())
                            .fold(0.0, f64::max);

                        if input_amp > 1e-6 {
                            let pass_ratio = output_amp / input_amp;
                            prop_assert!(
                                pass_ratio > 0.1,
                                "High frequency should pass: input_amp={} output_amp={} ratio={}",
                                input_amp,
                                output_amp,
                                pass_ratio
                            );
                        }
                    }
                }

                // Property 6: Cross-kernel consistency check
                for i in 2..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        // Both should handle non-finite values the same way
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "Non-finite mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        // Check floating point consistency
                        let abs_diff = (y - r).abs();
                        let rel_diff = if r.abs() > 1e-10 {
                            abs_diff / r.abs()
                        } else {
                            abs_diff
                        };

                        prop_assert!(
                            abs_diff < 1e-9 || rel_diff < 1e-10,
                            "Kernel mismatch at index {}: {} vs {} (abs_diff={}, rel_diff={})",
                            i,
                            y,
                            r,
                            abs_diff,
                            rel_diff
                        );
                    }
                }

                // Property 7: Filter stability - output should remain bounded
                // For bounded input, output should not explode
                let input_bound = data
                    .iter()
                    .filter(|x| x.is_finite())
                    .map(|x| x.abs())
                    .fold(0.0, f64::max);

                if input_bound > 0.0 && input_bound.is_finite() {
                    // Theoretical bound for 2-pole high-pass filter
                    // Maximum gain occurs at Nyquist frequency
                    let max_gain = 4.0; // Conservative upper bound for high-pass gain
                    let expected_bound = input_bound * max_gain;

                    for (i, &val) in out.iter().enumerate() {
                        if val.is_finite() && i >= 2 {
                            prop_assert!(
                                val.abs() <= expected_bound,
                                "Output {} at index {} exceeds stability bound {} (input_bound={})",
                                val,
                                i,
                                expected_bound,
                                input_bound
                            );
                        }
                    }
                }

                // Property 8: Check for poison values
                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        let bits = val.to_bits();
                        prop_assert!(
                            bits != 0x11111111_11111111,
                            "Found alloc_with_nan_prefix poison at index {}",
                            i
                        );
                        prop_assert!(
                            bits != 0x22222222_22222222,
                            "Found init_matrix_prefixes poison at index {}",
                            i
                        );
                        prop_assert!(
                            bits != 0x33333333_33333333,
                            "Found make_uninit_matrix poison at index {}",
                            i
                        );
                    }
                }

                // Property 9: Boundary k values (near 0 and 1) should still produce valid output
                if k < 0.02 || k > 0.98 {
                    // Even at extreme k values, filter should remain stable
                    let finite_count = out.iter().filter(|x| x.is_finite()).count();
                    let expected_finite = data.iter().filter(|x| x.is_finite()).count();
                    prop_assert!(
                        finite_count >= expected_finite.saturating_sub(2),
                        "Filter unstable at k={}: only {} finite outputs from {} finite inputs",
                        k,
                        finite_count,
                        expected_finite
                    );
                }

                Ok(())
            })
            .unwrap();
        Ok(())
    }

    // Keep the existing simple property test as a fallback for non-proptest builds
    #[cfg(not(feature = "proptest"))]
    fn check_highpass2_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        // Simple test when proptest feature is not enabled
        let data = vec![5.0; 100];
        let params = HighPass2Params {
            period: Some(10),
            k: Some(0.707),
        };
        let input = HighPass2Input::from_slice(&data, params);
        let out = highpass_2_pole_with_kernel(&input, kernel)?;
        assert_eq!(out.values.len(), data.len());
        Ok(())
    }
    fn check_highpass2_first_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Create test data with leading NaNs
        let data = vec![
            f64::NAN,
            f64::NAN,
            f64::NAN, // First 3 values are NaN
            100.0,
            102.0,
            98.0,
            103.0,
            97.0,
            105.0,
            99.0, // Valid data starts at index 3
            101.0,
            104.0,
            96.0,
            102.0,
            100.0,
        ];

        let params = HighPass2Params {
            period: Some(5),
            k: Some(0.707),
        };
        let input = HighPass2Input::from_slice(&data, params);
        let output = highpass_2_pole_with_kernel(&input, kernel)?;

        // Check that leading NaNs are preserved
        for i in 0..3 {
            assert!(
                output.values[i].is_nan(),
                "[{}] Output at index {} should be NaN but got {}",
                test_name,
                i,
                output.values[i]
            );
        }

        // Check that we have valid output after warmup
        // first = 3, period = 5, so warmup_end = 3 + 5 - 1 = 7
        for i in 7..data.len() {
            assert!(
                !output.values[i].is_nan(),
                "[{}] Output at index {} should be valid but got NaN",
                test_name,
                i
            );
        }

        // Test with into_slice variant to ensure it also respects first
        let mut out_slice = vec![999.0; data.len()]; // Fill with non-NaN value
        highpass_2_pole_into(&input, &mut out_slice)?;

        // Check that leading NaNs are preserved
        for i in 0..3 {
            assert!(
                out_slice[i].is_nan(),
                "[{}] into_slice: Output at index {} should be NaN but got {}",
                test_name,
                i,
                out_slice[i]
            );
        }

        Ok(())
    }

    macro_rules! generate_all_highpass2_tests {
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
    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_highpass2_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations
        let test_cases = vec![
            HighPass2Params {
                period: Some(48),
                k: Some(0.707),
            }, // default
            HighPass2Params {
                period: Some(10),
                k: Some(0.3),
            }, // small period, low k
            HighPass2Params {
                period: Some(100),
                k: Some(0.9),
            }, // large period, high k
            HighPass2Params {
                period: Some(20),
                k: Some(0.5),
            }, // medium values
            HighPass2Params {
                period: Some(2),
                k: Some(0.1),
            }, // minimum period, low k
            HighPass2Params {
                period: Some(60),
                k: Some(0.8),
            }, // larger period
            HighPass2Params {
                period: Some(30),
                k: Some(0.2),
            }, // different combination
            HighPass2Params {
                period: None,
                k: None,
            }, // None values (use defaults)
            HighPass2Params {
                period: Some(15),
                k: Some(0.95),
            }, // edge case high k
        ];

        for params in test_cases {
            let input = HighPass2Input::from_candles(&candles, "close", params.clone());
            let output = highpass_2_pole_with_kernel(&input, kernel)?;

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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}, k={:?}",
                        test_name, val, bits, i, params.period, params.k
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                         with params period={:?}, k={:?}",
                        test_name, val, bits, i, params.period, params.k
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}, k={:?}",
                        test_name, val, bits, i, params.period, params.k
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_highpass2_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_highpass2_tests!(
        check_highpass2_partial_params,
        check_highpass2_accuracy,
        check_highpass2_default_candles,
        check_highpass2_zero_period,
        check_highpass2_period_exceeds_length,
        check_highpass2_very_small_dataset,
        check_highpass2_reinput,
        check_highpass2_nan_handling,
        check_highpass2_empty_input,
        check_highpass2_invalid_k,
        check_highpass2_no_poison,
        check_highpass2_first_handling
    );

    #[cfg(feature = "proptest")]
    generate_all_highpass2_tests!(check_highpass2_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = HighPass2BatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = HighPass2Params::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            445.29073821108943,
            359.51467478973296,
            250.7236793408186,
            394.04381266217234,
            -52.65414073315134,
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
    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations with different parameter ranges
        let batch_configs = vec![
            // Original test case
            ((10, 30, 10), (0.5, 0.9, 0.2)),
            // Edge cases
            ((2, 2, 0), (0.1, 0.1, 0.0)), // Single parameter combo, minimum period
            ((100, 120, 20), (0.2, 0.8, 0.3)), // Larger periods
            ((5, 15, 5), (0.1, 0.5, 0.2)), // Small periods, varying k
            ((20, 60, 20), (0.3, 0.9, 0.3)), // Medium periods
            ((48, 48, 0), (0.707, 0.707, 0.0)), // Default only
            ((3, 12, 3), (0.05, 0.95, 0.45)), // Wide k range
        ];

        for ((p_start, p_end, p_step), (k_start, k_end, k_step)) in batch_configs {
            let output = HighPass2BatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .k_range(k_start, k_end, k_step)
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
                let combo = &output.combos[row];

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}, k={:?}",
						test, val, bits, row, col, idx, combo.period, combo.k
					);
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}, k={:?}",
						test, val, bits, row, col, idx, combo.period, combo.k
					);
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}, k={:?}",
						test, val, bits, row, col, idx, combo.period, combo.k
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

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

/// highpass_2_pole_batch_inner_into writes directly to the output buffer
#[inline(always)]
fn highpass_2_pole_batch_inner_into(
    data: &[f64],
    sweep: &HighPass2BatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<HighPass2Params>, HighPass2Error> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(HighPass2Error::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPass2Error::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(HighPass2Error::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // per-row warmups: first + period - 1
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| warmup_end(first, c.period.unwrap()))
        .collect();

    // Reinterpret output slice as MaybeUninit
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    // Initialize NaN prefixes
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let k = combos[row].k.unwrap();
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => highpass_2_pole_row_avx512(data, first, period, k, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => highpass_2_pole_row_avx2(data, first, period, k, out_row),
            _ => highpass_2_pole_row_scalar(data, first, period, k, out_row),
        }
        // The kernels now properly handle first, no NaN restoration needed
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

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "highpass_2_pole")]
#[pyo3(signature = (data, period, k, kernel=None))]
pub fn highpass_2_pole_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    k: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = HighPass2Params {
        period: Some(period),
        k: Some(k),
    };
    let hp2_in = HighPass2Input::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| highpass_2_pole_with_kernel(&hp2_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "highpass_2_pole_batch")]
#[pyo3(signature = (data, period_range, k_range, kernel=None))]
pub fn highpass_2_pole_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    k_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = HighPass2BatchRange {
        period: period_range,
        k: k_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

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
            highpass_2_pole_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
    dict.set_item(
        "k",
        combos
            .iter()
            .map(|p| p.k.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "highpass_2_pole_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range=(48, 48, 0), k_range=(0.707, 0.707, 0.0), device_id=0))]
pub fn highpass_2_pole_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    k_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = HighPass2BatchRange {
        period: period_range,
        k: k_range,
    };

    let inner = py.allow_threads(|| -> PyResult<_> {
        let cuda =
            CudaHighPass2::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.highpass2_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "highpass_2_pole_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, k, device_id=0))]
pub fn highpass_2_pole_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    period: usize,
    k: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    if period < 2 {
        return Err(PyValueError::new_err("period must be >= 2"));
    }
    if !(k > 0.0) || !k.is_finite() {
        return Err(PyValueError::new_err("k must be positive and finite"));
    }

    let flat = data_tm_f32.as_slice()?;
    let shape = data_tm_f32.shape();
    let series_len = shape[0];
    let num_series = shape[1];
    let params = HighPass2Params {
        period: Some(period),
        k: Some(k),
    };

    let inner = py.allow_threads(|| -> PyResult<_> {
        let cuda =
            CudaHighPass2::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.highpass2_many_series_one_param_time_major_dev(flat, num_series, series_len, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyclass(name = "HighPass2Stream")]
pub struct HighPass2StreamPy {
    inner: HighPass2Stream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HighPass2StreamPy {
    #[new]
    pub fn new(period: usize, k: f64) -> PyResult<Self> {
        let params = HighPass2Params {
            period: Some(period),
            k: Some(k),
        };
        let inner =
            HighPass2Stream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// ================== WASM bindings ==================
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn highpass_2_pole_js(data: &[f64], period: usize, k: f64) -> Result<Vec<f64>, JsValue> {
    let params = HighPass2Params {
        period: Some(period),
        k: Some(k),
    };
    let input = HighPass2Input::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    highpass_2_pole_into(&input, &mut output).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn highpass_2_pole_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn highpass_2_pole_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = highpass_2_pole_into)]
pub fn highpass_2_pole_into_wasm(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    k: f64,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = HighPass2Params {
            period: Some(period),
            k: Some(k),
        };
        let input = HighPass2Input::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid corruption during sliding window computation
            let mut temp = vec![0.0; len];
            highpass_2_pole_into(&input, &mut temp)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            highpass_2_pole_into(&input, out).map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// ================== Batch Processing ==================

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct HighPass2BatchConfig {
    pub period_range: (usize, usize, usize),
    pub k_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct HighPass2BatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HighPass2Params>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = highpass_2_pole_batch)]
pub fn highpass_2_pole_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: HighPass2BatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = HighPass2BatchRange {
        period: config.period_range,
        k: config.k_range,
    };

    // match ALMA pattern: run batch detector then map to compute kernel without rayon in WASM
    let batch_kernel = detect_best_batch_kernel();
    let compute_kernel = match batch_kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };
    let out = highpass_2_pole_batch_inner(data, &sweep, compute_kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = HighPass2BatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };

    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ================== Optimized Batch Processing ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn highpass_2_pole_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    k_start: f64,
    k_end: f64,
    k_step: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = HighPass2BatchRange {
            period: (period_start, period_end, period_step),
            k: (k_start, k_end, k_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        if rows * cols == 0 {
            return Err(JsValue::from_str("Invalid dimensions"));
        }

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Detect best batch kernel
        let kernel = detect_best_batch_kernel();

        // Map batch kernel to regular kernel for computation
        let compute_kernel = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        };

        // Batch computation directly into output buffer
        highpass_2_pole_batch_inner_into(data, &sweep, compute_kernel, true, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(())
    }
}
