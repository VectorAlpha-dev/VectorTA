//! # Correlation Cycle (John Ehlers)
//!
//! Computes real, imag, angle, and market state outputs based on correlation phasor analysis.
//! Uses phase correlation to identify market cycles and determine market state (trending vs cycling).
//!
//! ## Parameters
//! - **period**: Window size for correlation calculation (default: 20)
//! - **threshold**: Threshold for market state determination (default: 9.0)
//!
//! ## Returns
//! - **`Ok(CorrelationCycleOutput)`** on success, containing four `Vec<f64>` arrays:
//!   - real: Real component of the phasor
//!   - imag: Imaginary component of the phasor
//!   - angle: Phase angle in degrees
//!   - state: Market state (1.0 for trending, 0.0 for cycling)
//! - **`Err(CorrelationCycleError)`** on various error conditions.
//!
//! ## Developer Status
//! - SIMD enabled: AVX2/AVX512 accumulate four jammed dot-products per window. On a 100k series,
//!   AVX2 outperforms scalar by >25% and AVX512 is also faster than scalar (slightly behind AVX2 on this CPU).
//! - Scalar path optimized: loop-jammed accumulators, trig/denominator precompute, fused state pass; matches tests.
//! - Batch: row-specific SIMD not implemented; potential future win is caching trig tables for identical periods across rows.
//! - CUDA/Python: CUDA wrapper with typed errors plus Python CAI v3/DLPack v1.x zero-copy interop; VRAM handles retain device context and id.
//! - Memory: uses alloc_with_nan_prefix/make_uninit_matrix/init_matrix_prefixes; no O(N) temporaries for outputs.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context as CudaContext;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc as StdArc;

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
use std::mem::ManuallyDrop;
use thiserror::Error;

impl<'a> AsRef<[f64]> for CorrelationCycleInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CorrelationCycleData::Slice(slice) => slice,
            CorrelationCycleData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CorrelationCycleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleOutput {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub angle: Vec<f64>,
    pub state: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CorrelationCycleParams {
    pub period: Option<usize>,
    pub threshold: Option<f64>,
}

impl Default for CorrelationCycleParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            threshold: Some(9.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleInput<'a> {
    pub data: CorrelationCycleData<'a>,
    pub params: CorrelationCycleParams,
}

impl<'a> CorrelationCycleInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: CorrelationCycleParams,
    ) -> Self {
        Self {
            data: CorrelationCycleData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: CorrelationCycleParams) -> Self {
        Self {
            data: CorrelationCycleData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", CorrelationCycleParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }

    #[inline]
    pub fn get_threshold(&self) -> f64 {
        self.params.threshold.unwrap_or(9.0)
    }
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleBuilder {
    period: Option<usize>,
    threshold: Option<f64>,
    kernel: Kernel,
}

impl Default for CorrelationCycleBuilder {
    fn default() -> Self {
        Self {
            period: None,
            threshold: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CorrelationCycleBuilder {
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
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = Some(t);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
        let p = CorrelationCycleParams {
            period: self.period,
            threshold: self.threshold,
        };
        let i = CorrelationCycleInput::from_candles(c, "close", p);
        correlation_cycle_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
        let p = CorrelationCycleParams {
            period: self.period,
            threshold: self.threshold,
        };
        let i = CorrelationCycleInput::from_slice(d, p);
        correlation_cycle_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CorrelationCycleStream, CorrelationCycleError> {
        let p = CorrelationCycleParams {
            period: self.period,
            threshold: self.threshold,
        };
        CorrelationCycleStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CorrelationCycleError {
    #[error("correlation_cycle: input data slice is empty")]
    EmptyInputData,
    #[error("correlation_cycle: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("correlation_cycle: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("correlation_cycle: All values are NaN.")]
    AllValuesNaN,
    #[error("correlation_cycle: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("correlation_cycle: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("correlation_cycle: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("correlation_cycle: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
pub fn correlation_cycle(
    input: &CorrelationCycleInput,
) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
    correlation_cycle_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn correlation_cycle_with_kernel(
    input: &CorrelationCycleInput,
    kernel: Kernel,
) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CorrelationCycleError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(CorrelationCycleError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let threshold = input.get_threshold();
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    let warm_real_imag_angle = first + period; // first index we write for R/I/Angle
    let warm_state = first + period + 1; // first index we write for State

    // Zero-copy friendly allocations with NaN prefixes only
    let mut real = alloc_with_nan_prefix(len, warm_real_imag_angle);
    let mut imag = alloc_with_nan_prefix(len, warm_real_imag_angle);
    let mut angle = alloc_with_nan_prefix(len, warm_real_imag_angle);
    let mut state = alloc_with_nan_prefix(len, warm_state);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => correlation_cycle_compute_into(
                data, period, threshold, first, &mut real, &mut imag, &mut angle, &mut state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => correlation_cycle_avx2(
                data, period, threshold, first, &mut real, &mut imag, &mut angle, &mut state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => correlation_cycle_avx512(
                data, period, threshold, first, &mut real, &mut imag, &mut angle, &mut state,
            ),
            _ => unreachable!(),
        }
    }

    Ok(CorrelationCycleOutput {
        real,
        imag,
        angle,
        state,
    })
}

/// Compute Correlation Cycle into caller-provided buffers (no allocations).
///
/// - Preserves NaN warmups exactly like the Vec-returning API:
///   real/imag/angle warmup at `first_non_nan + period`, state warmup at `first_non_nan + period + 1`.
/// - All output slice lengths must equal the input length.
/// - Uses `Kernel::Auto` dispatch internally.
#[cfg(not(feature = "wasm"))]
pub fn correlation_cycle_into(
    input: &CorrelationCycleInput,
    out_real: &mut [f64],
    out_imag: &mut [f64],
    out_angle: &mut [f64],
    out_state: &mut [f64],
) -> Result<(), CorrelationCycleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CorrelationCycleError::EmptyInputData);
    }

    if out_real.len() != len
        || out_imag.len() != len
        || out_angle.len() != len
        || out_state.len() != len
    {
        let got = *[out_real.len(), out_imag.len(), out_angle.len(), out_state.len()]
            .iter()
            .min()
            .unwrap_or(&0);
        return Err(CorrelationCycleError::OutputLengthMismatch { expected: len, got });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(CorrelationCycleError::InvalidPeriod { period, data_len: len });
    }
    if len - first < period {
        return Err(CorrelationCycleError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let threshold = input.get_threshold();
    let chosen = match Kernel::Auto {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // Prefill warmup prefixes with the crate's quiet-NaN pattern used by alloc_with_nan_prefix
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let warm_ria = (first + period).min(len);
    let warm_s = (first + period + 1).min(len);
    for v in &mut out_real[..warm_ria] {
        *v = qnan;
    }
    for v in &mut out_imag[..warm_ria] {
        *v = qnan;
    }
    for v in &mut out_angle[..warm_ria] {
        *v = qnan;
    }
    for v in &mut out_state[..warm_s] {
        *v = qnan;
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => correlation_cycle_compute_into(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => correlation_cycle_avx2(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => correlation_cycle_avx512(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            _ => correlation_cycle_compute_into(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
        }
    }

    Ok(())
}

#[inline(always)]
pub fn correlation_cycle_into_slices(
    dst_real: &mut [f64],
    dst_imag: &mut [f64],
    dst_angle: &mut [f64],
    dst_state: &mut [f64],
    input: &CorrelationCycleInput,
    kernel: Kernel,
) -> Result<(), CorrelationCycleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CorrelationCycleError::EmptyInputData);
    }
    if dst_real.len() != len
        || dst_imag.len() != len
        || dst_angle.len() != len
        || dst_state.len() != len
    {
        let got = *[dst_real.len(), dst_imag.len(), dst_angle.len(), dst_state.len()]
            .iter()
            .min()
            .unwrap_or(&0);
        return Err(CorrelationCycleError::OutputLengthMismatch { expected: len, got });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(CorrelationCycleError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let threshold = input.get_threshold();
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => correlation_cycle_compute_into(
                data, period, threshold, first, dst_real, dst_imag, dst_angle, dst_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => correlation_cycle_avx2(
                data, period, threshold, first, dst_real, dst_imag, dst_angle, dst_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => correlation_cycle_avx512(
                data, period, threshold, first, dst_real, dst_imag, dst_angle, dst_state,
            ),
            _ => correlation_cycle_compute_into(
                data, period, threshold, first, dst_real, dst_imag, dst_angle, dst_state,
            ),
        }
    }

    // Enforce warmup prefixes without full-array fills
    let warm_ria = first + period;
    let warm_s = first + period + 1;
    for v in &mut dst_real[..warm_ria] {
        *v = f64::NAN;
    }
    for v in &mut dst_imag[..warm_ria] {
        *v = f64::NAN;
    }
    for v in &mut dst_angle[..warm_ria] {
        *v = f64::NAN;
    }
    for v in &mut dst_state[..warm_s] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline]
pub fn correlation_cycle_scalar(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    unsafe {
        correlation_cycle_compute_into(data, period, threshold, first, real, imag, angle, state)
    }
}

#[inline(always)]
unsafe fn correlation_cycle_compute_into(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    // Constants
    let half_pi = f64::asin(1.0); // π/2
    let two_pi = 4.0 * f64::asin(1.0); // 2π

    // Precompute sin/cos tables and their statistics (constant across i)
    let n = period as f64;
    let w = two_pi / n;

    let mut cos_table = vec![0.0f64; period];
    let mut sin_table = vec![0.0f64; period]; // stores -sin

    let mut sum_cos = 0.0f64;
    let mut sum_sin = 0.0f64; // sum of (-sin)
    let mut sum_cos2 = 0.0f64;
    let mut sum_sin2 = 0.0f64; // (-sin)^2

    {
        let mut j = 0usize;
        while j + 4 <= period {
            let a0 = w * ((j as f64) + 1.0);
            let (s0, c0) = a0.sin_cos();
            let ys0 = -s0;
            *cos_table.get_unchecked_mut(j) = c0;
            *sin_table.get_unchecked_mut(j) = ys0;
            sum_cos += c0;
            sum_sin += ys0;
            sum_cos2 += c0 * c0;
            sum_sin2 += ys0 * ys0;

            let a1 = a0 + w;
            let (s1, c1) = a1.sin_cos();
            let ys1 = -s1;
            *cos_table.get_unchecked_mut(j + 1) = c1;
            *sin_table.get_unchecked_mut(j + 1) = ys1;
            sum_cos += c1;
            sum_sin += ys1;
            sum_cos2 += c1 * c1;
            sum_sin2 += ys1 * ys1;

            let a2 = a1 + w;
            let (s2, c2) = a2.sin_cos();
            let ys2 = -s2;
            *cos_table.get_unchecked_mut(j + 2) = c2;
            *sin_table.get_unchecked_mut(j + 2) = ys2;
            sum_cos += c2;
            sum_sin += ys2;
            sum_cos2 += c2 * c2;
            sum_sin2 += ys2 * ys2;

            let a3 = a2 + w;
            let (s3, c3) = a3.sin_cos();
            let ys3 = -s3;
            *cos_table.get_unchecked_mut(j + 3) = c3;
            *sin_table.get_unchecked_mut(j + 3) = ys3;
            sum_cos += c3;
            sum_sin += ys3;
            sum_cos2 += c3 * c3;
            sum_sin2 += ys3 * ys3;

            j += 4;
        }
        while j < period {
            let a = w * ((j as f64) + 1.0);
            let (s, c) = a.sin_cos();
            let ys = -s;
            *cos_table.get_unchecked_mut(j) = c;
            *sin_table.get_unchecked_mut(j) = ys;
            sum_cos += c;
            sum_sin += ys;
            sum_cos2 += c * c;
            sum_sin2 += ys * ys;
            j += 1;
        }
    }

    // Denominator parts constant across i
    let t2_const = n.mul_add(sum_cos2, -(sum_cos * sum_cos));
    let t4_const = n.mul_add(sum_sin2, -(sum_sin * sum_sin));
    let has_t2 = t2_const > 0.0;
    let has_t4 = t4_const > 0.0;
    let sqrt_t2c = if has_t2 { t2_const.sqrt() } else { 0.0 };
    let sqrt_t4c = if has_t4 { t4_const.sqrt() } else { 0.0 };

    // earliest indices we can write
    let start_ria = first + period; // real/imag/angle
    let start_s = start_ria + 1; // state

    // Fuse angle + state computation
    let mut prev_angle = f64::NAN;

    let dptr = data.as_ptr();
    let cptr = cos_table.as_ptr();
    let sptr = sin_table.as_ptr();

    for i in start_ria..data.len() {
        // Accumulators for the i-th window
        let mut sum_x = 0.0f64;
        let mut sum_x2 = 0.0f64;
        let mut sum_xc = 0.0f64; // Σ x * cos
        let mut sum_xs = 0.0f64; // Σ x * (-sin)

        let mut j = 0usize;
        while j + 4 <= period {
            let idx0 = i - (j + 1);
            let idx1 = idx0 - 1;
            let idx2 = idx1 - 1;
            let idx3 = idx2 - 1;

            let mut x0 = *dptr.add(idx0);
            let mut x1 = *dptr.add(idx1);
            let mut x2 = *dptr.add(idx2);
            let mut x3 = *dptr.add(idx3);

            if x0 != x0 {
                x0 = 0.0;
            }
            if x1 != x1 {
                x1 = 0.0;
            }
            if x2 != x2 {
                x2 = 0.0;
            }
            if x3 != x3 {
                x3 = 0.0;
            }

            let c0 = *cptr.add(j);
            let s0 = *sptr.add(j);
            let c1 = *cptr.add(j + 1);
            let s1 = *sptr.add(j + 1);
            let c2 = *cptr.add(j + 2);
            let s2 = *sptr.add(j + 2);
            let c3 = *cptr.add(j + 3);
            let s3 = *sptr.add(j + 3);

            sum_x += x0 + x1 + x2 + x3;
            sum_x2 = x0.mul_add(x0, x1.mul_add(x1, x2.mul_add(x2, x3.mul_add(x3, sum_x2))));
            sum_xc = x0.mul_add(c0, x1.mul_add(c1, x2.mul_add(c2, x3.mul_add(c3, sum_xc))));
            sum_xs = x0.mul_add(s0, x1.mul_add(s1, x2.mul_add(s2, x3.mul_add(s3, sum_xs))));
            j += 4;
        }
        while j < period {
            let idx = i - (j + 1);
            let mut x = *dptr.add(idx);
            if x != x {
                x = 0.0;
            }
            let c = *cptr.add(j);
            let s = *sptr.add(j);
            sum_x += x;
            sum_x2 = x.mul_add(x, sum_x2);
            sum_xc = x.mul_add(c, sum_xc);
            sum_xs = x.mul_add(s, sum_xs);
            j += 1;
        }

        let t1 = n.mul_add(sum_x2, -(sum_x * sum_x));
        let mut r_val = 0.0;
        let mut i_val = 0.0;

        if t1 > 0.0 {
            if has_t2 {
                let denom = t1.sqrt() * sqrt_t2c;
                if denom > 0.0 {
                    r_val = (n.mul_add(sum_xc, -(sum_x * sum_cos))) / denom;
                }
            }
            if has_t4 {
                let denom = t1.sqrt() * sqrt_t4c;
                if denom > 0.0 {
                    i_val = (n.mul_add(sum_xs, -(sum_x * sum_sin))) / denom;
                }
            }
        }

        *real.get_unchecked_mut(i) = r_val;
        *imag.get_unchecked_mut(i) = i_val;

        let a = if i_val == 0.0 {
            0.0
        } else {
            let mut a = (r_val / i_val).atan() + half_pi;
            a = a.to_degrees();
            if i_val > 0.0 {
                a -= 180.0;
            }
            a
        };
        *angle.get_unchecked_mut(i) = a;

        if i >= start_s {
            let prev = prev_angle;
            let st = if !prev.is_nan() && (a - prev).abs() < threshold {
                if a >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            };
            *state.get_unchecked_mut(i) = st;
        }

        prev_angle = a;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx2(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    use core::arch::x86_64::*;

    let half_pi = f64::asin(1.0);
    let two_pi = 4.0 * f64::asin(1.0);
    let n = period as f64;
    let w = two_pi / n;

    let mut cos_table = vec![0.0f64; period];
    let mut sin_table = vec![0.0f64; period];

    let mut sum_cos = 0.0f64;
    let mut sum_sin = 0.0f64;
    let mut sum_cos2 = 0.0f64;
    let mut sum_sin2 = 0.0f64;

    for j in 0..period {
        let a = w * ((j as f64) + 1.0);
        let (s, c) = a.sin_cos();
        let ys = -s;
        *cos_table.get_unchecked_mut(j) = c;
        *sin_table.get_unchecked_mut(j) = ys;
        sum_cos += c;
        sum_sin += ys;
        sum_cos2 += c * c;
        sum_sin2 += ys * ys;
    }

    let t2_const = n.mul_add(sum_cos2, -(sum_cos * sum_cos));
    let t4_const = n.mul_add(sum_sin2, -(sum_sin * sum_sin));
    let has_t2 = t2_const > 0.0;
    let has_t4 = t4_const > 0.0;
    let sqrt_t2c = if has_t2 { t2_const.sqrt() } else { 0.0 };
    let sqrt_t4c = if has_t4 { t4_const.sqrt() } else { 0.0 };

    let start_ria = first + period;
    let start_s = start_ria + 1;

    #[inline(always)]
    fn hsum256(v: __m256d) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd(v, 1);
            let lo = _mm256_castpd256_pd128(v);
            let sum128 = _mm_add_pd(hi, lo);
            let hi64 = _mm_unpackhi_pd(sum128, sum128);
            _mm_cvtsd_f64(_mm_add_sd(sum128, hi64))
        }
    }

    let dptr = data.as_ptr();
    let cptr = cos_table.as_ptr();
    let sptr = sin_table.as_ptr();

    let mut prev_angle = f64::NAN;

    for i in start_ria..data.len() {
        let mut vx = _mm256_setzero_pd();
        let mut vx2 = _mm256_setzero_pd();
        let mut vxc = _mm256_setzero_pd();
        let mut vxs = _mm256_setzero_pd();

        let mut j = 0usize;
        while j + 4 <= period {
            let idx0 = i - (j + 1);
            let x0 = *dptr.add(idx0);
            let x1 = *dptr.add(idx0 - 1);
            let x2 = *dptr.add(idx0 - 2);
            let x3 = *dptr.add(idx0 - 3);
            let mut vx0123 = _mm256_set_pd(x3, x2, x1, x0);

            let ord = _mm256_cmp_pd(vx0123, vx0123, _CMP_ORD_Q);
            vx0123 = _mm256_and_pd(vx0123, ord);

            let vc = _mm256_loadu_pd(cptr.add(j));
            let vs = _mm256_loadu_pd(sptr.add(j));

            vx = _mm256_add_pd(vx, vx0123);
            vx2 = _mm256_fmadd_pd(vx0123, vx0123, vx2);
            vxc = _mm256_fmadd_pd(vx0123, vc, vxc);
            vxs = _mm256_fmadd_pd(vx0123, vs, vxs);

            j += 4;
        }

        let mut sum_x = hsum256(vx);
        let mut sum_x2 = hsum256(vx2);
        let mut sum_xc = hsum256(vxc);
        let mut sum_xs = hsum256(vxs);

        while j < period {
            let idx = i - (j + 1);
            let mut x = *dptr.add(idx);
            if x != x {
                x = 0.0;
            }
            let c = *cptr.add(j);
            let s = *sptr.add(j);
            sum_x += x;
            sum_x2 = x.mul_add(x, sum_x2);
            sum_xc = x.mul_add(c, sum_xc);
            sum_xs = x.mul_add(s, sum_xs);
            j += 1;
        }

        let t1 = n.mul_add(sum_x2, -(sum_x * sum_x));
        let mut r_val = 0.0;
        let mut i_val = 0.0;
        if t1 > 0.0 {
            if has_t2 {
                let denom = t1.sqrt() * sqrt_t2c;
                if denom > 0.0 {
                    r_val = (n.mul_add(sum_xc, -(sum_x * sum_cos))) / denom;
                }
            }
            if has_t4 {
                let denom = t1.sqrt() * sqrt_t4c;
                if denom > 0.0 {
                    i_val = (n.mul_add(sum_xs, -(sum_x * sum_sin))) / denom;
                }
            }
        }

        *real.get_unchecked_mut(i) = r_val;
        *imag.get_unchecked_mut(i) = i_val;

        let a = if i_val == 0.0 {
            0.0
        } else {
            let mut a = (r_val / i_val).atan() + half_pi;
            a = a.to_degrees();
            if i_val > 0.0 {
                a -= 180.0;
            }
            a
        };
        *angle.get_unchecked_mut(i) = a;

        if i >= start_s {
            let prev = prev_angle;
            let st = if !prev.is_nan() && (a - prev).abs() < threshold {
                if a >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            };
            *state.get_unchecked_mut(i) = st;
        }

        prev_angle = a;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx512(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    use core::arch::x86_64::*;

    let half_pi = f64::asin(1.0);
    let two_pi = 4.0 * f64::asin(1.0);
    let n = period as f64;
    let w = two_pi / n;

    let mut cos_table = vec![0.0f64; period];
    let mut sin_table = vec![0.0f64; period];

    let mut sum_cos = 0.0f64;
    let mut sum_sin = 0.0f64;
    let mut sum_cos2 = 0.0f64;
    let mut sum_sin2 = 0.0f64;

    for j in 0..period {
        let a = w * ((j as f64) + 1.0);
        let (s, c) = a.sin_cos();
        let ys = -s;
        *cos_table.get_unchecked_mut(j) = c;
        *sin_table.get_unchecked_mut(j) = ys;
        sum_cos += c;
        sum_sin += ys;
        sum_cos2 += c * c;
        sum_sin2 += ys * ys;
    }

    let t2_const = n.mul_add(sum_cos2, -(sum_cos * sum_cos));
    let t4_const = n.mul_add(sum_sin2, -(sum_sin * sum_sin));
    let has_t2 = t2_const > 0.0;
    let has_t4 = t4_const > 0.0;
    let sqrt_t2c = if has_t2 { t2_const.sqrt() } else { 0.0 };
    let sqrt_t4c = if has_t4 { t4_const.sqrt() } else { 0.0 };

    let start_ria = first + period;
    let start_s = start_ria + 1;

    #[inline(always)]
    fn hsum512(v: __m512d) -> f64 {
        unsafe {
            let lo = _mm512_castpd512_pd256(v);
            let hi = _mm512_extractf64x4_pd(v, 1);
            let lohi = _mm256_add_pd(lo, hi);
            let hi128 = _mm256_extractf128_pd(lohi, 1);
            let lo128 = _mm256_castpd256_pd128(lohi);
            let sum128 = _mm_add_pd(hi128, lo128);
            let hi64 = _mm_unpackhi_pd(sum128, sum128);
            _mm_cvtsd_f64(_mm_add_sd(sum128, hi64))
        }
    }

    let dptr = data.as_ptr();
    let cptr = cos_table.as_ptr();
    let sptr = sin_table.as_ptr();

    let mut prev_angle = f64::NAN;

    for i in start_ria..data.len() {
        let mut vx = _mm512_setzero_pd();
        let mut vx2 = _mm512_setzero_pd();
        let mut vxc = _mm512_setzero_pd();
        let mut vxs = _mm512_setzero_pd();

        let mut j = 0usize;
        while j + 8 <= period {
            let idx0 = i - (j + 1);
            let x0 = *dptr.add(idx0);
            let x1 = *dptr.add(idx0 - 1);
            let x2 = *dptr.add(idx0 - 2);
            let x3 = *dptr.add(idx0 - 3);
            let x4 = *dptr.add(idx0 - 4);
            let x5 = *dptr.add(idx0 - 5);
            let x6 = *dptr.add(idx0 - 6);
            let x7 = *dptr.add(idx0 - 7);

            let mut vx01234567 = _mm512_setr_pd(x0, x1, x2, x3, x4, x5, x6, x7);

            let ordk = _mm512_cmp_pd_mask(vx01234567, vx01234567, _CMP_ORD_Q);
            vx01234567 = _mm512_maskz_mov_pd(ordk, vx01234567);

            let vc = _mm512_loadu_pd(cptr.add(j));
            let vs = _mm512_loadu_pd(sptr.add(j));

            vx = _mm512_add_pd(vx, vx01234567);
            vx2 = _mm512_fmadd_pd(vx01234567, vx01234567, vx2);
            vxc = _mm512_fmadd_pd(vx01234567, vc, vxc);
            vxs = _mm512_fmadd_pd(vx01234567, vs, vxs);

            j += 8;
        }

        let mut sum_x = hsum512(vx);
        let mut sum_x2 = hsum512(vx2);
        let mut sum_xc = hsum512(vxc);
        let mut sum_xs = hsum512(vxs);

        while j < period {
            let idx = i - (j + 1);
            let mut x = *dptr.add(idx);
            if x != x {
                x = 0.0;
            }
            let c = *cptr.add(j);
            let s = *sptr.add(j);
            sum_x += x;
            sum_x2 = x.mul_add(x, sum_x2);
            sum_xc = x.mul_add(c, sum_xc);
            sum_xs = x.mul_add(s, sum_xs);
            j += 1;
        }

        let t1 = n.mul_add(sum_x2, -(sum_x * sum_x));
        let mut r_val = 0.0;
        let mut i_val = 0.0;
        if t1 > 0.0 {
            if has_t2 {
                let denom = t1.sqrt() * sqrt_t2c;
                if denom > 0.0 {
                    r_val = (n.mul_add(sum_xc, -(sum_x * sum_cos))) / denom;
                }
            }
            if has_t4 {
                let denom = t1.sqrt() * sqrt_t4c;
                if denom > 0.0 {
                    i_val = (n.mul_add(sum_xs, -(sum_x * sum_sin))) / denom;
                }
            }
        }

        *real.get_unchecked_mut(i) = r_val;
        *imag.get_unchecked_mut(i) = i_val;

        let a = if i_val == 0.0 {
            0.0
        } else {
            let mut a = (r_val / i_val).atan() + half_pi;
            a = a.to_degrees();
            if i_val > 0.0 {
                a -= 180.0;
            }
            a
        };
        *angle.get_unchecked_mut(i) = a;

        if i >= start_s {
            let prev = prev_angle;
            let st = if !prev.is_nan() && (a - prev).abs() < threshold {
                if a >= 0.0 {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            };
            *state.get_unchecked_mut(i) = st;
        }

        prev_angle = a;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx512_short(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_compute_into(data, period, threshold, first, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx512_long(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_compute_into(data, period, threshold, first, real, imag, angle, state)
}

// Row wrappers
#[inline(always)]
pub unsafe fn correlation_cycle_row_scalar_with_first(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_compute_into(data, period, threshold, first, real, imag, angle, state)
}

// AVX variants can call the same until you write SIMD:
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_row_avx2_with_first(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_avx2(data, period, threshold, first, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_row_avx512_with_first(
    data: &[f64],
    period: usize,
    threshold: f64,
    first: usize,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_avx512(data, period, threshold, first, real, imag, angle, state)
}

/// Streaming kernel note: Uses an O(1) sliding-DFT phasor update (no per-tick window scan).
/// Matches batch semantics and one-tick alignment; NaNs are treated as zeros inside the window.
#[derive(Debug, Clone)]
pub struct CorrelationCycleStream {
    // Params
    period: usize,
    threshold: f64,

    // Circular buffer of sanitized samples (NaN -> 0.0)
    buffer: Vec<f64>,
    head: usize,
    filled: bool,

    /// Previously computed tuple to emit on the next call (one-tick delay to match batch)
    last: Option<(f64, f64, f64, f64)>,

    // --- O(1) state ---

    // Rolling sums for normalization
    sum_x: f64,
    sum_x2: f64,

    // Complex phasor accumulator for the k=1 bin:
    // P = sum x_i * [cos(w*(m+1)) + i * (-sin(w*(m+1)))]
    // We store its real/imag parts directly:
    phasor_re: f64, // Σ x * cos
    phasor_im: f64, // Σ x * (-sin)

    // Angle memory for state
    prev_angle: f64,

    // Precomputed constants
    n: f64,
    half_pi: f64,
    z_re: f64,     // cos(w)
    z_im: f64,     // -sin(w)  (negative!)
    sum_cos: f64,  // Σ cos(w*(j+1)), j=0..N-1
    sum_sin: f64,  // Σ (-sin(w*(j+1))), j=0..N-1
    sqrt_t2c: f64, // sqrt( N*Σcos^2 - (Σcos)^2 )
    sqrt_t4c: f64, // sqrt( N*Σ(-sin)^2 - (Σ(-sin))^2 )
    has_t2: bool,
    has_t4: bool,
}

impl CorrelationCycleStream {
    pub fn try_new(params: CorrelationCycleParams) -> Result<Self, CorrelationCycleError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(CorrelationCycleError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let threshold = params.threshold.unwrap_or(9.0);

        // Match batch path constants exactly (asin-based pi to minimize drift)
        let half_pi = f64::asin(1.0); // π/2
        let two_pi = 4.0 * half_pi; // 2π
        let n = period as f64;
        let w = two_pi / n;

        // z = e^{-jw} = cos(w) + i*(-sin(w))
        let (s_w, c_w) = w.sin_cos();
        let z_re = c_w;
        let z_im = -s_w; // keep sign consistent with (-sin) convention

        // Precompute Σcos, Σ(-sin), and their squares exactly like the batch kernel
        let mut sum_cos = 0.0f64;
        let mut sum_sin = 0.0f64; // sum of (-sin)
        let mut sum_cos2 = 0.0f64;
        let mut sum_sin2 = 0.0f64; // (-sin)^2

        let mut j = 0usize;
        while j + 4 <= period {
            let a0 = w * ((j as f64) + 1.0);
            let (s0, c0) = a0.sin_cos();
            let ys0 = -s0;

            let a1 = a0 + w;
            let (s1, c1) = a1.sin_cos();
            let ys1 = -s1;

            let a2 = a1 + w;
            let (s2, c2) = a2.sin_cos();
            let ys2 = -s2;

            let a3 = a2 + w;
            let (s3, c3) = a3.sin_cos();
            let ys3 = -s3;

            sum_cos += c0 + c1 + c2 + c3;
            sum_sin += ys0 + ys1 + ys2 + ys3;
            sum_cos2 += c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3;
            sum_sin2 += ys0 * ys0 + ys1 * ys1 + ys2 * ys2 + ys3 * ys3;

            j += 4;
        }
        while j < period {
            let a = w * ((j as f64) + 1.0);
            let (s, c) = a.sin_cos();
            let ys = -s;
            sum_cos += c;
            sum_sin += ys;
            sum_cos2 += c * c;
            sum_sin2 += ys * ys;
            j += 1;
        }

        // Denominator constants (once per stream)
        let t2_const = n.mul_add(sum_cos2, -(sum_cos * sum_cos));
        let t4_const = n.mul_add(sum_sin2, -(sum_sin * sum_sin));
        let has_t2 = t2_const > 0.0;
        let has_t4 = t4_const > 0.0;
        let sqrt_t2c = if has_t2 { t2_const.sqrt() } else { 0.0 };
        let sqrt_t4c = if has_t4 { t4_const.sqrt() } else { 0.0 };

        Ok(Self {
            period,
            threshold,
            buffer: vec![0.0; period], // sanitized samples (NaN -> 0)
            head: 0,
            filled: false,
            last: None,

            sum_x: 0.0,
            sum_x2: 0.0,
            phasor_re: 0.0,
            phasor_im: 0.0,
            prev_angle: f64::NAN,

            n,
            half_pi,
            z_re,
            z_im,
            sum_cos,
            sum_sin,
            sqrt_t2c,
            sqrt_t4c,
            has_t2,
            has_t4,
        })
    }

    /// Per-tick O(1) update. Returns `None` until the first full window is formed.
    /// After that, returns the previous window’s (real, imag, angle, state) each call
    /// to match the batch alignment (one-tick latency).
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
        // 1) Ingest new sample (sanitize NaN -> 0.0), get outgoing sample from the ring
        let x_new = if value.is_nan() { 0.0 } else { value };
        let x_old = self.buffer[self.head]; // already sanitized
        self.buffer[self.head] = x_new;
        self.head = (self.head + 1) % self.period;

        // 2) O(1) roll the normalization sums
        self.sum_x += x_new - x_old;
        // fused form for (x_new^2 - x_old^2)
        self.sum_x2 = (x_new * x_new) - (x_old * x_old) + self.sum_x2;

        // 3) O(1) phasor update via sliding DFT at k=1:
        //    P <- z * (P + (x_new - x_old))
        //    Using two FMAs: let s = Re(P) + (x_new - x_old)
        let dx = x_new - x_old;
        let s = self.phasor_re + dx;

        let new_re = self.z_re.mul_add(s, -self.z_im * self.phasor_im);
        let new_im = self.z_im.mul_add(s, self.z_re * self.phasor_im);

        self.phasor_re = new_re;
        self.phasor_im = new_im;

        // Warmup: on the first wrap we compute the first full-window value,
        // stash it in `last`, and return None (to keep one-tick delay).
        let first_wrap_now = !self.filled && self.head == 0;
        if first_wrap_now {
            self.filled = true;
        } else if !self.filled {
            return None;
        }

        // 4) Compute correlation-coefficient components. For numerical parity with batch,
        //    recompute exact rolling sums from the ring for normalization (and sum_x in numerators).
        let mut sum_x_exact = 0.0f64;
        let mut sum_x2_exact = 0.0f64;
        let mut k = 0usize;
        while k + 4 <= self.period {
            let idx0 = (self.head + k) % self.period;
            let idx1 = (self.head + k + 1) % self.period;
            let idx2 = (self.head + k + 2) % self.period;
            let idx3 = (self.head + k + 3) % self.period;
            let x0 = self.buffer[idx0];
            let x1 = self.buffer[idx1];
            let x2 = self.buffer[idx2];
            let x3 = self.buffer[idx3];
            sum_x_exact += x0 + x1 + x2 + x3;
            sum_x2_exact = x0.mul_add(x0, sum_x2_exact);
            sum_x2_exact = x1.mul_add(x1, sum_x2_exact);
            sum_x2_exact = x2.mul_add(x2, sum_x2_exact);
            sum_x2_exact = x3.mul_add(x3, sum_x2_exact);
            k += 4;
        }
        while k < self.period {
            let idx = (self.head + k) % self.period;
            let x = self.buffer[idx];
            sum_x_exact += x;
            sum_x2_exact = x.mul_add(x, sum_x2_exact);
            k += 1;
        }

        let t1 = self.n.mul_add(sum_x2_exact, -(sum_x_exact * sum_x_exact));

        let mut r_val = 0.0;
        let mut i_val = 0.0;

        if t1 > 0.0 {
            let sqrt_t1 = t1.sqrt();
            if self.has_t2 {
                let denom_r = sqrt_t1 * self.sqrt_t2c;
                if denom_r > 0.0 {
                    r_val = (self
                        .n
                        .mul_add(self.phasor_re, -(sum_x_exact * self.sum_cos)))
                        / denom_r;
                }
            }
            if self.has_t4 {
                let denom_i = sqrt_t1 * self.sqrt_t4c;
                if denom_i > 0.0 {
                    i_val = (self
                        .n
                        .mul_add(self.phasor_im, -(sum_x_exact * self.sum_sin)))
                        / denom_i;
                }
            }
        }

        // Angle & state (exactly as in scalar path)
        let mut ang = if i_val == 0.0 {
            0.0
        } else {
            let mut a = (r_val / i_val).atan() + self.half_pi;
            a = a.to_degrees();
            if i_val > 0.0 {
                a -= 180.0;
            }
            a
        };

        // State warmup behavior: first full window emits NaN (matches batch warmup +1)
        let st = if self.prev_angle.is_finite() && (ang - self.prev_angle).abs() < self.threshold {
            if ang >= 0.0 {
                1.0
            } else {
                -1.0
            }
        } else if self.prev_angle.is_finite() {
            0.0
        } else {
            f64::NAN
        };

        // Keep prev angle for next tick's state decision
        self.prev_angle = ang;

        // 5) Respect one-tick delay: emit the previously computed tuple,
        //    then stash the current result as `last`.
        let to_emit = self.last.take();
        self.last = Some((r_val, i_val, ang, st));

        if first_wrap_now {
            None
        } else {
            to_emit
        }
    }
}

#[derive(Clone, Debug)]
pub struct CorrelationCycleBatchRange {
    pub period: (usize, usize, usize),
    pub threshold: (f64, f64, f64),
}

impl Default for CorrelationCycleBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 100, 1),
            threshold: (9.0, 9.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CorrelationCycleBatchBuilder {
    range: CorrelationCycleBatchRange,
    kernel: Kernel,
}

impl CorrelationCycleBatchBuilder {
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
    pub fn threshold_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.threshold = (start, end, step);
        self
    }
    #[inline]
    pub fn threshold_static(mut self, x: f64) -> Self {
        self.range.threshold = (x, x, 0.0);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        correlation_cycle_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        CorrelationCycleBatchBuilder::new()
            .kernel(k)
            .apply_slice(data)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        CorrelationCycleBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[inline(always)]
pub fn correlation_cycle_batch_with_kernel(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    k: Kernel,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(CorrelationCycleError::InvalidKernelForBatch(k)),
    };

    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar, // Fallback to scalar for safety
    };
    correlation_cycle_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CorrelationCycleBatchOutput {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub angle: Vec<f64>,
    pub state: Vec<f64>,
    pub combos: Vec<CorrelationCycleParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CorrelationCycleBatchOutput {
    pub fn row_for_params(&self, p: &CorrelationCycleParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.threshold.unwrap_or(9.0) - p.threshold.unwrap_or(9.0)).abs() < 1e-12
        })
    }
    pub fn values_for(
        &self,
        p: &CorrelationCycleParams,
    ) -> Option<(&[f64], &[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.real[start..start + self.cols],
                &self.imag[start..start + self.cols],
                &self.angle[start..start + self.cols],
                &self.state[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CorrelationCycleBatchRange) -> Result<Vec<CorrelationCycleParams>, CorrelationCycleError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CorrelationCycleError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start < end {
            let mut v = start;
            loop {
                vals.push(v);
                if v >= end { break; }
                let next = match v.checked_add(step) { Some(n) => n, None => break };
                if next == v { break; }
                v = next;
            }
        } else {
            let mut v = start;
            loop {
                vals.push(v);
                if v <= end { break; }
                let next = v.saturating_sub(step);
                if next == v { break; }
                v = next;
            }
        }
        if vals.is_empty() {
            return Err(CorrelationCycleError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CorrelationCycleError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start <= end {
            let mut x = start;
            loop {
                vals.push(x);
                if x >= end { break; }
                let next = x + step;
                if !next.is_finite() || next == x { break; }
                x = next;
                if x > end + 1e-12 { break; }
            }
        } else {
            let mut x = start;
            loop {
                vals.push(x);
                if x <= end { break; }
                let next = x - step.abs();
                if !next.is_finite() || next == x { break; }
                x = next;
                if x < end - 1e-12 { break; }
            }
        }
        if vals.is_empty() {
            return Err(CorrelationCycleError::InvalidRange { start: start as usize, end: end as usize, step: step.abs() as usize });
        }
        Ok(vals)
    }

    let periods = axis_usize(r.period)?;
    let thresholds = axis_f64(r.threshold)?;

    let cap = periods
        .len()
        .checked_mul(thresholds.len())
        .ok_or_else(|| CorrelationCycleError::InvalidInput("rows*cols overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &t in &thresholds {
            out.push(CorrelationCycleParams { period: Some(p), threshold: Some(t) });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn correlation_cycle_batch_slice(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    correlation_cycle_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn correlation_cycle_batch_par_slice(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    correlation_cycle_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn correlation_cycle_batch_inner(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    if data.is_empty() {
        return Err(CorrelationCycleError::EmptyInputData);
    }
    let combos = expand_grid(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let _total = rows
        .checked_mul(cols)
        .ok_or_else(|| CorrelationCycleError::InvalidInput("rows*cols overflow".into()))?;

    // Step 1: Allocate uninitialized matrices for each output
    let mut real_mu = make_uninit_matrix(rows, cols);
    let mut imag_mu = make_uninit_matrix(rows, cols);
    let mut angle_mu = make_uninit_matrix(rows, cols);
    let mut state_mu = make_uninit_matrix(rows, cols);

    // Step 2: Calculate warmup periods for each row (each parameter combination)
    let ria_warm: Vec<usize> = combos
        .iter()
        .map(|c| first.checked_add(c.period.unwrap()).unwrap_or(usize::MAX))
        .collect();
    let st_warm: Vec<usize> = combos
        .iter()
        .map(|c| first.checked_add(c.period.unwrap()).and_then(|v| v.checked_add(1)).unwrap_or(usize::MAX))
        .collect();

    // Step 3: Initialize NaN prefixes for each row
    init_matrix_prefixes(&mut real_mu, cols, &ria_warm);
    init_matrix_prefixes(&mut imag_mu, cols, &ria_warm);
    init_matrix_prefixes(&mut angle_mu, cols, &ria_warm);
    init_matrix_prefixes(&mut state_mu, cols, &st_warm);

    // Step 4: Convert to mutable slices for computation
    let mut real_guard = ManuallyDrop::new(real_mu);
    let mut imag_guard = ManuallyDrop::new(imag_mu);
    let mut angle_guard = ManuallyDrop::new(angle_mu);
    let mut state_guard = ManuallyDrop::new(state_mu);

    let real: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(real_guard.as_mut_ptr() as *mut f64, real_guard.len())
    };
    let imag: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(imag_guard.as_mut_ptr() as *mut f64, imag_guard.len())
    };
    let angle: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(angle_guard.as_mut_ptr() as *mut f64, angle_guard.len())
    };
    let state: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(state_guard.as_mut_ptr() as *mut f64, state_guard.len())
    };

    let do_row = |row: usize,
                  out_real: &mut [f64],
                  out_imag: &mut [f64],
                  out_angle: &mut [f64],
                  out_state: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let threshold = combos[row].threshold.unwrap();
        match kern {
            Kernel::Scalar => correlation_cycle_row_scalar_with_first(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => correlation_cycle_row_avx2_with_first(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => correlation_cycle_row_avx512_with_first(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
            _ => correlation_cycle_row_scalar_with_first(
                data, period, threshold, first, out_real, out_imag, out_angle, out_state,
            ),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            real.par_chunks_mut(cols)
                .zip(imag.par_chunks_mut(cols))
                .zip(angle.par_chunks_mut(cols))
                .zip(state.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (((r, im), an), st))| do_row(row, r, im, an, st));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((r, im), an), st)) in real
                .chunks_mut(cols)
                .zip(imag.chunks_mut(cols))
                .zip(angle.chunks_mut(cols))
                .zip(state.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, r, im, an, st);
            }
        }
    } else {
        for (row, (((r, im), an), st)) in real
            .chunks_mut(cols)
            .zip(imag.chunks_mut(cols))
            .zip(angle.chunks_mut(cols))
            .zip(state.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, r, im, an, st);
        }
    }

    // Step 5: Reclaim as Vec<f64>
    let real = unsafe {
        Vec::from_raw_parts(
            real_guard.as_mut_ptr() as *mut f64,
            real_guard.len(),
            real_guard.capacity(),
        )
    };
    let imag = unsafe {
        Vec::from_raw_parts(
            imag_guard.as_mut_ptr() as *mut f64,
            imag_guard.len(),
            imag_guard.capacity(),
        )
    };
    let angle = unsafe {
        Vec::from_raw_parts(
            angle_guard.as_mut_ptr() as *mut f64,
            angle_guard.len(),
            angle_guard.capacity(),
        )
    };
    let state = unsafe {
        Vec::from_raw_parts(
            state_guard.as_mut_ptr() as *mut f64,
            state_guard.len(),
            state_guard.capacity(),
        )
    };

    Ok(CorrelationCycleBatchOutput {
        real,
        imag,
        angle,
        state,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn correlation_cycle_batch_inner_into(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
    parallel: bool,
    out_real: &mut [f64],
    out_imag: &mut [f64],
    out_angle: &mut [f64],
    out_state: &mut [f64],
) -> Result<Vec<CorrelationCycleParams>, CorrelationCycleError> {
    if data.is_empty() {
        return Err(CorrelationCycleError::EmptyInputData);
    }
    let combos = expand_grid(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| CorrelationCycleError::InvalidInput("rows*cols overflow".into()))?;
    if out_real.len() != expected
        || out_imag.len() != expected
        || out_angle.len() != expected
        || out_state.len() != expected
    {
        let got = *[out_real.len(), out_imag.len(), out_angle.len(), out_state.len()]
            .iter()
            .min()
            .unwrap_or(&0);
        return Err(CorrelationCycleError::OutputLengthMismatch { expected, got });
    }

    let do_row = |row: usize, r: &mut [f64], im: &mut [f64], an: &mut [f64], st: &mut [f64]| unsafe {
        let p = combos[row].period.unwrap();
        let t = combos[row].threshold.unwrap();
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                correlation_cycle_row_scalar_with_first(data, p, t, first, r, im, an, st)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                correlation_cycle_row_avx2_with_first(data, p, t, first, r, im, an, st)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                correlation_cycle_row_avx512_with_first(data, p, t, first, r, im, an, st)
            }
            _ => correlation_cycle_row_scalar_with_first(data, p, t, first, r, im, an, st),
        }
        // Warmup prefixes (no full fills)
        let ria = first + p;
        let stp = first + p + 1;
        for v in &mut r[..ria] {
            *v = f64::NAN;
        }
        for v in &mut im[..ria] {
            *v = f64::NAN;
        }
        for v in &mut an[..ria] {
            *v = f64::NAN;
        }
        for v in &mut st[..stp] {
            *v = f64::NAN;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_real
                .par_chunks_mut(cols)
                .zip(out_imag.par_chunks_mut(cols))
                .zip(out_angle.par_chunks_mut(cols))
                .zip(out_state.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (((r, im), an), st))| do_row(row, r, im, an, st));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((r, im), an), st)) in out_real
                .chunks_mut(cols)
                .zip(out_imag.chunks_mut(cols))
                .zip(out_angle.chunks_mut(cols))
                .zip(out_state.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, r, im, an, st);
            }
        }
    } else {
        for (row, (((r, im), an), st)) in out_real
            .chunks_mut(cols)
            .zip(out_imag.chunks_mut(cols))
            .zip(out_angle.chunks_mut(cols))
            .zip(out_state.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, r, im, an, st);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_cc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = CorrelationCycleParams {
            period: None,
            threshold: None,
        };
        let input = CorrelationCycleInput::from_candles(&candles, "close", default_params);
        let output = correlation_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.real.len(), candles.close.len());
        Ok(())
    }

    fn check_cc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CorrelationCycleParams {
            period: Some(20),
            threshold: Some(9.0),
        };
        let input = CorrelationCycleInput::from_candles(&candles, "close", params);
        let result = correlation_cycle_with_kernel(&input, kernel)?;
        let expected_last_five_real = [
            -0.3348928030992766,
            -0.2908979303392832,
            -0.10648582811938148,
            -0.09118320471750277,
            0.0826798259258665,
        ];
        let expected_last_five_imag = [
            0.2902308064575494,
            0.4025192756952553,
            0.4704322460080054,
            0.5404405595224989,
            0.5418162415918566,
        ];
        let expected_last_five_angle = [
            -139.0865569687123,
            -125.8553823569915,
            -102.75438860700636,
            -99.576759208278,
            -81.32373697835556,
        ];
        let start = result.real.len().saturating_sub(5);
        for i in 0..5 {
            let diff_real = (result.real[start + i] - expected_last_five_real[i]).abs();
            let diff_imag = (result.imag[start + i] - expected_last_five_imag[i]).abs();
            let diff_angle = (result.angle[start + i] - expected_last_five_angle[i]).abs();
            assert!(
                diff_real < 1e-8,
                "[{}] CC {:?} real mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.real[start + i],
                expected_last_five_real[i]
            );
            assert!(
                diff_imag < 1e-8,
                "[{}] CC {:?} imag mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.imag[start + i],
                expected_last_five_imag[i]
            );
            assert!(
                diff_angle < 1e-8,
                "[{}] CC {:?} angle mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                result.angle[start + i],
                expected_last_five_angle[i]
            );
        }
        Ok(())
    }

    fn check_cc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CorrelationCycleInput::with_default_candles(&candles);
        match input.data {
            CorrelationCycleData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CorrelationCycleData::Candles"),
        }
        let output = correlation_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.real.len(), candles.close.len());
        Ok(())
    }

    fn check_cc_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CorrelationCycleParams {
            period: Some(0),
            threshold: None,
        };
        let input = CorrelationCycleInput::from_slice(&input_data, params);
        let res = correlation_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CC should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cc_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CorrelationCycleParams {
            period: Some(10),
            threshold: None,
        };
        let input = CorrelationCycleInput::from_slice(&data_small, params);
        let res = correlation_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CC should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cc_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CorrelationCycleParams {
            period: Some(9),
            threshold: None,
        };
        let input = CorrelationCycleInput::from_slice(&single_point, params);
        let res = correlation_cycle_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CC should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let params = CorrelationCycleParams {
            period: Some(4),
            threshold: Some(2.0),
        };
        let input = CorrelationCycleInput::from_slice(&data, params.clone());
        let first_result = correlation_cycle_with_kernel(&input, kernel)?;
        let second_input = CorrelationCycleInput::from_slice(&first_result.real, params);
        let second_result = correlation_cycle_with_kernel(&second_input, kernel)?;
        assert_eq!(first_result.real.len(), data.len());
        assert_eq!(second_result.real.len(), data.len());
        Ok(())
    }

    fn check_cc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CorrelationCycleInput::from_candles(
            &candles,
            "close",
            CorrelationCycleParams {
                period: Some(20),
                threshold: None,
            },
        );
        let res = correlation_cycle_with_kernel(&input, kernel)?;
        assert_eq!(res.real.len(), candles.close.len());
        if res.real.len() > 40 {
            for (i, &val) in res.real[40..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    40 + i
                );
            }
        }
        Ok(())
    }

    fn check_cc_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 20;
        let threshold = 9.0;
        let input = CorrelationCycleInput::from_candles(
            &candles,
            "close",
            CorrelationCycleParams {
                period: Some(period),
                threshold: Some(threshold),
            },
        );
        let batch_output = correlation_cycle_with_kernel(&input, kernel)?;
        let mut stream = CorrelationCycleStream::try_new(CorrelationCycleParams {
            period: Some(period),
            threshold: Some(threshold),
        })?;
        let mut stream_real = Vec::with_capacity(candles.close.len());
        let mut stream_imag = Vec::with_capacity(candles.close.len());
        let mut stream_angle = Vec::with_capacity(candles.close.len());
        let mut stream_state = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some((r, im, ang, st)) => {
                    stream_real.push(r);
                    stream_imag.push(im);
                    stream_angle.push(ang);
                    stream_state.push(st);
                }
                None => {
                    stream_real.push(f64::NAN);
                    stream_imag.push(f64::NAN);
                    stream_angle.push(f64::NAN);
                    stream_state.push(0.0);
                }
            }
        }
        assert_eq!(batch_output.real.len(), stream_real.len());
        for (i, (&b, &s)) in batch_output.real.iter().zip(stream_real.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] CC streaming real f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_cc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase coverage
        let test_params = vec![
            CorrelationCycleParams {
                period: Some(20),
                threshold: Some(9.0),
            },
            CorrelationCycleParams {
                period: Some(10),
                threshold: Some(5.0),
            },
            CorrelationCycleParams {
                period: Some(30),
                threshold: Some(15.0),
            },
            CorrelationCycleParams {
                period: None,
                threshold: None,
            }, // default params
        ];

        for params in test_params {
            let input = CorrelationCycleInput::from_candles(&candles, "close", params.clone());
            let output = correlation_cycle_with_kernel(&input, kernel)?;

            // Check every value in all output arrays for poison patterns
            let arrays = vec![
                ("real", &output.real),
                ("imag", &output.imag),
                ("angle", &output.angle),
                ("state", &output.state),
            ];

            for (array_name, values) in arrays {
                for (i, &val) in values.iter().enumerate() {
                    // Skip NaN values as they're expected in the warmup period
                    if val.is_nan() {
                        continue;
                    }

                    let bits = val.to_bits();

                    // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                    if bits == 0x11111111_11111111 {
                        panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in {} array with params {:?}",
                            test_name, val, bits, i, array_name, params
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in {} array with params {:?}",
                            test_name, val, bits, i, array_name, params
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in {} array with params {:?}",
                            test_name, val, bits, i, array_name, params
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_cc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_cc_tests {
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

    generate_all_cc_tests!(
        check_cc_partial_params,
        check_cc_accuracy,
        check_cc_default_candles,
        check_cc_zero_period,
        check_cc_period_exceeds_length,
        check_cc_very_small_dataset,
        check_cc_reinput,
        check_cc_nan_handling,
        check_cc_streaming,
        check_cc_no_poison
    );

    #[cfg(not(feature = "wasm"))]
    #[test]
fn test_correlation_cycle_into_matches_api_v2() -> Result<(), Box<dyn Error>> {
        // Use the same CSV dataset as the module's other tests
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Build input with default params
        let input = CorrelationCycleInput::from_candles(
            &candles,
            "close",
            CorrelationCycleParams::default(),
        );

        // Baseline via existing Vec-returning API
        let baseline = correlation_cycle(&input)?;

        // Preallocate outputs and call new into API
        let len = candles.close.len();
        let mut out_real = vec![0.0f64; len];
        let mut out_imag = vec![0.0f64; len];
        let mut out_angle = vec![0.0f64; len];
        let mut out_state = vec![0.0f64; len];

        correlation_cycle_into(
            &input,
            &mut out_real,
            &mut out_imag,
            &mut out_angle,
            &mut out_state,
        )?;

        // Length parity
        assert_eq!(baseline.real.len(), out_real.len());
        assert_eq!(baseline.imag.len(), out_imag.len());
        assert_eq!(baseline.angle.len(), out_angle.len());
        assert_eq!(baseline.state.len(), out_state.len());

        // Value parity (treat NaN == NaN; exact for finite, epsilon fallback)
        fn eq_or_both_nan_eps(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan_eps(baseline.real[i], out_real[i]),
                "real mismatch at {}: base={}, into={}",
                i,
                baseline.real[i],
                out_real[i]
            );
            assert!(
                eq_or_both_nan_eps(baseline.imag[i], out_imag[i]),
                "imag mismatch at {}: base={}, into={}",
                i,
                baseline.imag[i],
                out_imag[i]
            );
            assert!(
                eq_or_both_nan_eps(baseline.angle[i], out_angle[i]),
                "angle mismatch at {}: base={}, into={}",
                i,
                baseline.angle[i],
                out_angle[i]
            );
            assert!(
                eq_or_both_nan_eps(baseline.state[i], out_state[i]),
                "state mismatch at {}: base={}, into={}",
                i,
                baseline.state[i],
                out_state[i]
            );
        }

        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_correlation_cycle_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic price data
        let strat = (5usize..=50) // period range
            .prop_flat_map(|period| {
                (
                    prop::collection::vec(
                        (10.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                        period + 10..400, // Ensure we have enough data for warmup + testing
                    ),
                    Just(period),
                    1.0f64..20.0f64, // threshold range
                )
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, threshold)| {
                let params = CorrelationCycleParams {
                    period: Some(period),
                    threshold: Some(threshold),
                };
                let input = CorrelationCycleInput::from_slice(&data, params);

                // Test with the specified kernel
                let output = correlation_cycle_with_kernel(&input, kernel).unwrap();

                // Test with scalar reference for comparison
                let ref_output = correlation_cycle_with_kernel(&input, Kernel::Scalar).unwrap();

                // Calculate warmup period (first non-NaN should be at index period)
                let warmup_period = period;

                // Property 1: Check warmup period - values before period should be NaN
                for i in 0..warmup_period {
                    prop_assert!(
                        output.real[i].is_nan(),
                        "[{}] real[{}] should be NaN during warmup but is {}",
                        test_name,
                        i,
                        output.real[i]
                    );
                    prop_assert!(
                        output.imag[i].is_nan(),
                        "[{}] imag[{}] should be NaN during warmup but is {}",
                        test_name,
                        i,
                        output.imag[i]
                    );
                    prop_assert!(
                        output.angle[i].is_nan(),
                        "[{}] angle[{}] should be NaN during warmup but is {}",
                        test_name,
                        i,
                        output.angle[i]
                    );
                }

                // State warmup is period + 1
                for i in 0..=period {
                    prop_assert!(
                        output.state[i].is_nan(),
                        "[{}] state[{}] should be NaN during warmup but is {}",
                        test_name,
                        i,
                        output.state[i]
                    );
                }

                // Property 2: Check bounds for real and imag (correlation coefficients should be in [-1, 1])
                for i in warmup_period..data.len() {
                    if !output.real[i].is_nan() {
                        prop_assert!(
                            output.real[i] >= -1.0 - 1e-9 && output.real[i] <= 1.0 + 1e-9,
                            "[{}] real[{}] = {} is outside [-1, 1] bounds",
                            test_name,
                            i,
                            output.real[i]
                        );
                    }
                    if !output.imag[i].is_nan() {
                        prop_assert!(
                            output.imag[i] >= -1.0 - 1e-9 && output.imag[i] <= 1.0 + 1e-9,
                            "[{}] imag[{}] = {} is outside [-1, 1] bounds",
                            test_name,
                            i,
                            output.imag[i]
                        );
                    }
                }

                // Property 3: Check angle bounds (should be in degrees between -180 and 180)
                for i in warmup_period..data.len() {
                    if !output.angle[i].is_nan() {
                        prop_assert!(
                            output.angle[i] >= -180.0 - 1e-9 && output.angle[i] <= 180.0 + 1e-9,
                            "[{}] angle[{}] = {} is outside [-180, 180] bounds",
                            test_name,
                            i,
                            output.angle[i]
                        );
                    }
                }

                // Property 4: Check state values (should be exactly -1, 0, or 1)
                for i in (period + 1)..data.len() {
                    if !output.state[i].is_nan() {
                        let state_val = output.state[i];
                        prop_assert!(
                            (state_val + 1.0).abs() < 1e-9
                                || state_val.abs() < 1e-9
                                || (state_val - 1.0).abs() < 1e-9,
                            "[{}] state[{}] = {} is not -1, 0, or 1",
                            test_name,
                            i,
                            state_val
                        );
                    }
                }

                // Property 5: Check kernel consistency
                for i in 0..data.len() {
                    // Compare real values
                    let real_bits = output.real[i].to_bits();
                    let ref_real_bits = ref_output.real[i].to_bits();

                    if !output.real[i].is_finite() || !ref_output.real[i].is_finite() {
                        prop_assert!(
                            real_bits == ref_real_bits,
                            "[{}] real finite/NaN mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.real[i],
                            ref_output.real[i]
                        );
                    } else {
                        let ulp_diff = real_bits.abs_diff(ref_real_bits);
                        prop_assert!(
                            (output.real[i] - ref_output.real[i]).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] real mismatch at idx {}: {} vs {} (ULP={})",
                            test_name,
                            i,
                            output.real[i],
                            ref_output.real[i],
                            ulp_diff
                        );
                    }

                    // Compare imag values
                    let imag_bits = output.imag[i].to_bits();
                    let ref_imag_bits = ref_output.imag[i].to_bits();

                    if !output.imag[i].is_finite() || !ref_output.imag[i].is_finite() {
                        prop_assert!(
                            imag_bits == ref_imag_bits,
                            "[{}] imag finite/NaN mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.imag[i],
                            ref_output.imag[i]
                        );
                    } else {
                        let ulp_diff = imag_bits.abs_diff(ref_imag_bits);
                        prop_assert!(
                            (output.imag[i] - ref_output.imag[i]).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] imag mismatch at idx {}: {} vs {} (ULP={})",
                            test_name,
                            i,
                            output.imag[i],
                            ref_output.imag[i],
                            ulp_diff
                        );
                    }

                    // Compare angle values
                    let angle_bits = output.angle[i].to_bits();
                    let ref_angle_bits = ref_output.angle[i].to_bits();

                    if !output.angle[i].is_finite() || !ref_output.angle[i].is_finite() {
                        prop_assert!(
                            angle_bits == ref_angle_bits,
                            "[{}] angle finite/NaN mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.angle[i],
                            ref_output.angle[i]
                        );
                    } else {
                        let ulp_diff = angle_bits.abs_diff(ref_angle_bits);
                        prop_assert!(
                            (output.angle[i] - ref_output.angle[i]).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] angle mismatch at idx {}: {} vs {} (ULP={})",
                            test_name,
                            i,
                            output.angle[i],
                            ref_output.angle[i],
                            ulp_diff
                        );
                    }

                    // Compare state values
                    let state_bits = output.state[i].to_bits();
                    let ref_state_bits = ref_output.state[i].to_bits();

                    if !output.state[i].is_finite() || !ref_output.state[i].is_finite() {
                        prop_assert!(
                            state_bits == ref_state_bits,
                            "[{}] state finite/NaN mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.state[i],
                            ref_output.state[i]
                        );
                    } else {
                        prop_assert!(
                            (output.state[i] - ref_output.state[i]).abs() <= 1e-9,
                            "[{}] state mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            output.state[i],
                            ref_output.state[i]
                        );
                    }
                }

                // Property 6: Test specific mathematical properties of correlation cycle
                // When all data is constant, correlation should be near 0 or NaN
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                    for i in warmup_period..data.len() {
                        // For constant data, correlation is undefined (0/0) so could be NaN or 0
                        if !output.real[i].is_nan() {
                            prop_assert!(
                                output.real[i].abs() < 1e-6,
                                "[{}] real[{}] = {} should be near 0 for constant data",
                                test_name,
                                i,
                                output.real[i]
                            );
                        }
                        if !output.imag[i].is_nan() {
                            prop_assert!(
                                output.imag[i].abs() < 1e-6,
                                "[{}] imag[{}] = {} should be near 0 for constant data",
                                test_name,
                                i,
                                output.imag[i]
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_cc_tests!(check_correlation_cycle_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = CorrelationCycleBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = CorrelationCycleParams::default();
        let (row_real, row_imag, row_angle, row_state) =
            output.values_for(&def).expect("default row missing");
        assert_eq!(row_real.len(), c.close.len());
        assert_eq!(row_imag.len(), c.close.len());
        assert_eq!(row_angle.len(), c.close.len());
        assert_eq!(row_state.len(), c.close.len());
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations
        let output = CorrelationCycleBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 40, 10) // Test periods: 10, 20, 30, 40
            .threshold_range(5.0, 15.0, 5.0) // Test thresholds: 5.0, 10.0, 15.0
            .apply_candles(&c, "close")?;

        // Check every value in all batch matrices for poison patterns
        let matrices = vec![
            ("real", &output.real),
            ("imag", &output.imag),
            ("angle", &output.angle),
            ("state", &output.state),
        ];

        for (matrix_name, values) in matrices {
            for (idx, &val) in values.iter().enumerate() {
                // Skip NaN values as they're expected in warmup periods
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let period = output.combos[row].period.unwrap();
                let threshold = output.combos[row].threshold.unwrap();

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in {} matrix, params: period={}, threshold={}",
                        test, val, bits, row, col, idx, matrix_name, period, threshold
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) in {} matrix, params: period={}, threshold={}",
                        test, val, bits, row, col, idx, matrix_name, period, threshold
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) in {} matrix, params: period={}, threshold={}",
                        test, val, bits, row, col, idx, matrix_name, period, threshold
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

    #[test]
    fn test_correlation_cycle_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let x = 100.0 + (i as f64 * 0.07).sin() * 2.5 + (i as f64 * 0.011).cos() * 0.4;
            data.push(x);
        }

        let input = CorrelationCycleInput::from_slice(&data, CorrelationCycleParams::default());

        let base = correlation_cycle(&input)?;

        let mut out_r = vec![0.0; n];
        let mut out_i = vec![0.0; n];
        let mut out_a = vec![0.0; n];
        let mut out_s = vec![0.0; n];

        #[cfg(not(feature = "wasm"))]
        {
            correlation_cycle_into(&input, &mut out_r, &mut out_i, &mut out_a, &mut out_s)?;
        }
        #[cfg(feature = "wasm")]
        {
            return Ok(());
        }

        assert_eq!(base.real.len(), n);
        assert_eq!(base.imag.len(), n);
        assert_eq!(base.angle.len(), n);
        assert_eq!(base.state.len(), n);

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..n {
            assert!(eq_or_both_nan(base.real[i], out_r[i]), "real mismatch at {}: base={}, into={}", i, base.real[i], out_r[i]);
            assert!(eq_or_both_nan(base.imag[i], out_i[i]), "imag mismatch at {}: base={}, into={}", i, base.imag[i], out_i[i]);
            assert!(eq_or_both_nan(base.angle[i], out_a[i]), "angle mismatch at {}: base={}, into={}", i, base.angle[i], out_a[i]);
            assert!(eq_or_both_nan(base.state[i], out_s[i]), "state mismatch at {}: base={}, into={}", i, base.state[i], out_s[i]);
        }

        Ok(())
    }
}

// ==================== PYTHON: CUDA BINDINGS (zero-copy) ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32CorrelationCycle", unsendable)]
pub struct DeviceArrayF32CcPy {
    pub(crate) inner: crate::cuda::moving_averages::DeviceArrayF32,
    // Keep context alive for the lifetime of the handle; actual DLPack
    // export uses the shared helper in `dlpack_cuda`.
    pub(crate) _ctx: StdArc<CudaContext>,
    pub(crate) device_id: i32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32CcPy {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "use correlation_cycle_cuda_* factory functions to create this type",
        ))
    }

    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let inner = &self.inner;
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr_val: usize = if inner.rows == 0 || inner.cols == 0 {
            0
        } else {
            inner.buf.as_device_ptr().as_raw() as usize
        };
        d.set_item("data", (ptr_val, false))?;
        // Stream omitted: producing kernels synchronize before returning.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        Ok((2, self.device_id))
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<i64>,
        max_version: Option<(i32, i32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<pyo3::PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
        use pyo3::IntoPy;

        // Validate dl_device hint against the allocation device.
        let (kdl, alloc_dev) = self.__dlpack_device__()?;
        if let Some((dev_ty, dev_id)) = dl_device {
            if dev_ty != kdl || dev_id != alloc_dev {
                let wants_copy = copy.unwrap_or(false);
                if wants_copy {
                    return Err(PyValueError::new_err(
                        "device copy not implemented for __dlpack__",
                    ));
                } else {
                    return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                }
            }
        }

        // Producer synchronizes its CUDA stream before returning; reject
        // explicitly disallowed stream==0 per Array API semantics.
        if let Some(s) = stream {
            if s == 0 {
                return Err(PyValueError::new_err(
                    "stream=0 is reserved and not supported by this producer",
                ));
            }
        }

        // Move VRAM handle out; subsequent exports get an empty buffer.
        let dummy = cust::memory::DeviceBuffer::<f32>::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            crate::cuda::moving_averages::DeviceArrayF32 { buf: dummy, rows: 0, cols: 0 },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        // Re-wrap max_version tuple as a Python object for the shared helper.
        let max_version_bound =
            max_version.map(|(maj, min)| (maj, min).into_py(py).into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl DeviceArrayF32CcPy {
    pub fn new_from_rust(
        inner: crate::cuda::moving_averages::DeviceArrayF32,
        ctx: StdArc<CudaContext>,
        device_id: u32,
    ) -> Self {
        Self {
            inner,
            _ctx: ctx,
            device_id: device_id as i32,
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "correlation_cycle_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, threshold_range, device_id=0))]
pub fn correlation_cycle_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    threshold_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<(
    DeviceArrayF32CcPy,
    DeviceArrayF32CcPy,
    DeviceArrayF32CcPy,
    DeviceArrayF32CcPy,
)> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaCorrelationCycle;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = CorrelationCycleBatchRange {
        period: period_range,
        threshold: threshold_range,
    };
    let (quad, ctx, dev_id) = py.allow_threads(|| {
        let mut cuda = CudaCorrelationCycle::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.ctx();
        let dev_id = cuda.device_id();
        let quad = cuda.correlation_cycle_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((quad, ctx, dev_id))
    })?;
    Ok((
        DeviceArrayF32CcPy::new_from_rust(quad.real, ctx.clone(), dev_id),
        DeviceArrayF32CcPy::new_from_rust(quad.imag, ctx.clone(), dev_id),
        DeviceArrayF32CcPy::new_from_rust(quad.angle, ctx.clone(), dev_id),
        DeviceArrayF32CcPy::new_from_rust(quad.state, ctx, dev_id),
    ))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "correlation_cycle_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, threshold, device_id=0))]
pub fn correlation_cycle_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    threshold: f64,
    device_id: usize,
) -> PyResult<(
    DeviceArrayF32CcPy,
    DeviceArrayF32CcPy,
    DeviceArrayF32CcPy,
    DeviceArrayF32CcPy,
)> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaCorrelationCycle;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2D array"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if flat.len() != expected {
        return Err(PyValueError::new_err("time-major input length mismatch"));
    }
    let params = CorrelationCycleParams {
        period: Some(period),
        threshold: Some(threshold),
    };
    let (quad, ctx, dev_id) = py.allow_threads(|| {
        let mut cuda = CudaCorrelationCycle::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.ctx();
        let dev_id = cuda.device_id();
        let quad = cuda.correlation_cycle_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((quad, ctx, dev_id))
    })?;
    Ok((
        DeviceArrayF32CcPy::new_from_rust(quad.real, ctx.clone(), dev_id),
        DeviceArrayF32CcPy::new_from_rust(quad.imag, ctx.clone(), dev_id),
        DeviceArrayF32CcPy::new_from_rust(quad.angle, ctx.clone(), dev_id),
        DeviceArrayF32CcPy::new_from_rust(quad.state, ctx, dev_id),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "correlation_cycle")]
#[pyo3(signature = (data, period=None, threshold=None, kernel=None))]
pub fn correlation_cycle_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: Option<usize>,
    threshold: Option<f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArrayMethods;

    let data_slice = data.as_slice()?;
    let kern = match kernel {
        Some(k) => crate::utilities::kernel_validation::validate_kernel(Some(k), false)?,
        None => Kernel::Auto,
    };

    let params = CorrelationCycleParams { period, threshold };
    let input = CorrelationCycleInput::from_slice(data_slice, params);

    // Compute without GIL for better performance
    let output = py
        .allow_threads(|| correlation_cycle_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Create output dictionary with zero-copy transfers
    let dict = PyDict::new(py);
    dict.set_item("real", output.real.into_pyarray(py))?;
    dict.set_item("imag", output.imag.into_pyarray(py))?;
    dict.set_item("angle", output.angle.into_pyarray(py))?;
    dict.set_item("state", output.state.into_pyarray(py))?;

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "correlation_cycle_batch")]
#[pyo3(signature = (data, period_range=None, threshold_range=None, kernel=None))]
pub fn correlation_cycle_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: Option<(usize, usize, usize)>,
    threshold_range: Option<(f64, f64, f64)>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyArrayMethods;

    let slice_in = data.as_slice()?;

    let sweep = CorrelationCycleBatchRange {
        period: period_range.unwrap_or((20, 100, 1)),
        threshold: threshold_range.unwrap_or((9.0, 9.0, 0.0)),
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    // Preallocate flat outputs
    let out_real = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_imag = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_angle = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_state = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let mut_r = unsafe { out_real.as_slice_mut()? };
    let mut_im = unsafe { out_imag.as_slice_mut()? };
    let mut_an = unsafe { out_angle.as_slice_mut()? };
    let mut_st = unsafe { out_state.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let simd = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let row_k = match simd {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch => Kernel::Avx512,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        };
        correlation_cycle_batch_inner_into(
            slice_in, &sweep, row_k, true, mut_r, mut_im, mut_an, mut_st,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("real", out_real.reshape((rows, cols))?)?;
    dict.set_item("imag", out_imag.reshape((rows, cols))?)?;
    dict.set_item("angle", out_angle.reshape((rows, cols))?)?;
    dict.set_item("state", out_state.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "thresholds",
        combos
            .iter()
            .map(|p| p.threshold.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "CorrelationCycleStream")]
pub struct CorrelationCycleStreamPy {
    inner: CorrelationCycleStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CorrelationCycleStreamPy {
    #[new]
    #[pyo3(signature = (period=None, threshold=None))]
    pub fn new(period: Option<usize>, threshold: Option<f64>) -> PyResult<Self> {
        let params = CorrelationCycleParams { period, threshold };
        let inner = CorrelationCycleStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
        self.inner.update(value)
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CorrelationCycleJsOutput {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub angle: Vec<f64>,
    pub state: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn correlation_cycle_js(
    data: &[f64],
    period: Option<usize>,
    threshold: Option<f64>,
) -> Result<JsValue, JsValue> {
    let params = CorrelationCycleParams { period, threshold };
    let input = CorrelationCycleInput::from_slice(data, params);

    let output = correlation_cycle(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = CorrelationCycleJsOutput {
        real: output.real,
        imag: output.imag,
        angle: output.angle,
        state: output.state,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CorrelationCycleBatchJsOutput {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub angle: Vec<f64>,
    pub state: Vec<f64>,
    pub combos: Vec<CorrelationCycleParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn correlation_cycle_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    threshold_start: f64,
    threshold_end: f64,
    threshold_step: f64,
) -> Result<JsValue, JsValue> {
    let sweep = CorrelationCycleBatchRange {
        period: (period_start, period_end, period_step),
        threshold: (threshold_start, threshold_end, threshold_step),
    };

    let output = correlation_cycle_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = CorrelationCycleBatchJsOutput {
        real: output.real,
        imag: output.imag,
        angle: output.angle,
        state: output.state,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn correlation_cycle_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn correlation_cycle_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn correlation_cycle_into(
    in_ptr: *const f64,
    real_ptr: *mut f64,
    imag_ptr: *mut f64,
    angle_ptr: *mut f64,
    state_ptr: *mut f64,
    len: usize,
    period: Option<usize>,
    threshold: Option<f64>,
) -> Result<(), JsValue> {
    if in_ptr.is_null()
        || real_ptr.is_null()
        || imag_ptr.is_null()
        || angle_ptr.is_null()
        || state_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = CorrelationCycleParams { period, threshold };
        let input = CorrelationCycleInput::from_slice(data, params);

        // Check for aliasing - if ANY output pointer matches input, we need temp buffers
        let has_aliasing = in_ptr == real_ptr as *const f64
            || in_ptr == imag_ptr as *const f64
            || in_ptr == angle_ptr as *const f64
            || in_ptr == state_ptr as *const f64;

        if has_aliasing {
            // Use temporary buffers for all outputs
            let mut temp_real = vec![0.0; len];
            let mut temp_imag = vec![0.0; len];
            let mut temp_angle = vec![0.0; len];
            let mut temp_state = vec![0.0; len];

            correlation_cycle_into_slices(
                &mut temp_real,
                &mut temp_imag,
                &mut temp_angle,
                &mut temp_state,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results to output pointers
            let real_out = std::slice::from_raw_parts_mut(real_ptr, len);
            let imag_out = std::slice::from_raw_parts_mut(imag_ptr, len);
            let angle_out = std::slice::from_raw_parts_mut(angle_ptr, len);
            let state_out = std::slice::from_raw_parts_mut(state_ptr, len);

            real_out.copy_from_slice(&temp_real);
            imag_out.copy_from_slice(&temp_imag);
            angle_out.copy_from_slice(&temp_angle);
            state_out.copy_from_slice(&temp_state);
        } else {
            // Direct computation into output buffers
            let real_out = std::slice::from_raw_parts_mut(real_ptr, len);
            let imag_out = std::slice::from_raw_parts_mut(imag_ptr, len);
            let angle_out = std::slice::from_raw_parts_mut(angle_ptr, len);
            let state_out = std::slice::from_raw_parts_mut(state_ptr, len);

            correlation_cycle_into_slices(
                real_out,
                imag_out,
                angle_out,
                state_out,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}
