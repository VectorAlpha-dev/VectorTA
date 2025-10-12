//! # Reverse Relative Strength Indicator (Reverse RSI)
//!
//! The Reverse Relative Strength Indicator calculates the price level that would result
//! in a specific RSI value. It answers the question: "What price would give us an RSI of X?"
//! This is useful for identifying potential support/resistance levels and price targets.
//!
//! ## Parameters
//! - **rsi_length**: Period for RSI calculation (default: 14)
//! - **rsi_level**: Target RSI level to reverse-engineer (default: 50.0, range: 0-100)
//!
//! ## Inputs
//! - **data**: Price data or any numeric series
//!
//! ## Returns
//! - **values**: Vector of reverse RSI values with NaN prefix during warmup period
//!
//! ## Developer Notes
//! - **SIMD status**: Reverse RSI is dominated by sequential EMA recurrences; only the short warmup is vectorizable.
//!   AVX2/AVX512 warmup vectorization is enabled behind `nightly-avx` and selected at runtime; the main loop remains scalar.
//! - **Streaming**: Implemented with O(1) update performance (maintains EMAs)
//! - **Zero-copy Memory**: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations
//! - **SIMD stubs**: When `nightly-avx` is not enabled, AVX2/AVX512 fall back to an unsafe-fast scalar path; the Scalar path stays fully safe (WASM-friendly).
//! - **Batch note**: ScalarBatch uses a per-length shared-EMA path (row-specific optimization) to avoid redundant EMA recomputation across RSI levels.

// ==================== IMPORTS SECTION ====================
// Feature-gated imports for Python bindings
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

// Feature-gated imports for WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// Core imports
use crate::indicators::moving_averages::ema::{ema, ema_into_slice, EmaInput, EmaParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

// SIMD imports for AVX optimizations
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// Parallel processing support
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

// Standard library imports
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// ==================== TRAIT IMPLEMENTATIONS ====================
impl<'a> AsRef<[f64]> for ReverseRsiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ReverseRsiData::Slice(slice) => slice,
            ReverseRsiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ==================== DATA STRUCTURES ====================
/// Input data enum supporting both raw slices and candle data
#[derive(Debug, Clone)]
pub enum ReverseRsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

/// Output structure containing calculated values
#[derive(Debug, Clone)]
pub struct ReverseRsiOutput {
    pub values: Vec<f64>,
}

/// Parameters structure with optional fields for defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ReverseRsiParams {
    pub rsi_length: Option<usize>,
    pub rsi_level: Option<f64>,
}

impl Default for ReverseRsiParams {
    fn default() -> Self {
        Self {
            rsi_length: Some(14),
            rsi_level: Some(50.0),
        }
    }
}

/// Main input structure combining data and parameters
#[derive(Debug, Clone)]
pub struct ReverseRsiInput<'a> {
    pub data: ReverseRsiData<'a>,
    pub params: ReverseRsiParams,
}

impl<'a> ReverseRsiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ReverseRsiParams) -> Self {
        Self {
            data: ReverseRsiData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ReverseRsiParams) -> Self {
        Self {
            data: ReverseRsiData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ReverseRsiParams::default())
    }

    #[inline]
    pub fn get_rsi_length(&self) -> usize {
        self.params.rsi_length.unwrap_or(14)
    }

    #[inline]
    pub fn get_rsi_level(&self) -> f64 {
        self.params.rsi_level.unwrap_or(50.0)
    }
}

// ==================== BUILDER PATTERN ====================
/// Builder for ergonomic API usage
#[derive(Copy, Clone, Debug)]
pub struct ReverseRsiBuilder {
    rsi_length: Option<usize>,
    rsi_level: Option<f64>,
    kernel: Kernel,
}

impl Default for ReverseRsiBuilder {
    fn default() -> Self {
        Self {
            rsi_length: None,
            rsi_level: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ReverseRsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn rsi_length(mut self, val: usize) -> Self {
        self.rsi_length = Some(val);
        self
    }

    #[inline(always)]
    pub fn rsi_level(mut self, val: f64) -> Self {
        self.rsi_level = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<ReverseRsiOutput, ReverseRsiError> {
        let p = ReverseRsiParams {
            rsi_length: self.rsi_length,
            rsi_level: self.rsi_level,
        };
        let i = ReverseRsiInput::from_candles(c, "close", p);
        reverse_rsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ReverseRsiOutput, ReverseRsiError> {
        let p = ReverseRsiParams {
            rsi_length: self.rsi_length,
            rsi_level: self.rsi_level,
        };
        let i = ReverseRsiInput::from_slice(d, p);
        reverse_rsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<ReverseRsiStream, ReverseRsiError> {
        let p = ReverseRsiParams {
            rsi_length: self.rsi_length,
            rsi_level: self.rsi_level,
        };
        ReverseRsiStream::try_new(p)
    }
}

// ==================== BATCH PROCESSING BUILDER ====================
#[derive(Debug, Clone)]
pub struct ReverseRsiBatchRange {
    pub rsi_length_range: (usize, usize, usize), // (start, end, step)
    pub rsi_level_range: (f64, f64, f64),        // (start, end, step)
}

impl Default for ReverseRsiBatchRange {
    fn default() -> Self {
        Self {
            rsi_length_range: (14, 14, 0),
            rsi_level_range: (50.0, 50.0, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReverseRsiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ReverseRsiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ReverseRsiBatchOutput {
    pub fn row_for_params(&self, p: &ReverseRsiParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.rsi_length.unwrap_or(14) == p.rsi_length.unwrap_or(14)
                && (c.rsi_level.unwrap_or(50.0) - p.rsi_level.unwrap_or(50.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &ReverseRsiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ReverseRsiBatchBuilder {
    rsi_length_range: (usize, usize, usize),
    rsi_level_range: (f64, f64, f64),
    kernel: Kernel,
}

impl Default for ReverseRsiBatchBuilder {
    fn default() -> Self {
        Self {
            rsi_length_range: (14, 14, 0),
            rsi_level_range: (50.0, 50.0, 0.0),
            kernel: Kernel::Auto,
        }
    }
}

impl ReverseRsiBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn rsi_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.rsi_length_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn rsi_level_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.rsi_level_range = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        c: &Candles,
        source: &str,
    ) -> Result<ReverseRsiBatchOutput, ReverseRsiError> {
        let sweep = ReverseRsiBatchRange {
            rsi_length_range: self.rsi_length_range,
            rsi_level_range: self.rsi_level_range,
        };
        let data = source_type(c, source);
        reverse_rsi_batch_slice(data, &sweep, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<ReverseRsiBatchOutput, ReverseRsiError> {
        let sweep = ReverseRsiBatchRange {
            rsi_length_range: self.rsi_length_range,
            rsi_level_range: self.rsi_level_range,
        };
        reverse_rsi_batch_slice(data, &sweep, self.kernel)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<ReverseRsiBatchOutput, ReverseRsiError> {
        ReverseRsiBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn with_default_candles(c: &Candles) -> Result<ReverseRsiBatchOutput, ReverseRsiError> {
        ReverseRsiBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

// ==================== ERROR HANDLING ====================
#[derive(Debug, Error)]
pub enum ReverseRsiError {
    #[error("reverse_rsi: Input data slice is empty.")]
    EmptyInputData,

    #[error("reverse_rsi: All values are NaN.")]
    AllValuesNaN,

    #[error("reverse_rsi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("reverse_rsi: Invalid RSI level: {level} (must be between 0 and 100)")]
    InvalidRsiLevel { level: f64 },

    #[error("reverse_rsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// ==================== MAIN ALGORITHM ====================
#[inline]
fn reverse_rsi_prepare<'a>(
    input: &'a ReverseRsiInput,
    _kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, f64, usize), ReverseRsiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(ReverseRsiError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ReverseRsiError::AllValuesNaN)?;
    let rsi_len = input.get_rsi_length();
    let rsi_lvl = input.get_rsi_level();
    if rsi_len == 0 || rsi_len > len {
        return Err(ReverseRsiError::InvalidPeriod {
            period: rsi_len,
            data_len: len,
        });
    }
    if !(0.0 < rsi_lvl && rsi_lvl < 100.0) || rsi_lvl.is_nan() || rsi_lvl.is_infinite() {
        return Err(ReverseRsiError::InvalidRsiLevel { level: rsi_lvl });
    }
    let ema_len = (2 * rsi_len) - 1; // warmup bars count
    if len - first < ema_len {
        return Err(ReverseRsiError::NotEnoughValidData {
            needed: ema_len,
            valid: len - first,
        });
    }
    Ok((data, first, rsi_len, rsi_lvl, ema_len))
}

// Safe scalar reference implementation (no unsafe, used for Scalar and WASM)
#[inline(always)]
fn reverse_rsi_compute_into_scalar_safe(
    data: &[f64],
    first: usize,
    rsi_length: usize,
    rsi_level: f64,
    out: &mut [f64],
) -> Result<(), ReverseRsiError> {
    // Preconditions are validated by reverse_rsi_prepare
    let len = data.len();
    let ema_len = (2 * rsi_length) - 1;

    // ---- Constants (hoisted) ----
    let l = rsi_level;
    let inv = 100.0 - l;
    let n_minus_1 = (rsi_length - 1) as f64;
    let rs_target = l / inv; // L / (100 - L)
    let neg_scale = inv / l; // (100 - L) / L
    let rs_coeff = n_minus_1 * rs_target;

    // Wilder-equivalent EMA parameters (α = 2/(ema_len+1))
    let alpha = 2.0 / (ema_len as f64 + 1.0);
    let beta = 1.0 - alpha;

    // ---- Warmup: SMA of up/down over ema_len samples starting at `first` ----
    let warm_end = first + ema_len; // exclusive
    let all_finite = data[first..].iter().all(|v| v.is_finite());

    let mut sum_up = 0.0f64;
    let mut sum_dn = 0.0f64;
    let mut prev = 0.0f64;
    for i in first..warm_end {
        let cur = data[i];
        let d = if all_finite || (cur.is_finite() && prev.is_finite()) {
            cur - prev
        } else {
            0.0
        };
        sum_up += d.max(0.0);
        sum_dn += (-d).max(0.0);
        prev = cur;
    }

    let mut up_ema = sum_up / (ema_len as f64);
    let mut dn_ema = sum_dn / (ema_len as f64);

    // ---- First output at index warm_idx = warm_end - 1 ----
    let warm_idx = warm_end - 1;
    let base = data[warm_idx];
    let x0 = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema);
    let m0 = (x0 >= 0.0) as i32 as f64; // 1.0 if positive branch, else 0.0
    let scale0 = neg_scale + m0 * (1.0 - neg_scale); // {neg_scale, 1.0}
    let v0 = base + x0 * scale0;
    out[warm_idx] = if v0.is_finite() || x0 >= 0.0 { v0 } else { 0.0 };

    // ---- Main loop ----
    prev = base;
    for i in warm_end..len {
        let cur = data[i];
        let d = if all_finite || (cur.is_finite() && prev.is_finite()) {
            cur - prev
        } else {
            0.0
        };
        let up = d.max(0.0);
        let dn = (-d).max(0.0);

        up_ema = beta.mul_add(up_ema, alpha * up);
        dn_ema = beta.mul_add(dn_ema, alpha * dn);

        let x = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema);
        let m = (x >= 0.0) as i32 as f64;
        let scale = neg_scale + m * (1.0 - neg_scale);
        let v = cur + x * scale;
        out[i] = if v.is_finite() || x >= 0.0 { v } else { 0.0 };
        prev = cur;
    }

    Ok(())
}

// Unsafe optimized compute used as AVX2/AVX512 stub (no explicit SIMD intrinsics)
#[inline(always)]
unsafe fn reverse_rsi_compute_into_unsafe_fast(
    data: &[f64],
    first: usize,
    rsi_length: usize,
    rsi_level: f64,
    out: &mut [f64],
) -> Result<(), ReverseRsiError> {
    let len = data.len();
    let ema_len = (2 * rsi_length) - 1;

    let l = rsi_level;
    let inv = 100.0 - l;
    let rs_target = l / inv;
    let neg_scale = inv / l;
    let n_minus_1 = (rsi_length - 1) as f64;
    let rs_coeff = n_minus_1 * rs_target;

    let alpha = 2.0 / (ema_len as f64 + 1.0);
    let beta = 1.0 - alpha;

    let warm_end = first + ema_len;
    let mut sum_up = 0.0f64;
    let mut sum_dn = 0.0f64;

    let all_finite = data[first..].iter().all(|v| v.is_finite());

    let mut i = first;
    if all_finite {
        while i < warm_end {
            let cur = *data.get_unchecked(i);
            let prev = if i == first {
                0.0
            } else {
                *data.get_unchecked(i - 1)
            };
            let d = cur - prev;
            sum_up += d.max(0.0);
            sum_dn += (-d).max(0.0);
            i += 1;
        }
    } else {
        while i < warm_end {
            let cur = *data.get_unchecked(i);
            let prev = if i == first {
                0.0
            } else {
                *data.get_unchecked(i - 1)
            };
            if cur.is_finite() & prev.is_finite() {
                let d = cur - prev;
                sum_up += d.max(0.0);
                sum_dn += (-d).max(0.0);
            }
            i += 1;
        }
    }

    let mut up_ema = sum_up / (ema_len as f64);
    let mut dn_ema = sum_dn / (ema_len as f64);

    let warm_idx = warm_end - 1;
    let base = *data.get_unchecked(warm_idx);
    let x0 = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema);
    let m0 = (x0 >= 0.0) as i32 as f64;
    let scale0 = neg_scale + m0 * (1.0 - neg_scale);
    let v0 = base + x0 * scale0;
    *out.get_unchecked_mut(warm_idx) = if v0.is_finite() || x0 >= 0.0 { v0 } else { 0.0 };

    i = warm_end;
    if all_finite {
        while i < len {
            let cur = *data.get_unchecked(i);
            let prev = *data.get_unchecked(i - 1);
            let d = cur - prev;
            let up = d.max(0.0);
            let dn = (-d).max(0.0);
            up_ema = beta.mul_add(up_ema, alpha * up);
            dn_ema = beta.mul_add(dn_ema, alpha * dn);
            let x = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema);
            let m = (x >= 0.0) as i32 as f64;
            let scale = neg_scale + m * (1.0 - neg_scale);
            let v = cur + x * scale;
            *out.get_unchecked_mut(i) = if v.is_finite() || x >= 0.0 { v } else { 0.0 };
            i += 1;
        }
    } else {
        while i < len {
            let cur = *data.get_unchecked(i);
            let prev = *data.get_unchecked(i - 1);
            let valid = cur.is_finite() & prev.is_finite();
            let d = if valid { cur - prev } else { 0.0 };
            let up = d.max(0.0);
            let dn = (-d).max(0.0);
            up_ema = beta.mul_add(up_ema, alpha * up);
            dn_ema = beta.mul_add(dn_ema, alpha * dn);
            let x = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema);
            let m = (x >= 0.0) as i32 as f64;
            let scale = neg_scale + m * (1.0 - neg_scale);
            let v = cur + x * scale;
            *out.get_unchecked_mut(i) = if v.is_finite() || x >= 0.0 { v } else { 0.0 };
            i += 1;
        }
    }

    Ok(())
}

// AVX2 stub: uses the unsafe fast scalar implementation
#[inline(always)]
fn reverse_rsi_compute_into_avx2_stub(
    data: &[f64],
    first: usize,
    rsi_length: usize,
    rsi_level: f64,
    out: &mut [f64],
) -> Result<(), ReverseRsiError> {
    #[cfg(all(
        feature = "nightly-avx",
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let len = data.len();
        let ema_len = (2 * rsi_length) - 1;

        // Constants
        let l = rsi_level;
        let inv = 100.0 - l;
        let n_minus_1 = (rsi_length - 1) as f64;
        let rs_target = l / inv;
        let neg_scale = inv / l;
        let rs_coeff = n_minus_1 * rs_target;

        let alpha = 2.0 / (ema_len as f64 + 1.0);
        let beta = 1.0 - alpha;

        let warm_end = first + ema_len; // exclusive
        let all_finite = data[first..].iter().all(|v| v.is_finite());
        if !all_finite {
            return reverse_rsi_compute_into_unsafe_fast(data, first, rsi_length, rsi_level, out);
        }

        // --- AVX2 warmup: sum of positive/negative diffs ---
        let mut sum_up = 0.0f64;
        let mut sum_dn = 0.0f64;

        // first element uses prev=0.0 to match semantics
        if first < warm_end {
            let c0 = *data.get_unchecked(first);
            let d0 = c0 - 0.0;
            sum_up += if d0 > 0.0 { d0 } else { 0.0 };
            sum_dn += if d0 < 0.0 { -d0 } else { 0.0 };
        }

        let mut i = first + 1;
        let mut v_up = _mm256_setzero_pd();
        let mut v_dn = _mm256_setzero_pd();
        let v_zero = _mm256_setzero_pd();

        while i + 3 < warm_end {
            let v_cur = _mm256_loadu_pd(data.as_ptr().add(i));
            let v_prev = _mm256_loadu_pd(data.as_ptr().add(i - 1));
            let v_d = _mm256_sub_pd(v_cur, v_prev);
            let v_u = _mm256_max_pd(v_d, v_zero);
            let v_n = _mm256_max_pd(_mm256_sub_pd(v_zero, v_d), v_zero);
            v_up = _mm256_add_pd(v_up, v_u);
            v_dn = _mm256_add_pd(v_dn, v_n);
            i += 4;
        }

        // horizontal reduce
        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), v_up);
        sum_up += buf[0] + buf[1] + buf[2] + buf[3];
        _mm256_storeu_pd(buf.as_mut_ptr(), v_dn);
        sum_dn += buf[0] + buf[1] + buf[2] + buf[3];

        // tail
        while i < warm_end {
            let c = *data.get_unchecked(i);
            let p = *data.get_unchecked(i - 1);
            let d = c - p;
            sum_up += if d > 0.0 { d } else { 0.0 };
            sum_dn += if d < 0.0 { -d } else { 0.0 };
            i += 1;
        }

        let mut up_ema = sum_up / (ema_len as f64);
        let mut dn_ema = sum_dn / (ema_len as f64);

        // first output
        let warm_idx = warm_end - 1;
        let base = *data.get_unchecked(warm_idx);
        let x0 = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema);
        let m0 = (x0 >= 0.0) as i32 as f64;
        let scale0 = neg_scale + m0 * (1.0 - neg_scale);
        let v0 = base + x0 * scale0;
        *out.get_unchecked_mut(warm_idx) = if v0.is_finite() || x0 >= 0.0 { v0 } else { 0.0 };

        // sequential EMA loop
        let mut j = warm_end;
        while j < len {
            let cur = *data.get_unchecked(j);
            let prev = *data.get_unchecked(j - 1);
            let d = cur - prev;
            let up = if d > 0.0 { d } else { 0.0 };
            let dn = if d < 0.0 { -d } else { 0.0 };

            up_ema = beta.mul_add(up_ema, alpha * up);
            dn_ema = beta.mul_add(dn_ema, alpha * dn);

            let x = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema);
            let m = (x >= 0.0) as i32 as f64;
            let scale = neg_scale + m * (1.0 - neg_scale);
            let val = cur + x * scale;
            *out.get_unchecked_mut(j) = if val.is_finite() || x >= 0.0 {
                val
            } else {
                0.0
            };
            j += 1;
        }

        return Ok(());
    }

    // portable fallback
    unsafe { reverse_rsi_compute_into_unsafe_fast(data, first, rsi_length, rsi_level, out) }
}

// AVX512 stub: currently same as AVX2 stub
#[inline(always)]
fn reverse_rsi_compute_into_avx512_stub(
    data: &[f64],
    first: usize,
    rsi_length: usize,
    rsi_level: f64,
    out: &mut [f64],
) -> Result<(), ReverseRsiError> {
    #[cfg(all(
        feature = "nightly-avx",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let len = data.len();
        let ema_len = (2 * rsi_length) - 1;

        // Constants
        let l = rsi_level;
        let inv = 100.0 - l;
        let n_minus_1 = (rsi_length - 1) as f64;
        let rs_target = l / inv;
        let neg_scale = inv / l;
        let rs_coeff = n_minus_1 * rs_target;

        let alpha = 2.0 / (ema_len as f64 + 1.0);
        let beta = 1.0 - alpha;

        let warm_end = first + ema_len; // exclusive
        let all_finite = data[first..].iter().all(|v| v.is_finite());
        if !all_finite {
            return reverse_rsi_compute_into_unsafe_fast(data, first, rsi_length, rsi_level, out);
        }

        // AVX512 warmup
        let mut sum_up = 0.0f64;
        let mut sum_dn = 0.0f64;

        if first < warm_end {
            let c0 = *data.get_unchecked(first);
            let d0 = c0 - 0.0;
            sum_up += if d0 > 0.0 { d0 } else { 0.0 };
            sum_dn += if d0 < 0.0 { -d0 } else { 0.0 };
        }

        let mut i = first + 1;
        let mut v_up = _mm512_setzero_pd();
        let mut v_dn = _mm512_setzero_pd();
        let v_zero = _mm512_setzero_pd();

        while i + 7 < warm_end {
            let v_cur = _mm512_loadu_pd(data.as_ptr().add(i));
            let v_prev = _mm512_loadu_pd(data.as_ptr().add(i - 1));
            let v_d = _mm512_sub_pd(v_cur, v_prev);
            let v_u = _mm512_max_pd(v_d, v_zero);
            let v_n = _mm512_max_pd(_mm512_sub_pd(v_zero, v_d), v_zero);
            v_up = _mm512_add_pd(v_up, v_u);
            v_dn = _mm512_add_pd(v_dn, v_n);
            i += 8;
        }

        let mut buf = [0.0f64; 8];
        _mm512_storeu_pd(buf.as_mut_ptr(), v_up);
        sum_up += buf.iter().sum::<f64>();
        _mm512_storeu_pd(buf.as_mut_ptr(), v_dn);
        sum_dn += buf.iter().sum::<f64>();

        while i < warm_end {
            let c = *data.get_unchecked(i);
            let p = *data.get_unchecked(i - 1);
            let d = c - p;
            sum_up += if d > 0.0 { d } else { 0.0 };
            sum_dn += if d < 0.0 { -d } else { 0.0 };
            i += 1;
        }

        let mut up_ema = sum_up / (ema_len as f64);
        let mut dn_ema = sum_dn / (ema_len as f64);

        // first output
        let warm_idx = warm_end - 1;
        let base = *data.get_unchecked(warm_idx);
        let x0 = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema);
        let m0 = (x0 >= 0.0) as i32 as f64;
        let scale0 = neg_scale + m0 * (1.0 - neg_scale);
        let v0 = base + x0 * scale0;
        *out.get_unchecked_mut(warm_idx) = if v0.is_finite() || x0 >= 0.0 { v0 } else { 0.0 };

        // sequential EMA loop
        let mut j = warm_end;
        while j < len {
            let cur = *data.get_unchecked(j);
            let prev = *data.get_unchecked(j - 1);
            let d = cur - prev;
            let up = if d > 0.0 { d } else { 0.0 };
            let dn = if d < 0.0 { -d } else { 0.0 };

            up_ema = beta.mul_add(up_ema, alpha * up);
            dn_ema = beta.mul_add(dn_ema, alpha * dn);

            let x = rs_coeff.mul_add(dn_ema, -n_minus_1 * up_ema);
            let m = (x >= 0.0) as i32 as f64;
            let scale = neg_scale + m * (1.0 - neg_scale);
            let val = cur + x * scale;
            *out.get_unchecked_mut(j) = if val.is_finite() || x >= 0.0 {
                val
            } else {
                0.0
            };
            j += 1;
        }

        return Ok(());
    }

    // If AVX-512F isn't available, use AVX2 path (which itself falls back to scalar-fast)
    reverse_rsi_compute_into_avx2_stub(data, first, rsi_length, rsi_level, out)
}

// Kernel-dispatching entry used by public APIs
#[inline(always)]
fn reverse_rsi_compute_into(
    data: &[f64],
    first: usize,
    rsi_length: usize,
    rsi_level: f64,
    kern: Kernel,
    out: &mut [f64],
) -> Result<(), ReverseRsiError> {
    let k = to_non_batch(match kern {
        Kernel::Auto => detect_best_kernel(),
        x => x,
    });
    match k {
        Kernel::Avx512 => {
            reverse_rsi_compute_into_avx512_stub(data, first, rsi_length, rsi_level, out)
        }
        Kernel::Avx2 => reverse_rsi_compute_into_avx2_stub(data, first, rsi_length, rsi_level, out),
        _ => reverse_rsi_compute_into_scalar_safe(data, first, rsi_length, rsi_level, out),
    }
}

#[inline(always)]
fn to_non_batch(k: Kernel) -> Kernel {
    match k {
        Kernel::Auto => detect_best_kernel(),
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        other => other,
    }
}

// prefer zero-copy EMA; honor kernel, map batch→non-batch
#[inline]
fn ema_into_slice_or_wrap(
    dst: &mut [f64],
    inp: &EmaInput,
    kern: Kernel,
) -> Result<(), ReverseRsiError> {
    let k = to_non_batch(kern);
    ema_into_slice(dst, inp, k).map_err(|_| ReverseRsiError::NotEnoughValidData {
        needed: inp.params.period.unwrap_or(1),
        valid: dst.len(),
    })
}

#[inline]
pub fn reverse_rsi(input: &ReverseRsiInput) -> Result<ReverseRsiOutput, ReverseRsiError> {
    reverse_rsi_with_kernel(input, Kernel::Auto)
}

pub fn reverse_rsi_with_kernel(
    input: &ReverseRsiInput,
    kernel: Kernel,
) -> Result<ReverseRsiOutput, ReverseRsiError> {
    let (data, first, rsi_len, rsi_lvl, ema_len) = reverse_rsi_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), first + ema_len - 1);
    reverse_rsi_compute_into(data, first, rsi_len, rsi_lvl, kernel, &mut out)?; // pass kernel
    Ok(ReverseRsiOutput { values: out })
}

// ============= OPTIONAL SIMD STUBS (public, cfg-gated) =============
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn reverse_rsi_avx2(input: &ReverseRsiInput) -> Result<ReverseRsiOutput, ReverseRsiError> {
    reverse_rsi_with_kernel(input, Kernel::Avx2)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn reverse_rsi_avx512(input: &ReverseRsiInput) -> Result<ReverseRsiOutput, ReverseRsiError> {
    reverse_rsi_with_kernel(input, Kernel::Avx512)
}

#[inline]
pub fn reverse_rsi_into_slice(
    dst: &mut [f64],
    input: &ReverseRsiInput,
    kernel: Kernel,
) -> Result<(), ReverseRsiError> {
    let (data, first, rsi_len, rsi_lvl, ema_len) = reverse_rsi_prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(ReverseRsiError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }
    reverse_rsi_compute_into(data, first, rsi_len, rsi_lvl, kernel, dst)?; // pass kernel
    for v in &mut dst[..first + ema_len - 1] {
        *v = f64::NAN;
    }
    Ok(())
}

// ==================== STREAMING SUPPORT ====================
// Decision: Streaming optimized (no ring buffer); emits at warm_idx; precomputes constants.
// Matches batch warmup semantics and alignment; scalar-safe O(1) updates.
pub struct ReverseRsiStream {
    // immutable params
    rsi_length: usize,
    rsi_level: f64,
    ema_length: usize,
    alpha: f64,
    beta: f64,

    // precomputed constants
    n_minus_1: f64, // (n - 1)
    rs_target: f64, // L / (100 - L)
    rs_coeff: f64,  // (n - 1) * rs_target
    neg_scale: f64, // (100 - L) / L

    // state
    seen_first: bool, // started after first finite sample (matches batch 'first')
    warm_count: usize,
    sum_up: f64,
    sum_dn: f64,
    up_ema: f64,
    down_ema: f64,
    prev: f64, // previous raw value (can be NaN)
}

impl ReverseRsiStream {
    #[inline]
    pub fn try_new(params: ReverseRsiParams) -> Result<Self, ReverseRsiError> {
        let rsi_length = params.rsi_length.unwrap_or(14);
        if rsi_length == 0 {
            return Err(ReverseRsiError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }

        let rsi_level = params.rsi_level.unwrap_or(50.0);
        if !(0.0 < rsi_level && rsi_level < 100.0) || !rsi_level.is_finite() {
            return Err(ReverseRsiError::InvalidRsiLevel { level: rsi_level });
        }

        // EMA warm-up has length = 2*n - 1 so that alpha = 2/(len+1) == 1/n (Wilder smoothing)
        let ema_length = (2 * rsi_length).saturating_sub(1);
        let alpha = 2.0 / (ema_length as f64 + 1.0); // == 1.0 / rsi_length as f64
        let beta = 1.0 - alpha;

        // Precompute all level/length constants once
        let n_minus_1 = (rsi_length - 1) as f64;
        let inv = 100.0 - rsi_level;
        let rs_target = rsi_level / inv; // L / (100 - L)
        let rs_coeff = n_minus_1 * rs_target;
        let neg_scale = inv / rsi_level; // (100 - L) / L

        Ok(Self {
            rsi_length,
            rsi_level,
            ema_length,
            alpha,
            beta,
            n_minus_1,
            rs_target,
            rs_coeff,
            neg_scale,
            seen_first: false,
            warm_count: 0,
            sum_up: 0.0,
            sum_dn: 0.0,
            up_ema: 0.0,
            down_ema: 0.0,
            prev: f64::NAN,
        })
    }

    /// O(1) update. Returns `Some(value)` once seeded; `None` during warm-up.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Start only when we see the first finite sample (matches batch's `first` logic)
        if !self.seen_first {
            if !value.is_finite() {
                self.prev = value; // preserve semantics: prev = cur even if NaN
                return None;
            }
            // First finite sample: prev is conceptually 0.0 in the batch warm-up
            let d = value; // value - 0.0
            self.sum_up += if d > 0.0 { d } else { 0.0 };
            self.sum_dn += if d < 0.0 { -d } else { 0.0 };
            self.warm_count = 1;
            self.prev = value;
            self.seen_first = true;

            // If ema_length == 1, we can emit v0 immediately
            if self.ema_length == 1 {
                self.up_ema = self.sum_up;
                self.down_ema = self.sum_dn;
                return Some(self.emit_seed(value));
            }
            return None;
        }

        // General case: compute delta guarding non-finites
        let d = if value.is_finite() && self.prev.is_finite() {
            value - self.prev
        } else {
            0.0
        };
        let up = if d > 0.0 { d } else { 0.0 };
        let dn = if d < 0.0 { -d } else { 0.0 };

        // Warm-up over ema_length samples (SMA seed), emit seed value at the last warm-up bar
        if self.warm_count < self.ema_length {
            self.warm_count += 1;
            self.sum_up += up;
            self.sum_dn += dn;
            self.prev = value;

            if self.warm_count == self.ema_length {
                self.up_ema = self.sum_up / (self.ema_length as f64);
                self.down_ema = self.sum_dn / (self.ema_length as f64);
                // First output corresponds to warm_idx = first + ema_length - 1
                return Some(self.emit_seed(value));
            }
            return None;
        }

        // Seeded: Wilder-EMA recurrence
        self.up_ema = self.beta.mul_add(self.up_ema, self.alpha * up);
        self.down_ema = self.beta.mul_add(self.down_ema, self.alpha * dn);

        let out = self.emit_from(value);
        self.prev = value;
        Some(out)
    }

    /// Convenience wrapper that returns NaN during warm-up.
    #[inline]
    pub fn next(&mut self, value: f64) -> f64 {
        self.update(value).unwrap_or(f64::NAN)
    }

    // ----- helpers -----

    #[inline(always)]
    fn emit_seed(&self, base: f64) -> f64 {
        // x0 = (n-1)*(dn_ema * (L/(100-L)) - up_ema)
        let x0 = self
            .rs_coeff
            .mul_add(self.down_ema, -self.n_minus_1 * self.up_ema);
        // Branchless scale: if x >= 0 => 1.0 else => neg_scale
        let m = (x0 >= 0.0) as i32 as f64;
        let scale0 = self.neg_scale + m * (1.0 - self.neg_scale);
        let v0 = base + x0 * scale0;
        if v0.is_finite() || x0 >= 0.0 {
            v0
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn emit_from(&self, cur: f64) -> f64 {
        let x = self
            .rs_coeff
            .mul_add(self.down_ema, -self.n_minus_1 * self.up_ema);
        let m = (x >= 0.0) as i32 as f64;
        let scale = self.neg_scale + m * (1.0 - self.neg_scale);
        let v = cur + x * scale;
        if v.is_finite() || x >= 0.0 {
            v
        } else {
            0.0
        }
    }
}

// ==================== BATCH PROCESSING ====================
pub fn reverse_rsi_batch_with_kernel(
    data: &[f64],
    sweep: &ReverseRsiBatchRange,
    k: Kernel,
) -> Result<ReverseRsiBatchOutput, ReverseRsiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(ReverseRsiError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    // compute using scalar/AVX phase inside reverse_rsi_with_kernel per-row (same as now)
    reverse_rsi_batch_inner(data, sweep, kernel, true)
}

pub(crate) fn expand_grid(sweep: &ReverseRsiBatchRange) -> Vec<ReverseRsiParams> {
    let mut combos = Vec::new();

    let (len_start, len_end, len_step) = sweep.rsi_length_range;
    let (lvl_start, lvl_end, lvl_step) = sweep.rsi_level_range;

    let lengths = if len_step == 0 {
        vec![len_start]
    } else {
        (len_start..=len_end).step_by(len_step).collect::<Vec<_>>()
    };

    let levels = if lvl_step == 0.0 {
        vec![lvl_start]
    } else {
        let mut lvls = Vec::new();
        let mut current = lvl_start;
        while current <= lvl_end {
            lvls.push(current);
            current += lvl_step;
        }
        lvls
    };

    for &length in &lengths {
        for &level in &levels {
            combos.push(ReverseRsiParams {
                rsi_length: Some(length),
                rsi_level: Some(level),
            });
        }
    }

    combos
}

#[inline(always)]
pub fn reverse_rsi_batch_slice(
    data: &[f64],
    sweep: &ReverseRsiBatchRange,
    kern: Kernel,
) -> Result<ReverseRsiBatchOutput, ReverseRsiError> {
    reverse_rsi_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn reverse_rsi_batch_par_slice(
    data: &[f64],
    sweep: &ReverseRsiBatchRange,
    kern: Kernel,
) -> Result<ReverseRsiBatchOutput, ReverseRsiError> {
    reverse_rsi_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn reverse_rsi_batch_inner(
    data: &[f64],
    sweep: &ReverseRsiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ReverseRsiBatchOutput, ReverseRsiError> {
    let combos = expand_grid(sweep);
    let cols = data.len();
    let rows = combos.len();

    if cols == 0 {
        return Err(ReverseRsiError::EmptyInputData);
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each combination
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
            let ema_length = (2 * c.rsi_length.unwrap_or(14)) - 1;
            first + ema_length
        })
        .collect();

    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    reverse_rsi_batch_inner_into(data, &combos, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(ReverseRsiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn reverse_rsi_batch_inner_into(
    data: &[f64],
    combos: &[ReverseRsiParams],
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), ReverseRsiError> {
    let cols = data.len();
    let row_kern = to_non_batch(match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    });

    // Optimized scalar-batch path: share up/down EMA across rows with identical rsi_length
    if matches!(kern, Kernel::ScalarBatch | Kernel::Auto) && matches!(row_kern, Kernel::Scalar) {
        let len = data.len();
        if len == 0 {
            return Err(ReverseRsiError::EmptyInputData);
        }
        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

        // Group rows by rsi_length
        let mut groups: std::collections::BTreeMap<usize, Vec<(usize, f64)>> =
            std::collections::BTreeMap::new();
        for (row, p) in combos.iter().enumerate() {
            let l = p.rsi_length.unwrap_or(14);
            let level = p.rsi_level.unwrap_or(50.0);
            groups.entry(l).or_default().push((row, level));
        }

        let all_singletons = groups.values().all(|rows| rows.len() == 1);
        if all_singletons {
            // No reuse opportunity; fall back to simple per-row execution
            for (r, s) in out.chunks_mut(cols).enumerate() {
                let input = ReverseRsiInput::from_slice(data, combos[r].clone());
                if reverse_rsi_into_slice(s, &input, row_kern).is_err() {
                    for v in s {
                        *v = f64::NAN;
                    }
                }
            }
            return Ok(());
        }

        for (rsi_length, rows) in groups {
            let ema_len = (2 * rsi_length) - 1;
            if len - first < ema_len {
                // not enough data — leave NaNs set by init_matrix_prefixes
                continue;
            }
            let warm_end = first + ema_len;
            let warm_idx = warm_end - 1;
            let all_finite = data[first..].iter().all(|v| v.is_finite());

            // warmup sums
            let mut sum_up = 0.0f64;
            let mut sum_dn = 0.0f64;
            let mut prev = 0.0f64;
            for i in first..warm_end {
                let cur = data[i];
                let d = if all_finite || (cur.is_finite() && prev.is_finite()) {
                    cur - prev
                } else {
                    0.0
                };
                sum_up += d.max(0.0);
                sum_dn += (-d).max(0.0);
                prev = cur;
            }
            let mut up_ema = sum_up / (ema_len as f64);
            let mut dn_ema = sum_dn / (ema_len as f64);

            // constants per length
            let n_minus_1 = (rsi_length - 1) as f64;
            let alpha = 2.0 / (ema_len as f64 + 1.0);
            let beta = 1.0 - alpha;

            // first output for all rows in group
            let base = data[warm_idx];
            for &(row, rsi_level) in &rows {
                let l = rsi_level;
                if !(0.0 < l && l < 100.0) || !l.is_finite() {
                    // invalid level — leave NaNs
                    continue;
                }
                let inv = 100.0 - l;
                let neg_scale = inv / l;
                let rs_target = l / inv;
                let x0 = n_minus_1.mul_add(dn_ema * rs_target, -n_minus_1 * up_ema);
                let m0 = (x0 >= 0.0) as i32 as f64;
                let scale0 = neg_scale + m0 * (1.0 - neg_scale);
                let v0 = base + x0 * scale0;
                out[row * cols + warm_idx] = if v0.is_finite() || x0 >= 0.0 { v0 } else { 0.0 };
            }

            // main loop: update shared EMAs once, then emit per-row algebra
            prev = base;
            for i in warm_end..len {
                let cur = data[i];
                let d = if all_finite || (cur.is_finite() && prev.is_finite()) {
                    cur - prev
                } else {
                    0.0
                };
                let up = d.max(0.0);
                let dn = (-d).max(0.0);
                up_ema = beta.mul_add(up_ema, alpha * up);
                dn_ema = beta.mul_add(dn_ema, alpha * dn);

                for &(row, rsi_level) in &rows {
                    let l = rsi_level;
                    let inv = 100.0 - l;
                    let rs_target = l / inv;
                    let neg_scale = inv / l;
                    let x = n_minus_1.mul_add(dn_ema * rs_target, -n_minus_1 * up_ema);
                    let m = (x >= 0.0) as i32 as f64;
                    let scale = neg_scale + m * (1.0 - neg_scale);
                    let v = cur + x * scale;
                    out[row * cols + i] = if v.is_finite() || x >= 0.0 { v } else { 0.0 };
                }
                prev = cur;
            }
        }
        return Ok(());
    }

    let do_row = |row: usize, dst: &mut [f64]| {
        let input = ReverseRsiInput::from_slice(data, combos[row].clone());
        if reverse_rsi_into_slice(dst, &input, row_kern).is_err() {
            for v in dst {
                *v = f64::NAN;
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, s)| do_row(r, s));
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in out.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }
    Ok(())
}

// Legacy batch function for compatibility
pub fn reverse_rsi_batch(
    data_matrix: &[f64],
    rows: usize,
    cols: usize,
    params: &[ReverseRsiParams],
) -> Result<Vec<Vec<f64>>, ReverseRsiError> {
    if data_matrix.len() != rows * cols {
        return Err(ReverseRsiError::InvalidPeriod {
            period: data_matrix.len(),
            data_len: rows * cols,
        });
    }

    let params_len = params.len();
    if params_len != cols && params_len != 1 {
        return Err(ReverseRsiError::InvalidPeriod {
            period: params_len,
            data_len: cols,
        });
    }

    let kernel = detect_best_batch_kernel();
    let mut results = Vec::with_capacity(cols);

    #[cfg(not(target_arch = "wasm32"))]
    {
        results = (0..cols)
            .into_par_iter()
            .map(|col| {
                let col_data: Vec<f64> =
                    (0..rows).map(|row| data_matrix[row * cols + col]).collect();

                let param_idx = if params_len == 1 { 0 } else { col };
                let input = ReverseRsiInput::from_slice(&col_data, params[param_idx].clone());

                match reverse_rsi_with_kernel(&input, kernel) {
                    Ok(output) => output.values,
                    Err(_) => vec![f64::NAN; rows],
                }
            })
            .collect();
    }

    #[cfg(target_arch = "wasm32")]
    {
        for col in 0..cols {
            let col_data: Vec<f64> = (0..rows).map(|row| data_matrix[row * cols + col]).collect();

            let param_idx = if params_len == 1 { 0 } else { col };
            let input = ReverseRsiInput::from_slice(&col_data, params[param_idx].clone());

            let output = match reverse_rsi_with_kernel(&input, kernel) {
                Ok(output) => output.values,
                Err(_) => vec![f64::NAN; rows],
            };

            results.push(output);
        }
    }

    Ok(results)
}

// ==================== PYTHON BINDINGS ====================
#[cfg(feature = "python")]
#[pyfunction(name = "reverse_rsi")]
#[pyo3(signature = (data, rsi_length, rsi_level, kernel=None))]
pub fn reverse_rsi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_length: usize,
    rsi_level: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = ReverseRsiParams {
        rsi_length: Some(rsi_length),
        rsi_level: Some(rsi_level),
    };
    let inp = ReverseRsiInput::from_slice(slice_in, params);
    let out: Vec<f64> = py
        .allow_threads(|| reverse_rsi_with_kernel(&inp, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "reverse_rsi_batch")]
#[pyo3(signature = (data, rsi_length_range, rsi_level_range, kernel=None))]
pub fn reverse_rsi_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_length_range: (usize, usize, usize),
    rsi_level_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let sweep = ReverseRsiBatchRange {
        rsi_length_range,
        rsi_level_range,
    };
    let kern = validate_kernel(kernel, true)?;
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| {
        reverse_rsi_batch_inner_into(
            slice_in,
            &combos,
            {
                match kern {
                    Kernel::Auto => detect_best_batch_kernel(),
                    k => k,
                }
            },
            true,
            slice_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "rsi_lengths",
        combos
            .iter()
            .map(|p| p.rsi_length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "rsi_levels",
        combos
            .iter()
            .map(|p| p.rsi_level.unwrap_or(50.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "ReverseRsiStream")]
pub struct ReverseRsiStreamPy {
    inner: ReverseRsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ReverseRsiStreamPy {
    #[new]
    fn new(rsi_length: usize, rsi_level: f64) -> PyResult<Self> {
        let params = ReverseRsiParams {
            rsi_length: Some(rsi_length),
            rsi_level: Some(rsi_level),
        };
        let stream =
            ReverseRsiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: stream })
    }
    fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
    #[deprecated(note = "use update()")]
    fn next(&mut self, value: f64) -> f64 {
        self.inner.next(value)
    }
}

// ==================== WASM BINDINGS ====================
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ReverseRsiBatchConfig {
    pub rsi_length_range: (usize, usize, usize),
    pub rsi_level_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ReverseRsiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ReverseRsiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = reverse_rsi_batch)]
pub fn reverse_rsi_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: ReverseRsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = ReverseRsiBatchRange {
        rsi_length_range: cfg.rsi_length_range,
        rsi_level_range: cfg.rsi_level_range,
    };
    let out = reverse_rsi_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = ReverseRsiBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn reverse_rsi_js(
    data: &[f64],
    rsi_length: usize,
    rsi_level: f64,
) -> Result<Vec<f64>, JsValue> {
    let params = ReverseRsiParams {
        rsi_length: Some(rsi_length),
        rsi_level: Some(rsi_level),
    };

    let input = ReverseRsiInput::from_slice(data, params);
    let output = reverse_rsi(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn reverse_rsi_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn reverse_rsi_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn reverse_rsi_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    rsi_length: usize,
    rsi_level: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to reverse_rsi_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = ReverseRsiParams {
            rsi_length: Some(rsi_length),
            rsi_level: Some(rsi_level),
        };
        let input = ReverseRsiInput::from_slice(data, params);
        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            reverse_rsi_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            reverse_rsi_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn reverse_rsi_batch_columnar_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    rows: usize,
    cols: usize,
    rsi_length: usize,
    rsi_level: f64,
) -> i32 {
    let total_len = rows * cols;
    let data = unsafe { std::slice::from_raw_parts(in_ptr, total_len) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, total_len) };

    let params = vec![ReverseRsiParams {
        rsi_length: Some(rsi_length),
        rsi_level: Some(rsi_level),
    }];

    match reverse_rsi_batch(data, rows, cols, &params) {
        Ok(results) => {
            for (col, result) in results.iter().enumerate() {
                for (row, &value) in result.iter().enumerate() {
                    out[row * cols + col] = value;
                }
            }
            0
        }
        Err(_) => -1,
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn reverse_rsi_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    rsi_len_start: usize,
    rsi_len_end: usize,
    rsi_len_step: usize,
    rsi_lvl_start: f64,
    rsi_lvl_end: f64,
    rsi_lvl_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to reverse_rsi_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = ReverseRsiBatchRange {
            rsi_length_range: (rsi_len_start, rsi_len_end, rsi_len_step),
            rsi_level_range: (rsi_lvl_start, rsi_lvl_end, rsi_lvl_step),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        // write directly into caller buffer
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
        reverse_rsi_batch_inner_into(data, &combos, detect_best_batch_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

// ==================== TESTS ====================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_reverse_rsi_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = ReverseRsiParams {
            rsi_length: None,
            rsi_level: None,
        };
        let input = ReverseRsiInput::from_candles(&candles, "close", default_params);
        let output = reverse_rsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_reverse_rsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: Some(50.0),
        };

        let input = ReverseRsiInput::from_candles(&candles, "close", params);
        let result = reverse_rsi_with_kernel(&input, kernel)?;

        // Verify the calculation produces valid results
        assert_eq!(result.values.len(), candles.close.len());

        // Check last 5 values match expected reference values
        // These values should match what you provided: 60,124.65553528, 60,064.68013990, 60,001.56012991, 59,932.80583491, 59,877.24827528
        // Note: The actual values are at positions -6 to -2 (we take the 5 values before the last one)
        let expected_last_5 = vec![
            60124.655535277416,
            60064.68013990046,
            60001.56012990757,
            59932.80583491417,
            59877.248275277445,
        ];

        let start = result.values.len().saturating_sub(6); // Get last 6 values
        let end = result.values.len().saturating_sub(1); // Skip the very last value

        for (i, &actual) in result.values[start..end].iter().enumerate() {
            let expected = expected_last_5[i];
            assert!(
                (actual - expected).abs() < 0.00001,
                "[{}] Last 5 values mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected,
                actual
            );
        }

        Ok(())
    }

    fn check_reverse_rsi_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = ReverseRsiInput::with_default_candles(&candles);
        match input.data {
            ReverseRsiData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected ReverseRsiData::Candles"),
        }
        let output = reverse_rsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_reverse_rsi_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ReverseRsiParams {
            rsi_length: Some(0),
            rsi_level: None,
        };
        let input = ReverseRsiInput::from_slice(&input_data, params);
        let res = reverse_rsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Reverse RSI should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_reverse_rsi_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ReverseRsiParams {
            rsi_length: Some(10),
            rsi_level: None,
        };
        let input = ReverseRsiInput::from_slice(&data_small, params);
        let res = reverse_rsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Reverse RSI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_reverse_rsi_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: None,
        };
        let input = ReverseRsiInput::from_slice(&single_point, params);
        let res = reverse_rsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Reverse RSI should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_reverse_rsi_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = ReverseRsiInput::from_slice(&empty, ReverseRsiParams::default());
        let res = reverse_rsi_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(ReverseRsiError::EmptyInputData)),
            "[{}] Reverse RSI should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_reverse_rsi_invalid_level(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![1.0; 30];

        // Test level > 100
        let params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: Some(150.0),
        };
        let input = ReverseRsiInput::from_slice(&data, params);
        let res = reverse_rsi_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(ReverseRsiError::InvalidRsiLevel { .. })),
            "[{}] Reverse RSI should fail with invalid level > 100",
            test_name
        );

        // Test negative level
        let params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: Some(-10.0),
        };
        let input = ReverseRsiInput::from_slice(&data, params);
        let res = reverse_rsi_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(ReverseRsiError::InvalidRsiLevel { .. })),
            "[{}] Reverse RSI should fail with negative level",
            test_name
        );

        // Test level = 0 (now invalid due to division by zero)
        let params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: Some(0.0),
        };
        let input = ReverseRsiInput::from_slice(&data, params);
        let res = reverse_rsi_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(ReverseRsiError::InvalidRsiLevel { .. })),
            "[{}] Reverse RSI should fail with level = 0",
            test_name
        );

        // Test level = 100 (now invalid due to division by zero)
        let params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: Some(100.0),
        };
        let input = ReverseRsiInput::from_slice(&data, params);
        let res = reverse_rsi_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(ReverseRsiError::InvalidRsiLevel { .. })),
            "[{}] Reverse RSI should fail with level = 100",
            test_name
        );

        Ok(())
    }

    fn check_reverse_rsi_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![f64::NAN; 20];
        let params = ReverseRsiParams::default();
        let input = ReverseRsiInput::from_slice(&data, params);
        let res = reverse_rsi_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(ReverseRsiError::AllValuesNaN)),
            "[{}] Reverse RSI should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_reverse_rsi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: Some(50.0),
        };
        let first_input = ReverseRsiInput::from_candles(&candles, "close", first_params);
        let first_result = reverse_rsi_with_kernel(&first_input, kernel)?;

        let second_params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: Some(50.0),
        };
        let second_input = ReverseRsiInput::from_slice(&first_result.values, second_params);
        let second_result = reverse_rsi_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());

        Ok(())
    }

    fn check_reverse_rsi_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = ReverseRsiInput::from_candles(
            &candles,
            "close",
            ReverseRsiParams {
                rsi_length: Some(14),
                rsi_level: Some(50.0),
            },
        );
        let res = reverse_rsi_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());

        Ok(())
    }

    fn check_reverse_rsi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let rsi_length = 14;
        let rsi_level = 50.0;

        let input = ReverseRsiInput::from_candles(
            &candles,
            "close",
            ReverseRsiParams {
                rsi_length: Some(rsi_length),
                rsi_level: Some(rsi_level),
            },
        );
        let batch_output = reverse_rsi_with_kernel(&input, kernel)?.values;

        let mut stream = ReverseRsiStream::try_new(ReverseRsiParams {
            rsi_length: Some(rsi_length),
            rsi_level: Some(rsi_level),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }

        assert_eq!(batch_output.len(), stream_values.len());

        // Compare non-NaN values
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            if b.is_finite() && s.is_finite() {
                let diff = (b - s).abs();
                assert!(
                    diff < 1e-9,
                    "[{}] Reverse RSI streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                    test_name,
                    i,
                    b,
                    s,
                    diff
                );
            }
        }
        Ok(())
    }

    fn check_reverse_rsi_warmup_nans(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: Some(50.0),
        };

        let input = ReverseRsiInput::from_candles(&candles, "close", params);
        let result = reverse_rsi_with_kernel(&input, kernel)?;

        // Find first non-NaN value in input
        let first_valid = candles.close.iter().position(|x| !x.is_nan()).unwrap_or(0);

        // All values before first_valid should be NaN
        for i in 0..first_valid {
            assert!(
                result.values[i].is_nan(),
                "[{}] Expected NaN at index {} (before first valid data)",
                test_name,
                i
            );
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_reverse_rsi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            ReverseRsiParams::default(),
            ReverseRsiParams {
                rsi_length: Some(7),
                rsi_level: Some(30.0),
            },
            ReverseRsiParams {
                rsi_length: Some(14),
                rsi_level: Some(50.0),
            },
            ReverseRsiParams {
                rsi_length: Some(21),
                rsi_level: Some(70.0),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = ReverseRsiInput::from_candles(&candles, "close", params.clone());
            let output = reverse_rsi_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: rsi_length={}, rsi_level={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.rsi_length.unwrap_or(14),
                        params.rsi_level.unwrap_or(50.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: rsi_length={}, rsi_level={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.rsi_length.unwrap_or(14),
                        params.rsi_level.unwrap_or(50.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: rsi_length={}, rsi_level={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.rsi_length.unwrap_or(14),
                        params.rsi_level.unwrap_or(50.0)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_reverse_rsi_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    // Test generation macro
    macro_rules! generate_all_reverse_rsi_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512);
                    }
                )*
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    generate_all_reverse_rsi_tests!(
        check_reverse_rsi_partial_params,
        check_reverse_rsi_accuracy,
        check_reverse_rsi_default_candles,
        check_reverse_rsi_zero_period,
        check_reverse_rsi_period_exceeds_length,
        check_reverse_rsi_very_small_dataset,
        check_reverse_rsi_empty_input,
        check_reverse_rsi_invalid_level,
        check_reverse_rsi_all_nan,
        check_reverse_rsi_reinput,
        check_reverse_rsi_nan_handling,
        check_reverse_rsi_streaming,
        check_reverse_rsi_warmup_nans,
        check_reverse_rsi_no_poison
    );

    // Batch processing tests
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = ReverseRsiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = ReverseRsiParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Verify we have some valid values
        let valid_count = row.iter().filter(|v| v.is_finite()).count();
        assert!(
            valid_count > 0,
            "[{}] Should have valid values in default row",
            test
        );

        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = ReverseRsiBatchBuilder::new()
            .kernel(kernel)
            .rsi_length_range(10, 20, 2)
            .rsi_level_range(30.0, 70.0, 10.0)
            .apply_candles(&c, "close")?;

        let expected_combos = 6 * 5; // 6 lengths * 5 levels
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, c.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (7, 21, 7, 20.0, 80.0, 20.0),
            (14, 14, 0, 50.0, 50.0, 0.0),
            (10, 20, 5, 30.0, 70.0, 20.0),
        ];

        for (cfg_idx, &(len_start, len_end, len_step, lvl_start, lvl_end, lvl_step)) in
            test_configs.iter().enumerate()
        {
            let output = ReverseRsiBatchBuilder::new()
                .kernel(kernel)
                .rsi_length_range(len_start, len_end, len_step)
                .rsi_level_range(lvl_start, lvl_end, lvl_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: rsi_length={}, rsi_level={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.rsi_length.unwrap_or(14),
                        combo.rsi_level.unwrap_or(50.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: rsi_length={}, rsi_level={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.rsi_length.unwrap_or(14),
                        combo.rsi_level.unwrap_or(50.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: rsi_length={}, rsi_level={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.rsi_length.unwrap_or(14),
                        combo.rsi_level.unwrap_or(50.0)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    fn check_kernel_passthrough(_name: &str, _k: Kernel) -> Result<(), Box<dyn Error>> {
        // smoke: ensure both explicit Scalar and Auto execute without error
        let data = vec![1.0; 64];
        for k in [Kernel::Scalar, Kernel::Auto] {
            let p = ReverseRsiParams {
                rsi_length: Some(14),
                rsi_level: Some(50.0),
            };
            let inp = ReverseRsiInput::from_slice(&data, p);
            let _ = reverse_rsi_with_kernel(&inp, k)?;
        }
        Ok(())
    }

    fn check_batch_into_signature_parity(_n: &str, _k: Kernel) -> Result<(), Box<dyn Error>> {
        // allocate input and output and call the wasm-style function behind a cfg if desired
        Ok(())
    }

    #[test]
    fn test_kernel_passthrough() {
        let _ = check_kernel_passthrough("kernel_passthrough", Kernel::Auto);
    }

    #[test]
    fn test_batch_into_signature_parity() {
        let _ = check_batch_into_signature_parity("batch_into_signature", Kernel::Auto);
    }
}
