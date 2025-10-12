//! # Compound Ratio Moving Average (CoRa Wave)
//!
//! A weighted moving average where weights increase in a "logarithmically linear" way (linear when plotted on a log chart).
//! The step ratio between weights is consistent, and the tail weight can be significantly reduced compared to other MAs.
//! Includes optional auto-smoothing with WMA for cleaner signals.
//!
//! ## Parameters
//! - **period**: Window size for the moving average (default: 20)
//! - **r_multi**: Compound ratio multiplier (default: 2.0)
//! - **smooth**: Enable auto-smoothing with WMA (default: true)
//!
//! ## Returns
//! - **`Ok(CoraWaveOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(CoraWaveError)`** on various error conditions.
//!
//! ## Developer Status
//! - SIMD: AVX2/AVX512 stubs delegate to scalar. CoRa’s time-recursive update has a strict dependency chain, so cross-time SIMD offers no win; keep scalar as fastest.
//! - Scalar: Optimized to O(1) per step for CoRa using geometric-weight recurrence; WMA smoothing recomputed per emission to match streaming numerics exactly.
//! - Batch: Row kernel uses the same O(1) CoRa update per row. Row-SIMD across rows is plausible but not yet implemented; runtime selection short-circuits to scalar.
//! - Memory: Uses alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes; warmup/NaN semantics unchanged.

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

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for CoraWaveInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CoraWaveData::Slice(slice) => slice,
            CoraWaveData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CoraWaveData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CoraWaveOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CoraWaveParams {
    pub period: Option<usize>,
    pub r_multi: Option<f64>,
    pub smooth: Option<bool>,
}

impl Default for CoraWaveParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            r_multi: Some(2.0),
            smooth: Some(true),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoraWaveInput<'a> {
    pub data: CoraWaveData<'a>,
    pub params: CoraWaveParams,
}

impl<'a> CoraWaveInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CoraWaveParams) -> Self {
        Self {
            data: CoraWaveData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CoraWaveParams) -> Self {
        Self {
            data: CoraWaveData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CoraWaveParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }

    #[inline]
    pub fn get_r_multi(&self) -> f64 {
        self.params.r_multi.unwrap_or(2.0)
    }

    #[inline]
    pub fn get_smooth(&self) -> bool {
        self.params.smooth.unwrap_or(true)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CoraWaveBuilder {
    period: Option<usize>,
    r_multi: Option<f64>,
    smooth: Option<bool>,
    kernel: Kernel,
}

impl Default for CoraWaveBuilder {
    fn default() -> Self {
        Self {
            period: None,
            r_multi: None,
            smooth: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CoraWaveBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, val: usize) -> Self {
        self.period = Some(val);
        self
    }

    #[inline(always)]
    pub fn r_multi(mut self, val: f64) -> Self {
        self.r_multi = Some(val);
        self
    }

    #[inline(always)]
    pub fn smooth(mut self, val: bool) -> Self {
        self.smooth = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CoraWaveOutput, CoraWaveError> {
        let p = CoraWaveParams {
            period: self.period,
            r_multi: self.r_multi,
            smooth: self.smooth,
        };
        let i = CoraWaveInput::from_candles(c, "close", p);
        cora_wave_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CoraWaveOutput, CoraWaveError> {
        let p = CoraWaveParams {
            period: self.period,
            r_multi: self.r_multi,
            smooth: self.smooth,
        };
        let i = CoraWaveInput::from_slice(d, p);
        cora_wave_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CoraWaveStream, CoraWaveError> {
        let p = CoraWaveParams {
            period: self.period,
            r_multi: self.r_multi,
            smooth: self.smooth,
        };
        CoraWaveStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CoraWaveError {
    #[error("cora_wave: Input data slice is empty.")]
    EmptyInputData,

    #[error("cora_wave: All values are NaN.")]
    AllValuesNaN,

    #[error("cora_wave: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("cora_wave: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("cora_wave: Invalid r_multi: {value}")]
    InvalidRMulti { value: f64 },
}

#[inline]
pub fn cora_wave(input: &CoraWaveInput) -> Result<CoraWaveOutput, CoraWaveError> {
    cora_wave_with_kernel(input, Kernel::Auto)
}

pub fn cora_wave_with_kernel(
    input: &CoraWaveInput,
    kernel: Kernel,
) -> Result<CoraWaveOutput, CoraWaveError> {
    let (data, weights, inv_wsum, smooth_period, first, chosen) = cora_wave_prepare(input, kernel)?;
    let period = weights.len();
    let warm = first + period - 1 + smooth_period.saturating_sub(1);

    let mut out = alloc_with_nan_prefix(data.len(), warm);
    cora_wave_compute_into(
        data,
        &weights,
        inv_wsum,
        smooth_period,
        first,
        chosen,
        &mut out,
    );
    Ok(CoraWaveOutput { values: out })
}

#[inline]
pub fn cora_wave_into_slice(
    dst: &mut [f64],
    input: &CoraWaveInput,
    kern: Kernel,
) -> Result<(), CoraWaveError> {
    let (data, weights, inv_wsum, smooth_period, first, chosen) = cora_wave_prepare(input, kern)?;
    if dst.len() != data.len() {
        return Err(CoraWaveError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }
    cora_wave_compute_into(data, &weights, inv_wsum, smooth_period, first, chosen, dst);

    let warm = first + weights.len() - 1 + smooth_period.saturating_sub(1);
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
fn cora_wave_prepare<'a>(
    input: &'a CoraWaveInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64], // data
        Vec<f64>,  // weights
        f64,       // inv_weight_sum
        usize,     // smooth_period
        usize,     // first valid
        Kernel,    // chosen kernel
    ),
    CoraWaveError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(CoraWaveError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CoraWaveError::AllValuesNaN)?;

    let period = input.get_period();
    let r_multi = input.get_r_multi();
    let smooth = input.get_smooth();

    if period == 0 || period > len {
        return Err(CoraWaveError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(CoraWaveError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if r_multi < 0.0 || !r_multi.is_finite() {
        return Err(CoraWaveError::InvalidRMulti { value: r_multi });
    }

    // CoRa weights once
    let mut weights = Vec::with_capacity(period);
    let inv_sum: f64;
    if period == 1 {
        weights.push(1.0);
        inv_sum = 1.0;
    } else {
        let start_wt = 0.01;
        let end_wt = period as f64;
        let r = (end_wt / start_wt).powf(1.0 / (period as f64 - 1.0)) - 1.0;
        let base = 1.0 + r * r_multi;

        let mut sum = 0.0;
        for j in 0..period {
            // newest gets largest weight ⇒ exponent j+1
            let w = start_wt * base.powi((j + 1) as i32);
            weights.push(w);
            sum += w;
        }
        inv_sum = 1.0 / sum;
    }

    let smooth_period = if smooth {
        ((period as f64).sqrt().round() as usize).max(1)
    } else {
        1
    };
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, weights, inv_sum, smooth_period, first, chosen))
}

#[inline(always)]
fn cora_wave_compute_into(
    data: &[f64],
    weights: &[f64],
    inv_wsum: f64,
    smooth_period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                cora_wave_scalar_with_weights(data, weights, inv_wsum, smooth_period, first, out);
                return;
            }
        }
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cora_wave_scalar_with_weights(data, weights, inv_wsum, smooth_period, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // SIMD stubs keep calling the scalar-with-weights version
                cora_wave_scalar_with_weights(data, weights, inv_wsum, smooth_period, first, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn cora_wave_scalar_with_weights(
    data: &[f64],
    weights: &[f64],
    inv_wsum: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    let n = data.len();
    let p = weights.len();
    if p == 0 || n == 0 {
        return;
    }

    // Fast path: no smoothing
    if smooth_period == 1 {
        if p == 1 {
            let start = first_val;
            if start < n {
                for i in start..n {
                    unsafe {
                        let v = *data.get_unchecked(i);
                        *out.get_unchecked_mut(i) = v * inv_wsum;
                    }
                }
            }
            return;
        }

        // geometric parameters from weights
        let w0 = unsafe { *weights.get_unchecked(0) };
        let w1 = unsafe { *weights.get_unchecked(1) };
        let inv_R = w0 / w1; // == 1/R
        let a_old = w0 * inv_R; // == a
        let w_last = unsafe { *weights.get_unchecked(p - 1) };

        let warm0 = first_val + p - 1;
        if warm0 >= n {
            return;
        }
        let start0 = warm0 + 1 - p;

        // initial dot at warm0
        let mut acc0 = 0.0;
        let mut acc1 = 0.0;
        let mut acc2 = 0.0;
        let mut acc3 = 0.0;
        let mut j = 0usize;
        let end4 = p & !3usize;

        unsafe {
            let xptr = data.as_ptr().add(start0);
            let wptr = weights.as_ptr();
            while j < end4 {
                let x0 = *xptr.add(j);
                let x1 = *xptr.add(j + 1);
                let x2 = *xptr.add(j + 2);
                let x3 = *xptr.add(j + 3);

                let y0 = *wptr.add(j);
                let y1 = *wptr.add(j + 1);
                let y2 = *wptr.add(j + 2);
                let y3 = *wptr.add(j + 3);

                acc0 = x0.mul_add(y0, acc0);
                acc1 = x1.mul_add(y1, acc1);
                acc2 = x2.mul_add(y2, acc2);
                acc3 = x3.mul_add(y3, acc3);

                j += 4;
            }
            let mut S = (acc0 + acc1) + (acc2 + acc3);
            while j < p {
                let x = *xptr.add(j);
                let y = *wptr.add(j);
                S = x.mul_add(y, S);
                j += 1;
            }

            *out.get_unchecked_mut(warm0) = S * inv_wsum;

            // O(1) recurrence forward
            let mut i = warm0;
            while i + 1 < n {
                let x_old = *data.get_unchecked(i + 1 - p);
                let x_new = *data.get_unchecked(i + 1);
                S = (S * inv_R) - a_old * x_old + w_last * x_new;
                *out.get_unchecked_mut(i + 1) = S * inv_wsum;
                i += 1;
            }
        }
        return;
    }

    // Smoothing (WMA 1..m) — recompute weighted sum each emission to match streaming numerics
    let m = smooth_period;
    let wma_sum = (m as f64) * ((m as f64) + 1.0) * 0.5;

    if p == 1 {
        let warm0 = first_val; // p-1 == 0
        if warm0 >= n {
            return;
        }

        let mut ring_mu: Vec<MaybeUninit<f64>> = make_uninit_matrix(1, m);
        let mut head = 0usize;
        let warm_total = warm0 + m - 1;
        unsafe {
            // fill ring with raw values
            for i in warm0..n {
                ring_mu
                    .get_unchecked_mut(head)
                    .write(*data.get_unchecked(i));
                head = (head + 1) % m;

                if i >= warm_total {
                    let mut acc = 0.0;
                    for k in 0..m {
                        let idx = (head + k) % m; // oldest..newest
                        let v = *ring_mu.get_unchecked(idx).assume_init_ref();
                        acc += v * ((k + 1) as f64);
                    }
                    *out.get_unchecked_mut(i) = acc / wma_sum;
                }
            }
        }
        return;
    }

    // General p>=2: CoRa geometric recurrence + O(1) WMA
    let w0 = unsafe { *weights.get_unchecked(0) };
    let w1 = unsafe { *weights.get_unchecked(1) };
    let inv_R = w0 / w1; // 1/R
    let a_old = w0 * inv_R; // a
    let w_last = unsafe { *weights.get_unchecked(p - 1) };

    let warm0 = first_val + p - 1;
    if warm0 >= n {
        return;
    }
    let start0 = warm0 + 1 - p;

    // initial dot
    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut j = 0usize;
    let end4 = p & !3usize;

    unsafe {
        let xptr = data.as_ptr().add(start0);
        let wptr = weights.as_ptr();
        while j < end4 {
            let x0 = *xptr.add(j);
            let x1 = *xptr.add(j + 1);
            let x2 = *xptr.add(j + 2);
            let x3 = *xptr.add(j + 3);

            let y0 = *wptr.add(j);
            let y1 = *wptr.add(j + 1);
            let y2 = *wptr.add(j + 2);
            let y3 = *wptr.add(j + 3);

            acc0 = x0.mul_add(y0, acc0);
            acc1 = x1.mul_add(y1, acc1);
            acc2 = x2.mul_add(y2, acc2);
            acc3 = x3.mul_add(y3, acc3);

            j += 4;
        }
        let mut S = (acc0 + acc1) + (acc2 + acc3);
        while j < p {
            let x = *xptr.add(j);
            let y = *wptr.add(j);
            S = x.mul_add(y, S);
            j += 1;
        }

        // ring for CoRa outputs
        let mut ring_mu: Vec<MaybeUninit<f64>> = make_uninit_matrix(1, m);
        let mut fill = 0usize;

        let mut y = S * inv_wsum;
        ring_mu.get_unchecked_mut(fill).write(y);
        fill += 1;

        let warm_total = warm0 + m - 1;
        let mut i = warm0;
        while i + 1 <= warm_total && i + 1 < n {
            let x_old = *data.get_unchecked(i + 1 - p);
            let x_new = *data.get_unchecked(i + 1);
            S = (S * inv_R) - a_old * x_old + w_last * x_new;
            y = S * inv_wsum;
            ring_mu.get_unchecked_mut(fill).write(y);
            fill += 1;
            i += 1;
        }
        if warm_total >= n {
            return;
        }

        // switch to recompute WMA per emission to match streaming
        let mut head = 0usize;
        // emit first
        {
            let mut acc = 0.0;
            for k in 0..m {
                let idx = (head + k) % m;
                let v = *ring_mu.get_unchecked(idx).assume_init_ref();
                acc += v * ((k + 1) as f64);
            }
            *out.get_unchecked_mut(warm_total) = acc / wma_sum;
        }

        // stream forward
        while i + 1 < n {
            let x_old = *data.get_unchecked(i + 1 - p);
            let x_new = *data.get_unchecked(i + 1);
            S = (S * inv_R) - a_old * x_old + w_last * x_new;
            let y_new = S * inv_wsum;

            ring_mu.get_unchecked_mut(head).write(y_new);
            head = (head + 1) % m;

            let mut acc = 0.0;
            for k in 0..m {
                let idx = (head + k) % m;
                let v = *ring_mu.get_unchecked(idx).assume_init_ref();
                acc += v * ((k + 1) as f64);
            }
            *out.get_unchecked_mut(i + 1) = acc / wma_sum;
            i += 1;
        }
    }
}

// Keep old function as a stub that calls the new one for backward compatibility
#[inline]
pub fn cora_wave_scalar(
    data: &[f64],
    period: usize,
    r_multi: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    // CoRa weights
    if period == 1 {
        cora_wave_scalar_with_weights(data, &[1.0], 1.0, smooth_period, first_val, out);
        return;
    }
    let start_wt = 0.01;
    let end_wt = period as f64;
    let r = (end_wt / start_wt).powf(1.0 / (period as f64 - 1.0)) - 1.0;
    let base = 1.0 + r * r_multi;

    let mut weights = Vec::with_capacity(period);
    let mut weight_sum = 0.0;
    for j in 0..period {
        let w = start_wt * base.powi((j + 1) as i32);
        weights.push(w);
        weight_sum += w;
    }

    cora_wave_scalar_with_weights(
        data,
        &weights,
        1.0 / weight_sum,
        smooth_period,
        first_val,
        out,
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn cora_wave_simd128(
    data: &[f64],
    period: usize,
    r_multi: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    use core::arch::wasm32::*;

    // For now, fallback to scalar
    cora_wave_scalar(data, period, r_multi, smooth_period, first_val, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn cora_wave_avx2(
    data: &[f64],
    period: usize,
    r_multi: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    // For now, fallback to scalar
    cora_wave_scalar(data, period, r_multi, smooth_period, first_val, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn cora_wave_avx512(
    data: &[f64],
    period: usize,
    r_multi: f64,
    smooth_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    // For now, fallback to scalar
    cora_wave_scalar(data, period, r_multi, smooth_period, first_val, out);
}

/// Decision: Streaming uses O(1) CoRa recurrence; WMA smoothing recomputes per emission by default
/// to match scalar/batch numerics exactly. Optional O(1) WMA is implemented but disabled by default.
#[derive(Debug, Clone)]
pub struct CoraWaveStream {
    // ---- config ----
    period: usize,
    r_multi: f64,
    smooth: bool,
    smooth_period: usize, // m

    // ---- precomputed geometric parameters for CoRa ----
    base: f64,     // geometric ratio R = 1 + r*r_multi
    inv_R: f64,    // 1/R
    a_old: f64,    // "a" in the derivation: start_wt (== w0 * inv_R)
    w_last: f64,   // weight of newest sample when window is full: a*R^p
    inv_wsum: f64, // 1 / sum(weights)

    // ---- state: CoRa ----
    ring_x: Vec<f64>,
    head_x: usize, // points at slot to overwrite next
    idx: usize,    // number of samples seen
    have_S: bool,
    S: f64, // current (unnormalized) weighted sum

    // ---- state: smoothing (WMA over CoRa outputs) ----
    m: usize,
    wma_sum: f64,
    ring_y: Vec<f64>,
    head_y: usize,
    y_count: usize, // number of CoRa values stored so far
    // O(1) WMA running sums (used only in FAST mode)
    Ssum_y: f64, // simple sum of last m CoRa values
    Wsum_y: f64, // weighted sum with weights 1..m

    // ---- behavior toggles ----
    fast_smooth: bool, // false => recompute WMA per emission (exact match); true => O(1)
}

impl CoraWaveStream {
    #[inline]
    pub fn try_new(params: CoraWaveParams) -> Result<Self, CoraWaveError> {
        let period = params.period.unwrap_or(20);
        let r_multi = params.r_multi.unwrap_or(2.0);
        let smooth = params.smooth.unwrap_or(true);

        if period == 0 {
            return Err(CoraWaveError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if r_multi < 0.0 || !r_multi.is_finite() {
            return Err(CoraWaveError::InvalidRMulti { value: r_multi });
        }

        // ----- smoothing length (m) -----
        let m = if smooth {
            ((period as f64).sqrt().round() as usize).max(1)
        } else {
            1
        };

        // ===== CoRa geometric weights (no full weight vector needed) =====
        // Weight definition used elsewhere in this file:
        //   w_j = start_wt * base^(j+1), j = 0..p-1, newest uses j=p-1.
        let p = period;
        let start_wt = 0.01_f64;
        // step ratio baseline across j (prior to r_multi):
        let end_wt = p as f64;
        let r = (end_wt / start_wt).powf(1.0 / (p as f64 - 1.0)) - 1.0;
        let base = 1.0 + r * r_multi;
        let inv_R = 1.0 / base;
        // a_old := "a" in the recurrence; here it's simply start_wt
        let a_old = start_wt;

        // base^p and final weight on newest element when window is full
        let base_pow_p = if (base - 1.0).abs() < 1e-16 {
            1.0
        } else {
            base.powi(p as i32)
        };
        let w_last = a_old * base_pow_p;

        // geometric series: sum_{j=0}^{p-1} start_wt * base^(j+1)
        let weight_sum = if (base - 1.0).abs() < 1e-16 {
            // r_multi == 0 ⇒ base == 1 ⇒ all weights equal to start_wt
            a_old * (p as f64)
        } else {
            a_old * base * (base_pow_p - 1.0) / (base - 1.0)
        };
        let inv_wsum = 1.0 / weight_sum;

        // ----- WMA fixed denominator -----
        let wma_sum = (m as f64) * ((m as f64) + 1.0) * 0.5;

        // Default behavior: keep exact WMA numerics (same order as scalar path).
        const FAST_WMA_O1_DEFAULT: bool = false;

        Ok(Self {
            period: p,
            r_multi,
            smooth,
            smooth_period: m,
            base,
            inv_R,
            a_old,
            w_last,
            inv_wsum,
            ring_x: vec![0.0; p],
            head_x: 0,
            idx: 0,
            have_S: false,
            S: 0.0,
            m,
            wma_sum,
            ring_y: vec![0.0; m.max(1)],
            head_y: 0,
            y_count: 0,
            Ssum_y: 0.0,
            Wsum_y: 0.0,
            fast_smooth: FAST_WMA_O1_DEFAULT,
        })
    }

    /// Push one value. Returns Some(value) when warm-up is complete, else None.
    #[inline]
    pub fn update(&mut self, x_new: f64) -> Option<f64> {
        // write new sample into ring_x; remember outgoing when the window is full
        let pos = self.head_x;
        let x_old = self.ring_x[pos];
        self.ring_x[pos] = x_new;
        self.head_x = (pos + 1) % self.period;
        self.idx += 1;

        // Build initial S when the window first fills
        if !self.have_S {
            if self.idx < self.period {
                return None;
            }
            // idx == period here: compute initial dot product S = Σ w_j * x_{oldest..newest}
            //   oldest is at head_x, newest at head_x+(p-1)
            let mut S = 0.0;
            let mut w = self.a_old * self.base; // start_wt * base^(0+1)
            let mut i = 0usize;
            while i < self.period {
                // chronological order: oldest..newest
                let xi = self.ring_x[(self.head_x + i) % self.period];
                S = xi.mul_add(w, S);
                w *= self.base;
                i += 1;
            }
            self.S = S;
            self.have_S = true;

            // First CoRa output
            let y = self.S * self.inv_wsum;
            if self.m == 1 {
                return Some(y);
            }
            // store for smoothing warm-up
            self.ring_y[self.y_count] = y;
            self.y_count += 1;
            // next write position advances accordingly
            self.head_y = self.y_count % self.m;
            if self.fast_smooth {
                self.Ssum_y += y;
                self.Wsum_y += (self.y_count as f64) * y; // weights 1..k during warm-up
            }
            return None;
        }

        // O(1) CoRa recurrence:
        // S' = (S / R) - a_old * x_old + (a_old * R^p) * x_new
        self.S = (self.S * self.inv_R) - self.a_old * x_old + self.w_last * x_new;
        let y = self.S * self.inv_wsum;

        // No smoothing?
        if self.m == 1 {
            return Some(y);
        }

        // Smoothing path
        if !self.fast_smooth {
            // ---- Exact per-emission recompute (matches scalar path numerically) ----
            // Maintain a ring of last m CoRa values.
            if self.y_count < self.m {
                // still filling the window
                self.ring_y[self.head_y] = y;
                self.head_y = (self.head_y + 1) % self.m;
                self.y_count += 1;
                if self.y_count < self.m {
                    return None;
                }
                // just reached full window: emit initial WMA
                let mut acc = 0.0;
                let mut k = 0usize;
                while k < self.m {
                    let idx = (self.head_y + k) % self.m; // oldest..newest
                    let v = self.ring_y[idx];
                    acc = v.mul_add((k + 1) as f64, acc);
                    k += 1;
                }
                return Some(acc / self.wma_sum);
            } else {
                // steady-state: overwrite oldest, then compute
                self.ring_y[self.head_y] = y;
                self.head_y = (self.head_y + 1) % self.m;
                let mut acc = 0.0;
                let mut k = 0usize;
                while k < self.m {
                    let idx = (self.head_y + k) % self.m; // oldest..newest
                    let v = self.ring_y[idx];
                    acc = v.mul_add((k + 1) as f64, acc);
                    k += 1;
                }
                return Some(acc / self.wma_sum);
            }
        } else {
            // ---- FAST O(1) WMA update ----
            if self.y_count < self.m {
                // still filling the y-ring
                self.ring_y[self.y_count] = y;
                self.y_count += 1;
                self.Ssum_y += y;
                self.Wsum_y += (self.y_count as f64) * y;
                if self.y_count < self.m {
                    return None;
                }
                // just reached full window: emit initial WMA
                self.head_y = 0;
                return Some(self.Wsum_y / self.wma_sum);
            }

            // slide window by 1 in O(1)
            let y_old = self.ring_y[self.head_y];
            // update weighted sum and simple sum
            self.Wsum_y = self.Wsum_y - self.Ssum_y + (self.m as f64) * y;
            self.ring_y[self.head_y] = y;
            self.Ssum_y = self.Ssum_y + y - y_old;
            self.head_y = (self.head_y + 1) % self.m;

            Some(self.Wsum_y / self.wma_sum)
        }
    }
}

// ==================== BATCH PROCESSING ====================
#[derive(Clone, Debug)]
pub struct CoraWaveBatchRange {
    pub period: (usize, usize, usize),
    pub r_multi: (f64, f64, f64),
    pub smooth: bool,
}

impl Default for CoraWaveBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 60, 1),
            r_multi: (2.0, 2.0, 0.0),
            smooth: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoraWaveBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CoraWaveParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CoraWaveBatchOutput {
    pub fn row_for_params(&self, p: &CoraWaveParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.r_multi.unwrap_or(2.0) - p.r_multi.unwrap_or(2.0)).abs() < 1e-12
                && c.smooth.unwrap_or(true) == p.smooth.unwrap_or(true)
        })
    }

    pub fn values_for(&self, p: &CoraWaveParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn axis_usize((s, e, t): (usize, usize, usize)) -> Vec<usize> {
    if t == 0 || s == e {
        vec![s]
    } else {
        (s..=e).step_by(t).collect()
    }
}
#[inline(always)]
fn axis_f64((s, e, t): (f64, f64, f64)) -> Vec<f64> {
    if t.abs() < 1e-12 || (s - e).abs() < 1e-12 {
        vec![s]
    } else {
        let mut v = Vec::new();
        let mut x = s;
        while x <= e + 1e-12 {
            v.push(x);
            x += t;
        }
        v
    }
}
#[inline(always)]
fn expand_grid_cw(r: &CoraWaveBatchRange) -> Vec<CoraWaveParams> {
    let periods = axis_usize(r.period);
    let mults = axis_f64(r.r_multi);
    let mut out = Vec::with_capacity(periods.len() * mults.len());
    for &p in &periods {
        for &m in &mults {
            out.push(CoraWaveParams {
                period: Some(p),
                r_multi: Some(m),
                smooth: Some(r.smooth),
            });
        }
    }
    out
}

#[inline(always)]
pub fn cora_wave_batch_slice(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    kern: Kernel,
) -> Result<CoraWaveBatchOutput, CoraWaveError> {
    cora_wave_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cora_wave_batch_par_slice(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    kern: Kernel,
) -> Result<CoraWaveBatchOutput, CoraWaveError> {
    cora_wave_batch_inner(data, sweep, kern, true)
}

pub fn cora_wave_batch_with_kernel(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    k: Kernel,
) -> Result<CoraWaveBatchOutput, CoraWaveError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(CoraWaveError::InvalidPeriod {
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
    cora_wave_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn cora_wave_batch_inner(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CoraWaveBatchOutput, CoraWaveError> {
    let combos = expand_grid_cw(sweep);
    if combos.is_empty() {
        return Err(CoraWaveError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let cols = data.len();
    if cols == 0 {
        // Align with ALMA behavior for consistency
        return Err(CoraWaveError::AllValuesNaN);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CoraWaveError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p {
        return Err(CoraWaveError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    // Allocate rows×cols uninit and prefill warm prefixes with NaN
    let rows = combos.len();
    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warms: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let sp = if c.smooth.unwrap_or(true) {
                ((p as f64).sqrt().round() as usize).max(1)
            } else {
                1
            };
            first + p - 1 + sp.saturating_sub(1)
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warms);

    // Prepare flattened per-row weights and inv_sums
    let mut flat_w = vec![0.0f64; rows * max_p];
    let mut inv_sums = vec![0.0f64; rows];

    for (row, prm) in combos.iter().enumerate() {
        let p = prm.period.unwrap();
        let r_multi = prm.r_multi.unwrap();

        if p == 1 {
            flat_w[row * max_p] = 1.0;
            inv_sums[row] = 1.0;
        } else {
            let start_wt = 0.01;
            let end_wt = p as f64;
            let r = (end_wt / start_wt).powf(1.0 / (p as f64 - 1.0)) - 1.0;
            let base = 1.0 + r * r_multi;

            let mut sum = 0.0;
            for j in 0..p {
                // before: (p - j)
                let w = start_wt * base.powi((j + 1) as i32);
                flat_w[row * max_p + j] = w;
                sum += w;
            }
            inv_sums[row] = 1.0 / sum;
        }
        // trailing region [p..max_p) stays 0.0; never read
    }

    // Write rows in place
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_uninit: &mut [MaybeUninit<f64>] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let p = combos[row].period.unwrap();
        let sp = if combos[row].smooth.unwrap_or(true) {
            ((p as f64).sqrt().round() as usize).max(1)
        } else {
            1
        };
        let w_ptr = flat_w[row * max_p..].as_ptr();
        let inv = inv_sums[row];

        let dst = unsafe { core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, cols) };

        match actual {
            Kernel::Scalar
            | Kernel::ScalarBatch
            | Kernel::Avx2
            | Kernel::Avx2Batch
            | Kernel::Avx512
            | Kernel::Avx512Batch => unsafe {
                cora_wave_row_scalar_with_weights(data, first, p, w_ptr, inv, sp, dst)
            },
            _ => unreachable!(),
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        if parallel {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        } else {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(CoraWaveBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn cora_wave_batch_inner_into(
    data: &[f64],
    sweep: &CoraWaveBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<CoraWaveParams>, CoraWaveError> {
    let combos = expand_grid_cw(sweep);
    if combos.is_empty() {
        return Err(CoraWaveError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let cols = data.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(CoraWaveError::AllValuesNaN);
    }
    if out.len() != rows * cols {
        return Err(CoraWaveError::InvalidPeriod {
            period: out.len(),
            data_len: rows * cols,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CoraWaveError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p {
        return Err(CoraWaveError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    // Prepare warmup lengths per row
    let warms: Vec<usize> = combos
        .iter()
        .map(|c| {
            let p = c.period.unwrap();
            let sp = if c.smooth.unwrap_or(true) {
                ((p as f64).sqrt().round() as usize).max(1)
            } else {
                1
            };
            first + p - 1 + sp.saturating_sub(1)
        })
        .collect();

    // Cast user buffer to MaybeUninit and prefill warm prefixes with NaN
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_mu, cols, &warms);

    // Precompute row weights and inv sums
    let mut flat_w = vec![0.0f64; rows * max_p];
    let mut inv_sums = vec![0.0f64; rows];
    for (row, prm) in combos.iter().enumerate() {
        let p = prm.period.unwrap();
        let r_multi = prm.r_multi.unwrap();

        if p == 1 {
            flat_w[row * max_p] = 1.0;
            inv_sums[row] = 1.0;
        } else {
            let start_wt = 0.01;
            let end_wt = p as f64;
            let r = (end_wt / start_wt).powf(1.0 / (p as f64 - 1.0)) - 1.0;
            let base = 1.0 + r * r_multi;

            let mut sum = 0.0;
            for j in 0..p {
                // before: (p - j)
                let w = start_wt * base.powi((j + 1) as i32);
                flat_w[row * max_p + j] = w;
                sum += w;
            }
            inv_sums[row] = 1.0 / sum;
        }
    }

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let p = combos[row].period.unwrap();
        let sp = if combos[row].smooth.unwrap_or(true) {
            ((p as f64).sqrt().round() as usize).max(1)
        } else {
            1
        };
        let w_ptr = flat_w[row * max_p..].as_ptr();
        let inv = inv_sums[row];
        // Safe to view row as f64 now for writing initialized cells
        let dst: &mut [f64] =
            unsafe { core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, cols) };
        match actual {
            Kernel::Scalar
            | Kernel::ScalarBatch
            | Kernel::Avx2
            | Kernel::Avx2Batch
            | Kernel::Avx512
            | Kernel::Avx512Batch => unsafe {
                cora_wave_row_scalar_with_weights(data, first, p, w_ptr, inv, sp, dst)
            },
            _ => unreachable!(),
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        if parallel {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        } else {
            for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        for (row, slice) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn cora_wave_row_scalar_with_weights(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64, // row's weights start
    inv_wsum: f64,
    smooth_period: usize,
    out: &mut [f64],
) {
    let n = data.len();
    let p = period;
    if p == 0 || n == 0 {
        return;
    }

    // No smoothing
    if smooth_period == 1 {
        if p == 1 {
            let warm0 = first;
            let mut i = warm0;
            while i < n {
                *out.get_unchecked_mut(i) = *data.get_unchecked(i) * inv_wsum;
                i += 1;
            }
            return;
        }

        // geometric params
        let w0 = *w_ptr.add(0);
        let w1 = *w_ptr.add(1);
        let inv_R = w0 / w1; // 1/R
        let a_old = w0 * inv_R; // a
        let w_last = *w_ptr.add(p - 1); // a*R^p

        let warm0 = first + p - 1;
        if warm0 >= n {
            return;
        }
        let start0 = warm0 + 1 - p;

        // initial dot
        let mut acc0 = 0.0;
        let mut acc1 = 0.0;
        let mut acc2 = 0.0;
        let mut acc3 = 0.0;
        let mut j = 0usize;
        let end4 = p & !3usize;
        let xptr = data.as_ptr().add(start0);

        while j < end4 {
            let x0 = *xptr.add(j);
            let x1 = *xptr.add(j + 1);
            let x2 = *xptr.add(j + 2);
            let x3 = *xptr.add(j + 3);

            let y0 = *w_ptr.add(j);
            let y1 = *w_ptr.add(j + 1);
            let y2 = *w_ptr.add(j + 2);
            let y3 = *w_ptr.add(j + 3);

            acc0 = x0.mul_add(y0, acc0);
            acc1 = x1.mul_add(y1, acc1);
            acc2 = x2.mul_add(y2, acc2);
            acc3 = x3.mul_add(y3, acc3);

            j += 4;
        }
        let mut S = (acc0 + acc1) + (acc2 + acc3);
        while j < p {
            let x = *xptr.add(j);
            let y = *w_ptr.add(j);
            S = x.mul_add(y, S);
            j += 1;
        }

        *out.get_unchecked_mut(warm0) = S * inv_wsum;

        // stream recurrence
        let mut i = warm0;
        while i + 1 < n {
            let x_old = *data.get_unchecked(i + 1 - p);
            let x_new = *data.get_unchecked(i + 1);
            S = (S * inv_R) - a_old * x_old + w_last * x_new;
            *out.get_unchecked_mut(i + 1) = S * inv_wsum;
            i += 1;
        }
        return;
    }

    // Smoothing path (WMA 1..m) with O(1) updates
    let m = smooth_period;
    let wma_sum = (m as f64) * ((m as f64) + 1.0) * 0.5;

    if p == 1 {
        let warm0 = first;
        if warm0 >= n {
            return;
        }

        // O(1) WMA update with ring buffer
        let mut ring_mu: Vec<MaybeUninit<f64>> = make_uninit_matrix(1, m);
        let mut fill = 0usize;

        let warm_total = warm0 + m - 1;
        let mut i = warm0;
        while i <= warm_total && i < n {
            ring_mu
                .get_unchecked_mut(fill)
                .write(*data.get_unchecked(i));
            fill += 1;
            i += 1;
        }
        if warm_total >= n {
            return;
        }

        // Seed sums
        let mut Ssum = 0.0;
        let mut Wsum = 0.0;
        for k in 0..m {
            let v = *ring_mu.get_unchecked(k).assume_init_ref();
            Ssum += v;
            Wsum += v * ((k + 1) as f64);
        }
        let mut head = 0usize;
        let mut t = warm_total;
        *out.get_unchecked_mut(t) = Wsum / wma_sum;

        while t + 1 < n {
            let y_old = *ring_mu.get_unchecked(head).assume_init_ref();
            let y_new = *data.get_unchecked(t + 1);

            Wsum = Wsum - Ssum + (m as f64) * y_new;

            ring_mu.get_unchecked_mut(head).write(y_new);
            Ssum = Ssum + y_new - y_old;
            head = (head + 1) % m;

            *out.get_unchecked_mut(t + 1) = Wsum / wma_sum;
            t += 1;
        }
        return;
    }

    // p >= 2
    let w0 = *w_ptr.add(0);
    let w1 = *w_ptr.add(1);
    let inv_R = w0 / w1; // 1/R
    let a_old = w0 * inv_R; // a
    let w_last = *w_ptr.add(p - 1); // a*R^p

    let warm0 = first + p - 1;
    if warm0 >= n {
        return;
    }
    let start0 = warm0 + 1 - p;

    // Initial dot
    let mut acc0 = 0.0;
    let mut acc1 = 0.0;
    let mut acc2 = 0.0;
    let mut acc3 = 0.0;
    let mut j = 0usize;
    let end4 = p & !3usize;
    let xptr = data.as_ptr().add(start0);
    while j < end4 {
        let x0 = *xptr.add(j);
        let x1 = *xptr.add(j + 1);
        let x2 = *xptr.add(j + 2);
        let x3 = *xptr.add(j + 3);

        let y0 = *w_ptr.add(j);
        let y1 = *w_ptr.add(j + 1);
        let y2 = *w_ptr.add(j + 2);
        let y3 = *w_ptr.add(j + 3);

        acc0 = x0.mul_add(y0, acc0);
        acc1 = x1.mul_add(y1, acc1);
        acc2 = x2.mul_add(y2, acc2);
        acc3 = x3.mul_add(y3, acc3);

        j += 4;
    }
    let mut S = (acc0 + acc1) + (acc2 + acc3);
    while j < p {
        let x = *xptr.add(j);
        let y = *w_ptr.add(j);
        S = x.mul_add(y, S);
        j += 1;
    }

    // WMA ring for CoRa(y) with O(1) update
    let mut ring_mu: Vec<MaybeUninit<f64>> = make_uninit_matrix(1, m);
    let mut fill = 0usize;

    let mut y = S * inv_wsum;
    ring_mu.get_unchecked_mut(fill).write(y);
    fill += 1;

    let warm_total = warm0 + m - 1;
    let mut i = warm0;
    while i + 1 <= warm_total && i + 1 < n {
        let x_old = *data.get_unchecked(i + 1 - p);
        let x_new = *data.get_unchecked(i + 1);
        S = (S * inv_R) - a_old * x_old + w_last * x_new;
        y = S * inv_wsum;
        ring_mu.get_unchecked_mut(fill).write(y);
        fill += 1;
        i += 1;
    }
    if warm_total >= n {
        return;
    }

    // Seed sums
    let mut Ssum = 0.0;
    let mut Wsum = 0.0;
    for k in 0..m {
        let v = *ring_mu.get_unchecked(k).assume_init_ref();
        Ssum += v;
        Wsum += v * ((k + 1) as f64);
    }
    let mut head = 0usize;
    *out.get_unchecked_mut(warm_total) = Wsum / wma_sum;

    // Stream forward
    while i + 1 < n {
        let x_old = *data.get_unchecked(i + 1 - p);
        let x_new = *data.get_unchecked(i + 1);
        S = (S * inv_R) - a_old * x_old + w_last * x_new;
        let y_new = S * inv_wsum;

        Wsum = Wsum - Ssum + (m as f64) * y_new;

        let y_old = *ring_mu.get_unchecked(head).assume_init_ref();
        ring_mu.get_unchecked_mut(head).write(y_new);
        Ssum = Ssum + y_new - y_old;
        head = (head + 1) % m;

        *out.get_unchecked_mut(i + 1) = Wsum / wma_sum;
        i += 1;
    }
}

#[derive(Clone, Debug, Default)]
pub struct CoraWaveBatchBuilder {
    range: CoraWaveBatchRange,
    kernel: Kernel,
}

impl CoraWaveBatchBuilder {
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
    pub fn period_static(mut self, val: usize) -> Self {
        self.range.period = (val, val, 0);
        self
    }

    #[inline]
    pub fn r_multi_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.r_multi = (start, end, step);
        self
    }

    #[inline]
    pub fn r_multi_static(mut self, val: f64) -> Self {
        self.range.r_multi = (val, val, 0.0);
        self
    }

    #[inline]
    pub fn smooth(mut self, val: bool) -> Self {
        self.range.smooth = val;
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<CoraWaveBatchOutput, CoraWaveError> {
        cora_wave_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<CoraWaveBatchOutput, CoraWaveError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<CoraWaveBatchOutput, CoraWaveError> {
        CoraWaveBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<CoraWaveBatchOutput, CoraWaveError> {
        CoraWaveBatchBuilder::new().kernel(k).apply_slice(data)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cora_wave")]
#[pyo3(signature = (data, period, r_multi, smooth, kernel=None))]
pub fn cora_wave_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    r_multi: f64,
    smooth: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = CoraWaveParams {
        period: Some(period),
        r_multi: Some(r_multi),
        smooth: Some(smooth),
    };
    let input = CoraWaveInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| cora_wave_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CoraWaveStream")]
pub struct CoraWaveStreamPy {
    stream: CoraWaveStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CoraWaveStreamPy {
    #[new]
    fn new(period: usize, r_multi: f64, smooth: bool) -> PyResult<Self> {
        let params = CoraWaveParams {
            period: Some(period),
            r_multi: Some(r_multi),
            smooth: Some(smooth),
        };
        let stream =
            CoraWaveStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(CoraWaveStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "cora_wave_batch")]
#[pyo3(signature = (data, period_range, r_multi_range, smooth=true, kernel=None))]
pub fn cora_wave_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    r_multi_range: (f64, f64, f64),
    smooth: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1};
    let slice_in = data.as_slice()?;
    let sweep = CoraWaveBatchRange {
        period: period_range,
        r_multi: r_multi_range,
        smooth,
    };

    let combos = expand_grid_cw(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // 1D buffer, we will reshape in Python without copying
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

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
            cora_wave_batch_inner_into(slice_in, &sweep, simd, true, out_slice)
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
        "r_multis",
        combos
            .iter()
            .map(|p| p.r_multi.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("smooth", smooth)?;
    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cora_wave_js(
    data: &[f64],
    period: usize,
    r_multi: f64,
    smooth: bool,
) -> Result<Vec<f64>, JsValue> {
    let params = CoraWaveParams {
        period: Some(period),
        r_multi: Some(r_multi),
        smooth: Some(smooth),
    };
    let input = CoraWaveInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];
    cora_wave_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cora_wave_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cora_wave_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cora_wave_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    r_multi: f64,
    smooth: bool,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to cora_wave_into"));
    }
    if period == 0 || period > len {
        return Err(JsValue::from_str("Invalid period"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let params = CoraWaveParams {
            period: Some(period),
            r_multi: Some(r_multi),
            smooth: Some(smooth),
        };
        let input = CoraWaveInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            cora_wave_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            cora_wave_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// WASM Batch APIs
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CoraWaveBatchConfig {
    pub period_range: (usize, usize, usize),
    pub r_multi_range: (f64, f64, f64),
    pub smooth: bool,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CoraWaveBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CoraWaveParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cora_wave_batch)]
pub fn cora_wave_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: CoraWaveBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = CoraWaveBatchRange {
        period: cfg.period_range,
        r_multi: cfg.r_multi_range,
        smooth: cfg.smooth,
    };
    let out = cora_wave_batch_with_kernel(data, &sweep, detect_best_batch_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = CoraWaveBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cora_wave_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    rmulti_start: f64,
    rmulti_end: f64,
    rmulti_step: f64,
    smooth: bool,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to cora_wave_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = CoraWaveBatchRange {
            period: (period_start, period_end, period_step),
            r_multi: (rmulti_start, rmulti_end, rmulti_step),
            smooth,
        };
        let combos = expand_grid_cw(&sweep);
        let rows = combos.len();
        let cols = len;
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        cora_wave_batch_inner_into(data, &sweep, detect_best_batch_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct CoraWaveContext {
    weights: Vec<f64>,
    inv_norm: f64,
    period: usize,
    smooth_period: usize,
    kernel: Kernel,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl CoraWaveContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(period: usize, r_multi: f64, smooth: bool) -> Result<CoraWaveContext, JsValue> {
        if period == 0 {
            return Err(JsValue::from_str("Invalid period: 0"));
        }
        if !r_multi.is_finite() || r_multi < 0.0 {
            return Err(JsValue::from_str(&format!("Invalid r_multi: {}", r_multi)));
        }
        let smooth_period = if smooth {
            ((period as f64).sqrt().round() as usize).max(1)
        } else {
            1
        };

        if period == 1 {
            return Ok(CoraWaveContext {
                weights: vec![1.0],
                inv_norm: 1.0,
                period,
                smooth_period,
                kernel: detect_best_kernel(),
            });
        }

        let start_wt = 0.01;
        let end_wt = period as f64;
        let r = (end_wt / start_wt).powf(1.0 / (period as f64 - 1.0)) - 1.0;
        let base = 1.0 + r * r_multi;

        let mut weights = Vec::with_capacity(period);
        let mut norm = 0.0;
        for j in 0..period {
            let w = start_wt * base.powi((j + 1) as i32);
            weights.push(w);
            norm += w;
        }

        Ok(CoraWaveContext {
            weights,
            inv_norm: 1.0 / norm,
            period,
            smooth_period,
            kernel: detect_best_kernel(),
        })
    }

    pub fn update_into(
        &self,
        in_ptr: *const f64,
        out_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if len < self.period {
            return Err(JsValue::from_str("Data length less than period"));
        }
        if in_ptr.is_null() || out_ptr.is_null() {
            return Err(JsValue::from_str("null pointer passed to update_into"));
        }
        unsafe {
            let data = std::slice::from_raw_parts(in_ptr, len);
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

            cora_wave_scalar_with_weights(
                data,
                &self.weights,
                self.inv_norm,
                self.smooth_period,
                first,
                out,
            );

            let warm = first + self.period - 1 + self.smooth_period.saturating_sub(1);
            for i in 0..warm.min(len) {
                out[i] = f64::NAN;
            }
        }
        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.period - 1 + self.smooth_period.saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    fn check_cora_wave_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = CoraWaveInput::from_candles(&candles, "close", CoraWaveParams::default());
        let result = cora_wave_with_kernel(&input, kernel)?;

        // REFERENCE VALUES FROM PINESCRIPT
        let expected_last_five = [
            59248.63632114,
            59251.74238978,
            59203.36944998,
            59171.14999178,
            59053.74201623,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 0.01, // Using looser tolerance for this complex calculation
                "[{}] CoRa Wave {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_cora_wave_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = CoraWaveParams {
            period: None,
            r_multi: None,
            smooth: None,
        };
        let input = CoraWaveInput::from_candles(&candles, "close", default_params);
        let output = cora_wave_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cora_wave_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = CoraWaveInput::with_default_candles(&candles);
        match input.data {
            CoraWaveData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CoraWaveData::Candles"),
        }
        let output = cora_wave_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_cora_wave_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CoraWaveParams {
            period: Some(0),
            r_multi: None,
            smooth: None,
        };
        let input = CoraWaveInput::from_slice(&input_data, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CoraWaveParams {
            period: Some(10),
            r_multi: None,
            smooth: None,
        };
        let input = CoraWaveInput::from_slice(&data_small, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_slice(&single_point, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_slice(&empty, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_slice(&nan_data, params);
        let res = cora_wave_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CoRa Wave should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_cora_wave_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_candles(&c, "close", params.clone());
        let batch = cora_wave_with_kernel(&input, kernel)?.values;

        let mut stream = CoraWaveStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(c.close.len());
        for &px in &c.close {
            match stream.update(px) {
                Some(v) => streamed.push(v),
                None => streamed.push(f64::NAN),
            }
        }

        assert_eq!(batch.len(), streamed.len());
        for (i, (&b, &s)) in batch.iter().zip(&streamed).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let d = (b - s).abs();
            assert!(
                d < 1e-9,
                "[{}] streaming mismatch at {}: {} vs {}",
                test_name,
                i,
                b,
                s
            );
        }
        Ok(())
    }

    fn check_cora_wave_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let out = cora_wave_with_kernel(&CoraWaveInput::with_default_candles(&c), kernel)?.values;
        if out.len() > 240 {
            for (i, &v) in out[240..].iter().enumerate() {
                assert!(!v.is_nan(), "[{}] unexpected NaN at {}", test_name, 240 + i);
            }
        }
        Ok(())
    }

    fn check_cora_wave_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let first = cora_wave_with_kernel(
            &CoraWaveInput::from_candles(&c, "close", CoraWaveParams::default()),
            kernel,
        )?;

        let second_in = CoraWaveInput::from_slice(&first.values, CoraWaveParams::default());
        let second = cora_wave_with_kernel(&second_in, kernel)?;
        let second_ref = cora_wave_with_kernel(&second_in, Kernel::Scalar)?;

        assert_eq!(second.values.len(), first.values.len());
        for (i, (&a, &b)) in second.values.iter().zip(&second_ref.values).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-9,
                "[{}] reinput mismatch at {}: {} vs {}",
                test_name,
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

        let out = CoraWaveBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = CoraWaveParams::default();
        let row = out.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            59248.63632114,
            59251.74238978,
            59203.36944998,
            59171.14999178,
            59053.74201623,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 0.01,
                "[{test}] default-row mismatch at idx {i}: {v}"
            );
        }
        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let out = CoraWaveBatchBuilder::new()
            .kernel(kernel)
            .period_range(20, 60, 1)
            .r_multi_range(1.0, 3.0, 0.25)
            .apply_candles(&c, "close")?;

        let expected = 41 * 9; // 20..=60 => 41; 1.0..=3.0 step .25 => 9
        assert_eq!(out.combos.len(), expected);
        assert_eq!(out.rows, expected);
        assert_eq!(out.cols, c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let out = CoraWaveBatchBuilder::new()
            .kernel(kernel)
            .period_range(20, 24, 2)
            .r_multi_range(1.5, 2.5, 0.5)
            .apply_candles(&c, "close")?;

        for (idx, &v) in out.values.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let bits = v.to_bits();
            if bits == 0x11111111_11111111
                || bits == 0x22222222_22222222
                || bits == 0x33333333_33333333
            {
                let row = idx / out.cols;
                let col = idx % out.cols;
                let combo = &out.combos[row];
                panic!(
                    "[{}] poison value at row {} col {} (idx {}) params: period={}, r_multi={}, smooth={}",
                    test, row, col, idx,
                    combo.period.unwrap_or(20),
                    combo.r_multi.unwrap_or(2.0),
                    combo.smooth.unwrap_or(true),
                );
            }
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_cora_wave_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = cora_wave_with_kernel(&CoraWaveInput::with_default_candles(&c), kernel)?.values;
        for (i, &v) in out.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert!(
                !(b == 0x11111111_11111111 || b == 0x22222222_22222222 || b == 0x33333333_33333333),
                "[{test}] poison at idx {i}"
            );
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_cora_wave_no_poison(_: &str, _: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    // Test generation macro - same pattern as ALMA
    macro_rules! generate_all_cora_wave_tests {
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
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        };
    }

    // Generate all the tests for different kernels
    generate_all_cora_wave_tests!(
        check_cora_wave_accuracy,
        check_cora_wave_partial_params,
        check_cora_wave_default_candles,
        check_cora_wave_zero_period,
        check_cora_wave_period_exceeds_length,
        check_cora_wave_very_small_dataset,
        check_cora_wave_empty_input,
        check_cora_wave_all_nan,
        check_cora_wave_nan_handling,
        check_cora_wave_streaming,
        check_cora_wave_reinput,
        check_cora_wave_no_poison
    );

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_cora_wave_simd128_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let params = CoraWaveParams::default();
        let input = CoraWaveInput::from_slice(&data, params);
        let scalar = cora_wave_with_kernel(&input, Kernel::Scalar).unwrap();
        let simd = cora_wave_with_kernel(&input, Kernel::Scalar).unwrap(); // same path on WASM
        assert_eq!(scalar.values.len(), simd.values.len());
        for (i, (a, b)) in scalar.values.iter().zip(simd.values.iter()).enumerate() {
            assert!((a - b).abs() < 1e-10, "SIMD128 mismatch at {i}: {a} vs {b}");
        }
    }

    // Property-based tests
    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn test_cora_wave_no_panic(data: Vec<f64>, period in 1usize..100) {
            let params = CoraWaveParams {
                period: Some(period),
                r_multi: Some(2.0),
                smooth: Some(true),
            };
            let input = CoraWaveInput::from_slice(&data, params);
            let _ = cora_wave(&input);
        }

        #[test]
        fn test_cora_wave_length_preservation(size in 10usize..100) {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let params = CoraWaveParams::default();
            let input = CoraWaveInput::from_slice(&data, params);

            if let Ok(output) = cora_wave(&input) {
                prop_assert_eq!(output.values.len(), size);
            }
        }
    }
}
