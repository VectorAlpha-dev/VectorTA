//! # Laguerre RSI (LRSI)
//!
//! Momentum oscillator using Laguerre filter, similar to RSI but with different
//! responsiveness and smoothness characteristics.
//!
//! ## Parameters
//! - **alpha**: Smoothing factor between 0 and 1 (default: 0.2)
//!
//! ## Returns
//! - **`Ok(LrsiOutput)`** on success (`values: Vec<f64>` of length matching input)
//! - **`Err(LrsiError)`** on failure
//!
//! ## Developer Status
//! - **SIMD Kernels**: AVX2/AVX512 delegate to scalar (IIR dependency across time prevents lane speedup)
//! - **CUDA**: Wrapper mirrors scalar/streaming semantics; FP32 GPU paths are validated against f64 CPU within loose tolerances.
//! - **Python**: CUDA handles use shared DeviceArrayF32Py (CAI v3 + DLPack v1.x) with primary-context lifetime guards.
//! - **Streaming**: O(1) performance
//! - **Memory**: Good zero-copy usage (alloc_with_nan_prefix, make_uninit_matrix)
//! - **Batch**: Row kernels reuse a precomputed mid-price series for all rows
//!
//! Decision: Streaming kernel mirrors scalar FMA ordering and uses a branchless
//! CU/CD split; outputs match the batch scalar path. SIMD remains delegated due
//! to IIR time dependency; CUDA paths are enabled but validated against the
//! scalar reference with relaxed FP32 tolerances.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::{make_device_array_py, DeviceArrayF32Py};

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum LrsiData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct LrsiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct LrsiParams {
    pub alpha: Option<f64>,
}

impl Default for LrsiParams {
    fn default() -> Self {
        Self { alpha: Some(0.2) }
    }
}

#[derive(Debug, Clone)]
pub struct LrsiInput<'a> {
    pub data: LrsiData<'a>,
    pub params: LrsiParams,
}

impl<'a> LrsiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: LrsiParams) -> Self {
        Self {
            data: LrsiData::Candles { candles: c },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], p: LrsiParams) -> Self {
        Self {
            data: LrsiData::Slices { high, low },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, LrsiParams::default())
    }
    #[inline]
    pub fn get_alpha(&self) -> f64 {
        self.params.alpha.unwrap_or(0.2)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LrsiBuilder {
    alpha: Option<f64>,
    kernel: Kernel,
}

impl Default for LrsiBuilder {
    fn default() -> Self {
        Self {
            alpha: None,
            kernel: Kernel::Auto,
        }
    }
}

impl LrsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn alpha(mut self, x: f64) -> Self {
        self.alpha = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<LrsiOutput, LrsiError> {
        let p = LrsiParams { alpha: self.alpha };
        let i = LrsiInput::from_candles(c, p);
        lrsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<LrsiOutput, LrsiError> {
        let p = LrsiParams { alpha: self.alpha };
        let i = LrsiInput::from_slices(high, low, p);
        lrsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<LrsiStream, LrsiError> {
        let p = LrsiParams { alpha: self.alpha };
        LrsiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum LrsiError {
    #[error("lrsi: Empty input data slice.")]
    EmptyInputData,
    #[error("lrsi: Invalid alpha: alpha = {alpha}. Must be between 0 and 1.")]
    InvalidAlpha { alpha: f64 },
    #[error("lrsi: All values are NaN.")]
    AllValuesNaN,
    #[error("lrsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("lrsi: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("lrsi: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: String, end: String, step: String },
    #[error("lrsi: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
}

#[inline]
pub fn lrsi(input: &LrsiInput) -> Result<LrsiOutput, LrsiError> {
    lrsi_with_kernel(input, Kernel::Auto)
}

/// Writes LRSI into the provided output slice without allocating.
///
/// - Preserves NaN warmups exactly like the Vec-returning API.
/// - `out.len()` must equal the input length; returns `OutputLengthMismatch` otherwise.
/// - Uses `Kernel::Auto` for runtime kernel selection.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn lrsi_into(input: &LrsiInput, out: &mut [f64]) -> Result<(), LrsiError> {
    lrsi_into_slice(out, input, Kernel::Auto)
}

pub fn lrsi_with_kernel(input: &LrsiInput, kernel: Kernel) -> Result<LrsiOutput, LrsiError> {
    let (high, low) = match &input.data {
        LrsiData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| LrsiError::EmptyInputData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| LrsiError::EmptyInputData)?;
            if high.len() != low.len() {
                return Err(LrsiError::EmptyInputData);
            }
            (high, low)
        }
        LrsiData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(LrsiError::EmptyInputData);
    }

    let alpha = input.get_alpha();
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(LrsiError::InvalidAlpha { alpha });
    }

    // Find first valid price without allocating
    let mut first_valid_idx = None;
    for i in 0..high.len() {
        let price = (high[i] + low[i]) / 2.0;
        if !price.is_nan() {
            first_valid_idx = Some(i);
            break;
        }
    }

    let first_valid_idx = first_valid_idx.ok_or(LrsiError::AllValuesNaN)?;
    let n = high.len();
    if n - first_valid_idx < 4 {
        return Err(LrsiError::NotEnoughValidData {
            needed: 4,
            valid: n - first_valid_idx,
        });
    }

    let warmup_period = first_valid_idx + 3; // Needs at least 4 values
    let mut out = alloc_with_nan_prefix(n, warmup_period);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        // Reject batch kernels in single-row calculations
        Kernel::ScalarBatch | Kernel::Avx2Batch | Kernel::Avx512Batch => {
            return Err(LrsiError::NotEnoughValidData {
                needed: 2,
                valid: 1,
            });
        }
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar => lrsi_scalar_hl(high, low, alpha, first_valid_idx, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => lrsi_avx2_hl(high, low, alpha, first_valid_idx, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => lrsi_avx512_hl(high, low, alpha, first_valid_idx, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(LrsiOutput { values: out })
}

#[inline]
pub fn lrsi_into_slice(dst: &mut [f64], input: &LrsiInput, kern: Kernel) -> Result<(), LrsiError> {
    let (high, low) = match &input.data {
        LrsiData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| LrsiError::EmptyInputData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| LrsiError::EmptyInputData)?;
            if high.len() != low.len() {
                return Err(LrsiError::EmptyInputData);
            }
            (high, low)
        }
        LrsiData::Slices { high, low } => (*high, *low),
    };

    let alpha = input.get_alpha();
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(LrsiError::InvalidAlpha { alpha });
    }

    // Find first valid price without allocating
    let mut first_valid_idx = None;
    for i in 0..high.len() {
        let price = (high[i] + low[i]) / 2.0;
        if !price.is_nan() {
            first_valid_idx = Some(i);
            break;
        }
    }

    let first_valid_idx = first_valid_idx.ok_or(LrsiError::AllValuesNaN)?;
    let n = high.len();

    if dst.len() != n {
        return Err(LrsiError::OutputLengthMismatch {
            expected: n,
            got: dst.len(),
        });
    }

    if n - first_valid_idx < 4 {
        return Err(LrsiError::NotEnoughValidData {
            needed: 4,
            valid: n - first_valid_idx,
        });
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        // Reject batch kernels in single-row calculations
        Kernel::ScalarBatch | Kernel::Avx2Batch | Kernel::Avx512Batch => {
            return Err(LrsiError::InvalidKernelForBatch(kern));
        }
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar => lrsi_scalar_hl(high, low, alpha, first_valid_idx, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => lrsi_avx2_hl(high, low, alpha, first_valid_idx, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => lrsi_avx512_hl(high, low, alpha, first_valid_idx, dst),
            _ => unreachable!(),
        }
    }

    // Fill warmup period with NaN
    let warmup_end = first_valid_idx + 3;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(test)]
mod tests_into_api {
    use super::*;

    #[test]
    fn test_lrsi_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Construct a small input with a NaN prefix and varying values
        let n = 256usize;
        let mut high = vec![f64::NAN; n];
        let mut low = vec![f64::NAN; n];

        // First valid at index 3; ensure at least 4 valid after that
        let mut v = 100.0f64;
        for i in 3..n {
            // Create a simple rising high/low band
            high[i] = v + 1.0;
            low[i] = v - 1.0;
            // Inject a few NaNs deeper in the series to test handling
            if i % 37 == 0 { high[i] = f64::NAN; }
            if i % 53 == 0 { low[i] = f64::NAN; }
            v += (i as f64).sin() * 0.25 + 0.5;
        }

        let input = LrsiInput::from_slices(&high, &low, LrsiParams::default());

        // Baseline Vec-returning API
        let base = lrsi(&input)?.values;

        // New into API
        let mut out = vec![0.0f64; n];
        #[cfg(not(feature = "wasm"))]
        {
            lrsi_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            // Guard for wasm builds: fall back to existing into_slice path
            lrsi_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(base.len(), out.len());
        for (i, (&a, &b)) in base.iter().zip(out.iter()).enumerate() {
            let equal = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(
                equal,
                "mismatch at index {}: base={}, into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }
}

#[inline]
pub fn lrsi_scalar_hl(high: &[f64], low: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
    debug_assert_eq!(high.len(), low.len());

    let len = high.len();
    if len == 0 {
        return;
    }

    let gamma = 1.0 - alpha;
    let mgamma = -gamma;
    let warm = first + 3;

    // Initialize state from first valid price
    let first_price = (high[first] + low[first]) * 0.5;
    let mut l0 = first_price;
    let mut l1 = first_price;
    let mut l2 = first_price;
    let mut l3 = first_price;

    for i in (first + 1)..len {
        // mid-price = (high + low) * 0.5
        let p = (high[i] + low[i]) * 0.5;

        if p.is_nan() {
            // Keep poison-free contract after warmup
            if i >= warm {
                out[i] = f64::NAN;
            }
            continue;
        }

        // 4-stage Laguerre filter using FMAs where profitable
        let t0 = (p - l0).mul_add(alpha, l0);
        let t1 = gamma.mul_add(l1, mgamma.mul_add(t0, l0));
        let t2 = gamma.mul_add(l2, mgamma.mul_add(t1, l1));
        let t3 = gamma.mul_add(l3, mgamma.mul_add(t2, l2));

        if i >= warm {
            // Differences
            let d01 = t0 - t1;
            let d12 = t1 - t2;
            let d23 = t2 - t3;

            // Branchless up/down components via abs identities
            let a01 = d01.abs();
            let a12 = d12.abs();
            let a23 = d23.abs();

            let sum_abs = a01 + a12 + a23; // cu + cd
            let cu = 0.5 * (d01 + a01 + d12 + a12 + d23 + a23);

            let v = if sum_abs <= f64::EPSILON {
                0.0
            } else {
                cu / sum_abs
            };
            // Numerical safety: clamp to [0,1]
            out[i] = v.min(1.0).max(0.0);
        }

        // Commit state
        l0 = t0;
        l1 = t1;
        l2 = t2;
        l3 = t3;
    }
}

// Keep old function for compatibility with row functions
#[inline]
pub fn lrsi_scalar(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
    let len = price.len();
    if len == 0 {
        return;
    }

    let gamma = 1.0 - alpha;
    let mgamma = -gamma;
    let warm = first + 3;

    let mut l0 = price[first];
    let mut l1 = l0;
    let mut l2 = l0;
    let mut l3 = l0;

    for i in (first + 1)..len {
        let p = price[i];
        if p.is_nan() {
            if i >= warm {
                out[i] = f64::NAN;
            }
            continue;
        }

        let t0 = (p - l0).mul_add(alpha, l0);
        let t1 = gamma.mul_add(l1, mgamma.mul_add(t0, l0));
        let t2 = gamma.mul_add(l2, mgamma.mul_add(t1, l1));
        let t3 = gamma.mul_add(l3, mgamma.mul_add(t2, l2));

        if i >= warm {
            let d01 = t0 - t1;
            let d12 = t1 - t2;
            let d23 = t2 - t3;

            let a01 = d01.abs();
            let a12 = d12.abs();
            let a23 = d23.abs();

            let sum_abs = a01 + a12 + a23;
            let cu = 0.5 * (d01 + a01 + d12 + a12 + d23 + a23);

            let v = if sum_abs <= f64::EPSILON {
                0.0
            } else {
                cu / sum_abs
            };
            // Numerical safety: clamp to [0,1]
            out[i] = v.min(1.0).max(0.0);
        }

        l0 = t0;
        l1 = t1;
        l2 = t2;
        l3 = t3;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx2_hl(high: &[f64], low: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
    lrsi_scalar_hl(high, low, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512_hl(high: &[f64], low: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
    lrsi_scalar_hl(high, low, alpha, first, out)
}

// Keep old functions for compatibility with row functions
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx2(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
    lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
    lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512_short(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
    lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512_long(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
    lrsi_scalar(price, alpha, first, out)
}

// Streaming API
#[derive(Debug, Clone)]
pub struct LrsiStream {
    alpha: f64,
    gamma: f64,
    l0: f64,
    l1: f64,
    l2: f64,
    l3: f64,
    initialized: bool,
    count: usize, // Track number of updates to match batch warmup
}

impl LrsiStream {
    pub fn try_new(params: LrsiParams) -> Result<Self, LrsiError> {
        let alpha = params.alpha.unwrap_or(0.2);
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(LrsiError::InvalidAlpha { alpha });
        }
        Ok(Self {
            alpha,
            gamma: 1.0 - alpha,
            l0: f64::NAN,
            l1: f64::NAN,
            l2: f64::NAN,
            l3: f64::NAN,
            initialized: false,
            count: 0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, price: f64) -> Option<f64> {
        // Ignore NaNs; do not advance state
        if price.is_nan() {
            return None;
        }

        // One-time init to match batch warmup behavior
        if !self.initialized {
            self.l0 = price;
            self.l1 = price;
            self.l2 = price;
            self.l3 = price;
            self.initialized = true;
            self.count = 0;
            return None; // first seen sample doesn't produce output
        }

        // 4-stage Laguerre filter with FMA ordering (matches scalar batch path)
        let gamma = self.gamma;
        let mgamma = -gamma;

        let l0_prev = self.l0;
        let l1_prev = self.l1;
        let l2_prev = self.l2;
        let l3_prev = self.l3;

        let t0 = (price - l0_prev).mul_add(self.alpha, l0_prev);
        let t1 = gamma.mul_add(l1_prev, mgamma.mul_add(t0, l0_prev));
        let t2 = gamma.mul_add(l2_prev, mgamma.mul_add(t1, l1_prev));
        let t3 = gamma.mul_add(l3_prev, mgamma.mul_add(t2, l2_prev));

        // Commit state
        self.l0 = t0;
        self.l1 = t1;
        self.l2 = t2;
        self.l3 = t3;
        self.count += 1;

        // Warmup = 4 samples total (emit after 3 updates post-initialization)
        if self.count < 3 {
            return None;
        }

        // Branchless CU/CD via positive-part identity
        let d01 = t0 - t1;
        let d12 = t1 - t2;
        let d23 = t2 - t3;

        let a01 = d01.abs();
        let a12 = d12.abs();
        let a23 = d23.abs();

        let sum_abs = a01 + a12 + a23;
        if sum_abs <= f64::EPSILON {
            return Some(0.0);
        }

        // Match scalar addition order for numerical reproducibility
        let cu = 0.5 * (d01 + a01 + d12 + a12 + d23 + a23);

        // Use exact divide to mirror scalar kernel
        let v = cu / sum_abs;
        Some(v.min(1.0).max(0.0))
    }
}

#[derive(Clone, Debug)]
pub struct LrsiBatchRange {
    pub alpha: (f64, f64, f64),
}

impl Default for LrsiBatchRange {
    fn default() -> Self {
        Self {
            alpha: (0.2, 0.449, 0.001),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LrsiBatchBuilder {
    range: LrsiBatchRange,
    kernel: Kernel,
}

impl LrsiBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn alpha_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.alpha = (start, end, step);
        self
    }
    #[inline]
    pub fn alpha_static(mut self, x: f64) -> Self {
        self.range.alpha = (x, x, 0.0);
        self
    }

    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<LrsiBatchOutput, LrsiError> {
        lrsi_batch_with_kernel(high, low, &self.range, self.kernel)
    }

    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<LrsiBatchOutput, LrsiError> {
        LrsiBatchBuilder::new().kernel(k).apply_slices(high, low)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<LrsiBatchOutput, LrsiError> {
        let high = c
            .select_candle_field("high")
            .map_err(|_| LrsiError::EmptyInputData)?;
        let low = c
            .select_candle_field("low")
            .map_err(|_| LrsiError::EmptyInputData)?;
        if high.len() != low.len() {
            return Err(LrsiError::EmptyInputData);
        }
        self.apply_slices(high, low)
    }

    pub fn with_default_candles(c: &Candles) -> Result<LrsiBatchOutput, LrsiError> {
        LrsiBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

pub fn lrsi_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &LrsiBatchRange,
    k: Kernel,
) -> Result<LrsiBatchOutput, LrsiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(LrsiError::InvalidKernelForBatch(other)),
    };
    lrsi_batch_par_slice(high, low, sweep, kernel)
}

#[derive(Clone, Debug)]
pub struct LrsiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<LrsiParams>,
    pub rows: usize,
    pub cols: usize,
}
impl LrsiBatchOutput {
    pub fn row_for_params(&self, p: &LrsiParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| (c.alpha.unwrap_or(0.2) - p.alpha.unwrap_or(0.2)).abs() < 1e-12)
    }

    pub fn values_for(&self, p: &LrsiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &LrsiBatchRange) -> Result<Vec<LrsiParams>, LrsiError> {
    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, LrsiError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(LrsiError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            return Ok(v);
        }
        // Reversed bounds
        let mut v = Vec::new();
        let mut x = start;
        let st = step.abs();
        while x + 1e-12 >= end {
            v.push(x);
            x -= st;
        }
        if v.is_empty() {
            return Err(LrsiError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }

    let alphas = axis_f64(r.alpha)?;

    let mut out = Vec::with_capacity(alphas.len());
    for &a in &alphas {
        out.push(LrsiParams { alpha: Some(a) });
    }
    Ok(out)
}

#[inline(always)]
pub fn lrsi_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &LrsiBatchRange,
    kern: Kernel,
) -> Result<LrsiBatchOutput, LrsiError> {
    lrsi_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn lrsi_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &LrsiBatchRange,
    kern: Kernel,
) -> Result<LrsiBatchOutput, LrsiError> {
    lrsi_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn lrsi_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &LrsiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<LrsiBatchOutput, LrsiError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(LrsiError::InvalidRange {
            start: sweep.alpha.0.to_string(),
            end: sweep.alpha.1.to_string(),
            step: sweep.alpha.2.to_string(),
        });
    }
    if high.is_empty() || low.is_empty() {
        return Err(LrsiError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(LrsiError::EmptyInputData);
    }

    let cols = high.len();
    let rows = combos.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(LrsiError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".into(),
        })?;

    // Find first valid price
    let first = (0..cols)
        .find(|&i| ((high[i] + low[i]) / 2.0).is_finite())
        .ok_or(LrsiError::AllValuesNaN)?;
    if cols - first < 4 {
        return Err(LrsiError::NotEnoughValidData {
            needed: 4,
            valid: cols - first,
        });
    }

    // Allocate rows×cols uninit and set warm prefixes to NaN via helper
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm = vec![first + 3; rows];
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Temporary &mut [f64] view for filling
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_slice: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Resolve kernel for rows (batch→simd choice matches ALMA)
    let resolved = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k if k.is_batch() => k,
        other => return Err(LrsiError::InvalidKernelForBatch(other)),
    };
    let row_kernel = match resolved {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    let combos = lrsi_batch_inner_into(high, low, sweep, row_kernel, parallel, out_slice)?;

    // Turn buffer into Vec<f64> without copying
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(LrsiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn lrsi_row_scalar_hl(high: &[f64], low: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
    lrsi_scalar_hl(high, low, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx2_hl(high: &[f64], low: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
    lrsi_scalar_hl(high, low, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512_hl(high: &[f64], low: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
    lrsi_scalar_hl(high, low, alpha, first, out)
}

// Keep old functions for compatibility
#[inline(always)]
unsafe fn lrsi_row_scalar(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
    lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx2(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
    lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
    lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512_short(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
    lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512_long(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
    lrsi_scalar(price, alpha, first, out)
}

// WASM Bindings

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn lrsi_js(high: &[f64], low: &[f64], alpha: f64) -> Result<Vec<f64>, JsValue> {
    let params = LrsiParams { alpha: Some(alpha) };
    let input = LrsiInput::from_slices(high, low, params);

    let mut output = vec![0.0; high.len()];
    lrsi_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LrsiBatchConfig {
    pub alpha_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LrsiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<LrsiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = lrsi_batch)]
pub fn lrsi_batch_unified_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: LrsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = LrsiBatchRange {
        alpha: config.alpha_range,
    };
    let result = lrsi_batch_with_kernel(high, low, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let output = LrsiBatchJsOutput {
        values: result.values,
        combos: result.combos,
        rows: result.rows,
        cols: result.cols,
    };

    serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn lrsi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn lrsi_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn lrsi_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    alpha: f64,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to lrsi_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let params = LrsiParams { alpha: Some(alpha) };
        let input = LrsiInput::from_slices(high, low, params);

        // Check if we need to handle aliasing (in-place operation)
        if high_ptr == out_ptr || low_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            lrsi_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            lrsi_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn lrsi_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    alpha_start: f64,
    alpha_end: f64,
    alpha_step: f64,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to lrsi_batch_into"));
    }

    // Validate alpha parameters
    if !(0.0 < alpha_start && alpha_start <= 1.0) {
        return Err(JsValue::from_str(&format!(
            "Invalid alpha_start: {}",
            alpha_start
        )));
    }
    if !(0.0 < alpha_end && alpha_end <= 1.0) {
        return Err(JsValue::from_str(&format!(
            "Invalid alpha_end: {}",
            alpha_end
        )));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let sweep = LrsiBatchRange {
            alpha: (alpha_start, alpha_end, alpha_step),
        };
        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow in lrsi_batch"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        let row_kernel = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };
        lrsi_batch_inner_into(high, low, &sweep, row_kernel, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_lrsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = LrsiParams { alpha: None };
        let input = LrsiInput::from_candles(&candles, default_params);
        let output = lrsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_lrsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = LrsiInput::from_candles(&candles, LrsiParams::default());
        let lrsi_result = lrsi_with_kernel(&input, kernel)?;
        assert_eq!(lrsi_result.values.len(), candles.close.len());
        let expected_last_five_lrsi = [0.0, 0.0, 0.0, 0.0, 0.0];
        let start_index = lrsi_result.values.len() - 5;
        let result_last_five_lrsi = &lrsi_result.values[start_index..];
        for (i, &value) in result_last_five_lrsi.iter().enumerate() {
            let expected_value = expected_last_five_lrsi[i];
            assert!(
                (value - expected_value).abs() < 1e-9,
                "LRSI mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    fn check_lrsi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = LrsiInput::with_default_candles(&candles);
        let output = lrsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_lrsi_invalid_alpha(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [1.0, 2.0];
        let low = [1.0, 2.0];
        let params = LrsiParams { alpha: Some(1.2) };
        let input = LrsiInput::from_slices(&high, &low, params);
        let result = lrsi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_lrsi_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let params = LrsiParams::default();
        let input = LrsiInput::from_slices(&high, &low, params);
        let result = lrsi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_lrsi_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = LrsiParams::default();
        let input = LrsiInput::from_slices(&high, &low, params);
        let result = lrsi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_lrsi_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [1.0, 1.0];
        let low = [1.0, 1.0];
        let params = LrsiParams::default();
        let input = LrsiInput::from_slices(&high, &low, params);
        let result = lrsi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_lrsi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = candles.select_candle_field("high").unwrap();
        let low = candles.select_candle_field("low").unwrap();

        let input = LrsiInput::from_slices(high, low, LrsiParams::default());
        let batch_output = lrsi_with_kernel(&input, kernel)?.values;

        let mut stream = LrsiStream::try_new(LrsiParams::default())?;
        let mut stream_values = Vec::with_capacity(high.len());
        for i in 0..high.len() {
            let price = (high[i] + low[i]) / 2.0;
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
                "[{}] LRSI streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_lrsi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Fill poison values
        let len = candles.close.len();
        let mut high = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);
        let mut low = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);

        high.resize(len, f64::from_bits(0x11111111_11111111));
        low.resize(len, f64::from_bits(0x22222222_22222222));

        // Copy real data
        high.copy_from_slice(&candles.high);
        low.copy_from_slice(&candles.low);

        // Test with different alpha values
        let test_params = vec![
            LrsiParams { alpha: Some(0.1) },
            LrsiParams { alpha: Some(0.2) }, // default
            LrsiParams { alpha: Some(0.5) },
            LrsiParams { alpha: Some(0.8) },
            LrsiParams { alpha: Some(0.95) },
        ];

        for params in test_params {
            let input = LrsiInput::from_slices(&high, &low, params.clone());
            let result = lrsi_with_kernel(&input, kernel)?;

            // Check for poison values after warmup period
            for (i, &val) in result.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						with params: alpha={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.alpha.unwrap_or(0.2)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						with params: alpha={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.alpha.unwrap_or(0.2)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found third poison value {} (0x{:016X}) at index {} \
						with params: alpha={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.alpha.unwrap_or(0.2)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_lrsi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_lrsi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic high/low price pairs
        let strat = (4usize..=400, 0.01f64..0.99f64, prop::bool::weighted(0.1)).prop_flat_map(
            |(len, alpha, use_constant_price)| {
                if use_constant_price && len < 50 {
                    // Generate truly constant prices for testing Property 5
                    let constant_price = (10.0f64..200.0f64);
                    constant_price
                        .prop_map(move |price| {
                            let high = vec![price; len];
                            let low = vec![price; len];
                            (high, low, alpha)
                        })
                        .boxed()
                } else {
                    // Generate normal price data with spreads
                    (
                        // Generate base prices
                        proptest::collection::vec(
                            (10.0f64..200.0f64).prop_filter("finite", |x| x.is_finite()),
                            len,
                        ),
                        // Generate spread percentages for high/low
                        proptest::collection::vec(
                            (0.0f64..0.05f64), // 0-5% spread
                            len,
                        ),
                        Just(alpha),
                    )
                        .prop_map(|(base_prices, spreads, alpha)| {
                            // Create high/low pairs from base prices and spreads
                            let mut high = Vec::with_capacity(base_prices.len());
                            let mut low = Vec::with_capacity(base_prices.len());

                            for (base, spread) in base_prices.iter().zip(spreads.iter()) {
                                let half_spread = base * spread / 2.0;
                                high.push(base + half_spread);
                                low.push(base - half_spread);
                            }

                            (high, low, alpha)
                        })
                        .boxed()
                }
            },
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(high, low, alpha)| {
                let params = LrsiParams { alpha: Some(alpha) };
                let input = LrsiInput::from_slices(&high, &low, params.clone());

                // Test with specified kernel
                let result = lrsi_with_kernel(&input, kernel)?;
                let out = result.values;

                // Test with scalar reference
                let ref_result = lrsi_with_kernel(&input, Kernel::Scalar)?;
                let ref_out = ref_result.values;

                // Property 1: Output length matches input
                prop_assert_eq!(out.len(), high.len(), "Output length mismatch");

                // Find first valid price index
                let mut first_valid_idx = None;
                for i in 0..high.len() {
                    let price = (high[i] + low[i]) / 2.0;
                    if !price.is_nan() {
                        first_valid_idx = Some(i);
                        break;
                    }
                }

                if let Some(first_idx) = first_valid_idx {
                    let warmup_end = first_idx + 3;

                    // Property 2: Warmup period handling
                    // The implementation sets out[first] = 0.0 and starts calculating from first+1
                    // So only indices before first_idx should be NaN
                    for i in 0..first_idx {
                        prop_assert!(
                            out[i].is_nan(),
                            "Expected NaN before first valid at index {}, got {}",
                            i,
                            out[i]
                        );
                    }

                    // The first output value after warmup should be valid
                    // Warmup period is first_idx + 3, so first valid output is at first_idx + 3
                    let first_output_idx = first_idx + 3;
                    if first_output_idx < out.len() && !out[first_output_idx].is_nan() {
                        // First output after warmup should be a valid LRSI value [0, 1]
                        prop_assert!(
                            out[first_output_idx] >= 0.0 && out[first_output_idx] <= 1.0,
                            "First output after warmup at index {} = {}, should be in [0, 1]",
                            first_output_idx,
                            out[first_output_idx]
                        );
                    }

                    // Property 3: LRSI values bounded [0, 1]
                    // LRSI produces values starting from first_idx + 3 (after warmup)
                    for i in (first_idx + 3)..out.len() {
                        if !out[i].is_nan() {
                            prop_assert!(
                                out[i] >= 0.0 && out[i] <= 1.0,
                                "LRSI value {} at index {} outside [0, 1] range",
                                out[i],
                                i
                            );
                        }
                    }

                    // Property 4: Kernel consistency (comparing with scalar)
                    for i in 0..out.len() {
                        let y = out[i];
                        let r = ref_out[i];

                        if !y.is_finite() || !r.is_finite() {
                            prop_assert_eq!(
                                y.to_bits(),
                                r.to_bits(),
                                "NaN/infinite mismatch at index {}: {} vs {}",
                                i,
                                y,
                                r
                            );
                        } else {
                            let y_bits = y.to_bits();
                            let r_bits = r.to_bits();
                            let ulp_diff = y_bits.abs_diff(r_bits);

                            prop_assert!(
                                (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                                "Kernel mismatch at index {}: {} vs {} (ULP={}, alpha={})",
                                i,
                                y,
                                r,
                                ulp_diff,
                                alpha
                            );
                        }
                    }

                    // Property 5: Constant price behavior
                    // If high == low for all values, LRSI should stabilize
                    let is_constant = high
                        .iter()
                        .zip(low.iter())
                        .all(|(h, l)| (h - l).abs() < f64::EPSILON && h.is_finite());

                    if is_constant && out.len() > first_idx + 10 {
                        // After sufficient iterations, LRSI should stabilize near 0 or a constant
                        let last_values = &out[out.len() - 5..];
                        let valid_last = last_values
                            .iter()
                            .filter(|v| v.is_finite())
                            .collect::<Vec<_>>();

                        if valid_last.len() >= 2 {
                            let variance = valid_last
                                .windows(2)
                                .map(|w| (w[1] - w[0]).abs())
                                .fold(0.0, f64::max);

                            // LRSI can oscillate even with constant prices due to filter dynamics
                            // Allow reasonable variance for an oscillator
                            prop_assert!(
                                variance < 0.1,
                                "LRSI not stable for constant prices, variance: {}",
                                variance
                            );
                        }
                    }

                    // Property 6: Alpha boundary behavior
                    // Very low alpha (< 0.05) = less responsive, very high alpha (> 0.95) = more responsive
                    if out.len() > first_idx + 25 {
                        // Calculate input data volatility for dynamic threshold
                        let prices: Vec<f64> = high
                            .iter()
                            .zip(low.iter())
                            .map(|(h, l)| (h + l) / 2.0)
                            .filter(|p| p.is_finite())
                            .collect();

                        if prices.len() >= 10 {
                            let price_changes: Vec<f64> = prices
                                .windows(2)
                                .map(|w| ((w[1] - w[0]) / w[0]).abs())
                                .collect();
                            let input_volatility = if !price_changes.is_empty() {
                                price_changes.iter().sum::<f64>() / price_changes.len() as f64
                            } else {
                                0.01 // Default volatility
                            };

                            let start = (first_idx + 5).min(out.len().saturating_sub(5));
                            let end = out.len().saturating_sub(5);
                            if start < end {
                                let mid_section = &out[start..end];
                                let valid_mid: Vec<f64> = mid_section
                                    .iter()
                                    .filter(|v| v.is_finite())
                                    .copied()
                                    .collect();

                                if valid_mid.len() >= 10 {
                                    let avg_change = valid_mid
                                        .windows(2)
                                        .map(|w| (w[1] - w[0]).abs())
                                        .sum::<f64>()
                                        / (valid_mid.len() - 1) as f64;

                                    if alpha < 0.05 {
                                        // Low alpha should smooth the output relative to input volatility
                                        // The smoother the alpha, the more it should dampen volatility
                                        let expected_max_change =
                                            input_volatility * (alpha * 20.0).max(0.1);
                                        prop_assert!(
											avg_change <= expected_max_change,
											"Low alpha ({}) should produce smooth output relative to input volatility. \
											Avg change: {}, Expected max: {}, Input volatility: {}",
											alpha,
											avg_change,
											expected_max_change,
											input_volatility
										);
                                    } else if alpha > 0.95 {
                                        // High alpha should be more responsive
                                        // But LRSI is still bounded [0,1] so it won't match input volatility exactly
                                        // Use a more reasonable expectation
                                        let expected_min_change = (input_volatility * 0.2).min(0.1);
                                        // Only test if there's meaningful volatility
                                        if input_volatility > 0.01 {
                                            prop_assert!(
												avg_change >= expected_min_change || avg_change < 0.001,
												"High alpha ({}) should be responsive to input changes. \
												Avg change: {}, Expected min: {}, Input volatility: {}",
												alpha,
												avg_change,
												expected_min_change,
												input_volatility
											);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Property 7: Monotonic trend response (with tighter tolerance)
                    // Check if prices are monotonically increasing or decreasing with actual change
                    let is_monotonic_up = high.windows(2).all(|w| w[1] >= w[0])
                        && high.windows(2).any(|w| w[1] > w[0] + f64::EPSILON); // Must have actual increase
                    let is_monotonic_down = high.windows(2).all(|w| w[1] <= w[0])
                        && high.windows(2).any(|w| w[1] < w[0] - f64::EPSILON); // Must have actual decrease

                    if (is_monotonic_up || is_monotonic_down) && out.len() > first_idx + 20 {
                        let valid_out: Vec<(usize, f64)> = out
                            .iter()
                            .enumerate()
                            .skip(first_idx + 1)
                            .filter(|(_, v)| v.is_finite())
                            .map(|(i, v)| (i, *v))
                            .collect();

                        if valid_out.len() >= 20 {
                            // Check general trend direction over chunks
                            let chunk_size = valid_out.len() / 3;
                            if chunk_size >= 5 {
                                let first_chunk_avg =
                                    valid_out[..chunk_size].iter().map(|(_, v)| v).sum::<f64>()
                                        / chunk_size as f64;
                                let last_chunk_avg = valid_out[valid_out.len() - chunk_size..]
                                    .iter()
                                    .map(|(_, v)| v)
                                    .sum::<f64>()
                                    / chunk_size as f64;

                                // Calculate how strong the trend is
                                let price_range = high
                                    .iter()
                                    .zip(low.iter())
                                    .map(|(h, l)| (h + l) / 2.0)
                                    .filter(|p| p.is_finite())
                                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), p| {
                                        (min.min(p), max.max(p))
                                    });
                                let trend_strength = if price_range.1 > price_range.0 {
                                    (price_range.1 - price_range.0) / price_range.0
                                } else {
                                    0.01
                                };

                                // Tighter tolerance: 0.05 base, adjusted by trend strength and alpha
                                let tolerance = (0.05 * (1.0 - alpha * 0.5)).min(0.05);

                                if is_monotonic_up {
                                    // For uptrend, LRSI should increase
                                    prop_assert!(
										last_chunk_avg >= first_chunk_avg - tolerance,
										"LRSI should respond to uptrend, but first_avg={}, last_avg={}, \
										tolerance={}, alpha={}, trend_strength={}",
										first_chunk_avg,
										last_chunk_avg,
										tolerance,
										alpha,
										trend_strength
									);
                                } else if is_monotonic_down {
                                    // For downtrend, LRSI should decrease
                                    prop_assert!(
										last_chunk_avg <= first_chunk_avg + tolerance,
										"LRSI should respond to downtrend, but first_avg={}, last_avg={}, \
										tolerance={}, alpha={}, trend_strength={}",
										first_chunk_avg,
										last_chunk_avg,
										tolerance,
										alpha,
										trend_strength
									);
                                }
                            }
                        }
                    }

                    // Property 8: Extreme alpha values
                    // Test behavior at alpha boundaries
                    if alpha < 0.02 || alpha > 0.98 {
                        if out.len() > first_idx + 50 {
                            let valid_values: Vec<f64> = out
                                .iter()
                                .skip(first_idx + 10)
                                .filter(|v| v.is_finite())
                                .copied()
                                .collect();

                            if valid_values.len() >= 20 {
                                if alpha < 0.02 {
                                    // Very low alpha: LRSI should be smooth after initial settling
                                    // Skip first few values which might have larger steps due to initialization
                                    let settled_values = if valid_values.len() > 10 {
                                        &valid_values[5..] // Skip first 5 values
                                    } else {
                                        &valid_values[..]
                                    };

                                    if settled_values.len() >= 5 {
                                        let max_step = settled_values
                                            .windows(2)
                                            .map(|w| (w[1] - w[0]).abs())
                                            .fold(0.0f64, f64::max);

                                        // LRSI with low alpha can still have steps due to filter dynamics
                                        // but should be generally smooth
                                        prop_assert!(
											max_step < 0.7,  // Allow for Laguerre filter dynamics
											"Extreme low alpha ({}) should produce smooth output after settling, \
											but max step is {}",
											alpha,
											max_step
										);
                                    }

                                    // Should converge to a relatively stable value
                                    // Check the last values for stability
                                    if valid_values.len() >= 20 {
                                        let last_10 = &valid_values[valid_values.len() - 10..];
                                        // Check max difference in last 10 values
                                        let min_val =
                                            last_10.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                                        let max_val = last_10
                                            .iter()
                                            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                                        let range = max_val - min_val;

                                        // With extreme low alpha, the range should be small
                                        prop_assert!(
											range < 0.5,
											"Extreme low alpha ({}) should converge to stable value, \
											but range in last 10 values is {}",
											alpha,
											range
										);
                                    }
                                } else {
                                    // Very high alpha: LRSI should be highly responsive
                                    // Should have meaningful variation if input has variation
                                    let range = valid_values.iter().fold(
                                        (f64::INFINITY, f64::NEG_INFINITY),
                                        |(min, max), &v| (min.min(v), max.max(v)),
                                    );

                                    // Only test if input has meaningful variation
                                    let input_has_variation =
                                        high.windows(2).any(|w| (w[1] - w[0]).abs() > w[0] * 0.001);

                                    if input_has_variation {
                                        prop_assert!(
                                            range.1 - range.0 > 0.05,
                                            "Extreme high alpha ({}) should produce varied output \
											for varied input, but range is only {}",
                                            alpha,
                                            range.1 - range.0
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_lrsi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        Ok(())
    }

    macro_rules! generate_all_lrsi_tests {
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

    generate_all_lrsi_tests!(
        check_lrsi_partial_params,
        check_lrsi_accuracy,
        check_lrsi_default_candles,
        check_lrsi_invalid_alpha,
        check_lrsi_empty_data,
        check_lrsi_all_nan,
        check_lrsi_very_small_dataset,
        check_lrsi_streaming,
        check_lrsi_no_poison,
        check_lrsi_property
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = LrsiBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = LrsiParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [0.0, 0.0, 0.0, 0.0, 0.0];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-9,
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

        // Use smaller dataset for batch testing
        let slice_end = c.close.len().min(1000);
        let high_slice = &c.high[..slice_end];
        let low_slice = &c.low[..slice_end];

        // Test with different parameter sweeps
        let test_configs = vec![
            (0.1, 0.3, 0.1),    // Small alpha values
            (0.2, 0.8, 0.2),    // Medium range
            (0.5, 0.9, 0.1),    // Higher alpha values
            (0.1, 0.95, 0.15),  // Wide range
            (0.85, 0.95, 0.05), // High alpha values
        ];

        for (cfg_idx, &(a_start, a_end, a_step)) in test_configs.iter().enumerate() {
            let output = LrsiBatchBuilder::new()
                .kernel(kernel)
                .alpha_range(a_start, a_end, a_step)
                .apply_slices(high_slice, low_slice)?;

            // Check for poison values in output
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
						at row {} col {} (flat index {}) with params: alpha={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.alpha.unwrap_or(0.2)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: alpha={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.alpha.unwrap_or(0.2)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found third poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: alpha={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.alpha.unwrap_or(0.2)
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

// Batch function that writes directly to output slice for Python bindings
#[inline(always)]
fn lrsi_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &LrsiBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<LrsiParams>, LrsiError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(LrsiError::InvalidRange {
            start: sweep.alpha.0.to_string(),
            end: sweep.alpha.1.to_string(),
            step: sweep.alpha.2.to_string(),
        });
    }

    if high.is_empty() || low.is_empty() {
        return Err(LrsiError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(LrsiError::EmptyInputData);
    }

    // Precompute mid-price once for all rows and find first valid index
    let len = high.len();
    let mut prices = Vec::with_capacity(len);
    prices.extend((0..len).map(|i| (high[i] + low[i]) * 0.5));
    let first = (0..len)
        .find(|&i| prices[i].is_finite())
        .ok_or(LrsiError::AllValuesNaN)?;
    if len - first < 4 {
        return Err(LrsiError::NotEnoughValidData {
            needed: 4,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(LrsiError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".into(),
        })?;
    if out.len() != expected {
        return Err(LrsiError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    // Work through MaybeUninit to avoid UB on uninitialized cells
    let out_mu: &mut [std::mem::MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };

    let do_row = |row: usize, dst_row_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let alpha = combos[row].alpha.unwrap();
        // View as f64 for kernel writes
        let dst_row =
            std::slice::from_raw_parts_mut(dst_row_mu.as_mut_ptr() as *mut f64, dst_row_mu.len());

        // Initialize warmup period with NaN
        let warmup_end = first + 3;
        for i in 0..warmup_end.min(dst_row.len()) {
            dst_row[i] = f64::NAN;
        }

        match kern {
            Kernel::Scalar => lrsi_row_scalar(&prices, first, alpha, dst_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => lrsi_row_avx2(&prices, first, alpha, dst_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => lrsi_row_avx512(&prices, first, alpha, dst_row),
            Kernel::Auto | _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, row)| do_row(r, row));
        #[cfg(target_arch = "wasm32")]
        for (r, row) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    } else {
        for (r, row) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "lrsi")]
#[pyo3(signature = (high, low, alpha, kernel=None))]
pub fn lrsi_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = LrsiParams { alpha: Some(alpha) };
    let inp = LrsiInput::from_slices(h, l, params);

    let vec_out: Vec<f64> = py
        .allow_threads(|| lrsi_with_kernel(&inp, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(vec_out.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "LrsiStream")]
pub struct LrsiStreamPy {
    stream: LrsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LrsiStreamPy {
    #[new]
    fn new(alpha: f64) -> PyResult<Self> {
        let params = LrsiParams { alpha: Some(alpha) };
        let stream =
            LrsiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(LrsiStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        let price = (high + low) / 2.0;
        self.stream.update(price)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "lrsi_batch")]
#[pyo3(signature = (high, low, alpha_range, kernel=None))]
pub fn lrsi_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    alpha_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let sweep = LrsiBatchRange { alpha: alpha_range };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = h.len();

    // Preallocate NumPy 1D buffer rows*cols; we will fill it in place
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let resolved = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let row_kernel = match resolved {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };
        lrsi_batch_inner_into(h, l, &sweep, row_kernel, true, out_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "alphas",
        combos
            .iter()
            .map(|p| p.alpha.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

// ==================== PYTHON: CUDA BINDINGS (zero-copy) ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "lrsi_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, alpha_range, device_id=0))]
pub fn lrsi_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_f32: numpy::PyReadonlyArray1<'_, f32>,
    alpha_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaLrsi;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err("mismatched input lengths"));
    }
    let sweep = LrsiBatchRange { alpha: alpha_range };
    let inner = py.allow_threads(|| {
        let mut cuda =
            CudaLrsi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.lrsi_batch_dev(h, l, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let handle = make_device_array_py(device_id, inner)?;
    Ok(handle)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "lrsi_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, alpha, device_id=0))]
pub fn lrsi_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    alpha: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaLrsi;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let rows = high_tm_f32.shape()[0];
    let cols = high_tm_f32.shape()[1];
    if low_tm_f32.shape() != [rows, cols] {
        return Err(PyValueError::new_err("mismatched matrix shapes"));
    }
    let inner = py.allow_threads(|| {
        let mut cuda =
            CudaLrsi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.lrsi_many_series_one_param_time_major_dev(h, l, cols, rows, alpha)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let handle = make_device_array_py(device_id, inner)?;
    Ok(handle)
}
