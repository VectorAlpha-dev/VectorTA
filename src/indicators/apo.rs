//! # Absolute Price Oscillator (APO)
//!
//! Calculates the difference between two exponential moving averages (EMAs) of
//! different lengths (`short_period` and `long_period`), measuring momentum and
//! trend shifts. The interface and performance are structured similar to ALMA.
//!
//! ## Parameters
//! - **short_period**: EMA window size for the short period (defaults to 10).
//! - **long_period**: EMA window size for the long period (defaults to 20).
//!
//! ## Returns
//! - **`Ok(ApoOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(ApoError)`** otherwise.
//!
//! ## Developer Status
//! - **AVX2/AVX512 kernels**: Implemented (pairwise EMA lanes, no FMA for parity)
//! - Rationale: EMA is sequential; SIMD across time needs scans. Our kernels
//!   parallelize short/long EMAs per sample in lanes. On tested CPUs they were
//!   not >5% faster than the scalar path due to front-end/memory limits. Auto
//!   selection uses `detect_best_kernel()`; if you observe regressions, force
//!   `Kernel::Scalar` explicitly at call site.
//! - **Streaming update**: O(1) - Efficient dual EMA updates
//! - **Memory optimization**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix) ✓
//! - **CUDA + Python interop**: CUDA wrapper returns VRAM handles with CAI v3 + DLPack v1.x; SIMD is implemented but kept on par with scalar for correctness and stability.

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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

// --- Data Representation

#[derive(Debug, Clone)]
pub enum ApoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for ApoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ApoData::Slice(slice) => slice,
            ApoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// --- Structs and Params

#[derive(Debug, Clone)]
pub struct ApoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ApoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}
impl Default for ApoParams {
    fn default() -> Self {
        Self {
            short_period: Some(10),
            long_period: Some(20),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApoInput<'a> {
    pub data: ApoData<'a>,
    pub params: ApoParams,
}
impl<'a> ApoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ApoParams) -> Self {
        Self {
            data: ApoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ApoParams) -> Self {
        Self {
            data: ApoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ApoParams::default())
    }
    #[inline]
    pub fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(10)
    }
    #[inline]
    pub fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(20)
    }
}

// --- Error Types

#[derive(Debug, Error)]
pub enum ApoError {
    #[error("apo: Input data slice is empty.")]
    EmptyInputData,
    #[error("apo: All values are NaN.")]
    AllValuesNaN,
    #[error("apo: Invalid period: short={short}, long={long}")]
    InvalidPeriod { short: usize, long: usize },
    #[error("apo: short_period not less than long_period: short={short}, long={long}")]
    ShortPeriodNotLessThanLong { short: usize, long: usize },
    #[error("apo: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("apo: output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("apo: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("apo: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

// --- Builder API

#[derive(Copy, Clone, Debug)]
pub struct ApoBuilder {
    short_period: Option<usize>,
    long_period: Option<usize>,
    kernel: Kernel,
}
impl Default for ApoBuilder {
    fn default() -> Self {
        Self {
            short_period: None,
            long_period: None,
            kernel: Kernel::Auto,
        }
    }
}
impl ApoBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_period(mut self, n: usize) -> Self {
        self.short_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_period(mut self, n: usize) -> Self {
        self.long_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<ApoOutput, ApoError> {
        let p = ApoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = ApoInput::from_candles(c, "close", p);
        apo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ApoOutput, ApoError> {
        let p = ApoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = ApoInput::from_slice(d, p);
        apo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ApoStream, ApoError> {
        let p = ApoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        ApoStream::try_new(p)
    }
}

// --- Main Indicator Function

#[inline]
pub fn apo(input: &ApoInput) -> Result<ApoOutput, ApoError> {
    apo_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn apo_prepare<'a>(
    input: &'a ApoInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, usize, Kernel), ApoError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(ApoError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ApoError::AllValuesNaN)?;
    let short = input.get_short_period();
    let long = input.get_long_period();

    if short == 0 || long == 0 {
        return Err(ApoError::InvalidPeriod { short, long });
    }
    if short >= long {
        return Err(ApoError::ShortPeriodNotLessThanLong { short, long });
    }
    if (len - first) < long {
        return Err(ApoError::NotEnoughValidData {
            needed: long,
            valid: len - first,
        });
    }

    let mut chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    // APO is two EMAs (sequential recurrence); SIMD typically provides no benefit and can be slower
    // due to downclock/extra call overhead. Keep SIMD kernels for future work and batch use, but
    // prefer the scalar reference for `Kernel::Auto`.
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(kernel, Kernel::Auto) && matches!(chosen, Kernel::Avx2 | Kernel::Avx512) {
        chosen = Kernel::Scalar;
    }
    Ok((data, first, short, long, len, chosen))
}

#[inline(always)]
fn apo_compute_into(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            // Keep scalar for correctness; EMA is sequential.
            // SIMD path removed as it incorrectly uses the same previous EMA for both lanes
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => apo_scalar(data, short, long, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => apo_avx2(data, short, long, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => apo_avx512(data, short, long, first, out),
            _ => unreachable!(),
        }
    }
}

pub fn apo_with_kernel(input: &ApoInput, kernel: Kernel) -> Result<ApoOutput, ApoError> {
    let (data, first, short, long, len, chosen) = apo_prepare(input, kernel)?;

    // Calculate warmup period: first valid data point
    let warmup_period = first;

    // Use zero-copy allocation with NaN prefix
    let mut out = alloc_with_nan_prefix(len, warmup_period);

    apo_compute_into(data, first, short, long, chosen, &mut out);

    Ok(ApoOutput { values: out })
}

/// Writes APO values into a caller-provided output slice without allocating.
///
/// - Preserves NaN warmups exactly like the Vec-returning API (prefix filled
///   with a quiet-NaN pattern).
/// - The `out` length must match the input length.
#[cfg(not(feature = "wasm"))]
pub fn apo_into(input: &ApoInput, out: &mut [f64]) -> Result<(), ApoError> {
    let (data, first, short, long, len, chosen) = apo_prepare(input, Kernel::Auto)?;
    if out.len() != len {
        return Err(ApoError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    // Prefill NaN warmup prefix using the same quiet-NaN pattern as allocations
    if first > 0 {
        for v in &mut out[..first] {
            *v = f64::from_bits(0x7ff8_0000_0000_0000);
        }
    }

    apo_compute_into(data, first, short, long, chosen, out);
    Ok(())
}

// --- Scalar Kernel

#[inline(always)]
pub fn apo_scalar(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    // Hoist constants and avoid repeated (1 - alpha) recomputation.
    let alpha_s = 2.0 / (short as f64 + 1.0);
    let alpha_l = 2.0 / (long as f64 + 1.0);
    let oma_s = 1.0 - alpha_s;
    let oma_l = 1.0 - alpha_l;

    let n = data.len();
    debug_assert_eq!(out.len(), n);

    // Initialize EMAs at the first valid element; prefix NaNs are already set by caller.
    let mut se = data[first];
    let mut le = se;
    out[first] = 0.0;

    // Main loop unrolled by 2 to reduce loop overhead while preserving sequential EMA deps.
    let mut i = first + 1;
    while i + 1 < n {
        let p0 = data[i];
        se = alpha_s * p0 + oma_s * se;
        le = alpha_l * p0 + oma_l * le;
        out[i] = se - le;

        let p1 = data[i + 1];
        se = alpha_s * p1 + oma_s * se;
        le = alpha_l * p1 + oma_l * le;
        out[i + 1] = se - le;

        i += 2;
    }

    // Tail element
    if i < n {
        let p = data[i];
        se = alpha_s * p + oma_s * se;
        le = alpha_l * p + oma_l * le;
        out[i] = se - le;
    }
}

// --- AVX2/AVX512 Kernels

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn apo_avx2(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    let alpha_s = 2.0 / (short as f64 + 1.0);
    let alpha_l = 2.0 / (long as f64 + 1.0);
    let oma_s = 1.0 - alpha_s;
    let oma_l = 1.0 - alpha_l;

    let n = data.len();
    debug_assert_eq!(out.len(), n);

    let mut i = first;
    let x0 = *data.get_unchecked(i);
    // EMA state replicated across lanes; initial se = le = x0 in all lanes
    let mut ema = _mm256_set_pd(x0, x0, x0, x0);

    // Pack constants as [al, as, al, as] and [1-al, 1-as, 1-al, 1-as]
    // Note: set_pd places args as [lane3, lane2, lane1, lane0]
    let a = _mm256_set_pd(alpha_l, alpha_s, alpha_l, alpha_s);
    let oma = _mm256_set_pd(oma_l, oma_s, oma_l, oma_s);

    // First output at the first valid index is always 0.0 (se-le)
    *out.get_unchecked_mut(i) = 0.0;
    i += 1;

    while i < n {
        // Broadcast current price
        let p = _mm256_set1_pd(*data.get_unchecked(i));

        // ema = alpha*price + (1-alpha)*ema; keep mul,mul,add order (no FMA) for parity
        let t1 = _mm256_mul_pd(a, p);
        let t2 = _mm256_mul_pd(oma, ema);
        ema = _mm256_add_pd(t1, t2);

        // Form se-le by swapping adjacent pairs in each 128-bit lane
        let swapped = _mm256_permute_pd(ema, 0x5);
        let diff = _mm256_sub_pd(ema, swapped);

        // Extract lane 0 (se-le)
        let apo_val = _mm256_cvtsd_f64(diff);
        *out.get_unchecked_mut(i) = apo_val;

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_avx512(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    // For APO the long/short split does not change mechanics; forward to one path.
    apo_avx512_short(data, short, long, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_avx512_short(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let alpha_s = 2.0 / (short as f64 + 1.0);
    let alpha_l = 2.0 / (long as f64 + 1.0);
    let oma_s = 1.0 - alpha_s;
    let oma_l = 1.0 - alpha_l;

    let n = data.len();
    debug_assert_eq!(out.len(), n);

    let mut i = first;
    let x0 = *data.get_unchecked(i);

    // Replicate se/le pair across lanes: start with all equal to x0
    let mut ema = _mm512_set_pd(x0, x0, x0, x0, x0, x0, x0, x0);

    // Pack constants as [al,as,al,as,...]
    let a = _mm512_set_pd(
        alpha_l, alpha_s, alpha_l, alpha_s, alpha_l, alpha_s, alpha_l, alpha_s,
    );
    let oma = _mm512_set_pd(oma_l, oma_s, oma_l, oma_s, oma_l, oma_s, oma_l, oma_s);

    *out.get_unchecked_mut(i) = 0.0;
    i += 1;

    while i < n {
        let p = _mm512_set1_pd(*data.get_unchecked(i));

        // ema = a*p + (1-a)*ema (mul,mul,add order; no FMA)
        let t1 = _mm512_mul_pd(a, p);
        let t2 = _mm512_mul_pd(oma, ema);
        ema = _mm512_add_pd(t1, t2);

        // Pairwise swap inside 128-bit lanes, then compute se-le
        let swapped = _mm512_permute_pd(ema, 0b01010101);
        let diff = _mm512_sub_pd(ema, swapped);

        // Extract low lane value
        let low128 = _mm512_castpd512_pd128(diff);
        let apo_val = _mm_cvtsd_f64(low128);
        *out.get_unchecked_mut(i) = apo_val;

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_avx512_long(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    // Identical mechanics; keep one highly-tuned body for both.
    apo_avx512_short(data, short, long, first, out);
}

// --- SIMD128 for WASM
// NOTE: This function is intentionally unused. EMA calculations are sequential (each value depends
// on the previous), so vectorizing two values at once using the same previous EMA would produce
// incorrect results. We fall back to scalar implementation for correctness.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn apo_simd128(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    use core::arch::wasm32::*;

    let alpha_short = 2.0 / (short as f64 + 1.0);
    let alpha_long = 2.0 / (long as f64 + 1.0);

    let one_minus_alpha_short = 1.0 - alpha_short;
    let one_minus_alpha_long = 1.0 - alpha_long;

    // Initialize EMAs with the first valid value
    let mut short_ema = data[first];
    let mut long_ema = data[first];

    // First value is always 0.0 (difference between identical EMAs)
    out[first] = 0.0;

    // Process values using SIMD where possible
    let alpha_short_vec = f64x2_splat(alpha_short);
    let alpha_long_vec = f64x2_splat(alpha_long);
    let one_minus_alpha_short_vec = f64x2_splat(one_minus_alpha_short);
    let one_minus_alpha_long_vec = f64x2_splat(one_minus_alpha_long);

    let mut i = first + 1;

    // Process pairs of values
    while i + 1 < data.len() {
        // Load two consecutive values
        let price_vec = v128_load(&data[i] as *const f64 as *const v128);

        // Create EMA vectors
        let short_ema_vec = f64x2_splat(short_ema);
        let long_ema_vec = f64x2_splat(long_ema);

        // Calculate new EMAs for both values
        let new_short_ema_vec = f64x2_add(
            f64x2_mul(alpha_short_vec, price_vec),
            f64x2_mul(one_minus_alpha_short_vec, short_ema_vec),
        );

        let new_long_ema_vec = f64x2_add(
            f64x2_mul(alpha_long_vec, price_vec),
            f64x2_mul(one_minus_alpha_long_vec, long_ema_vec),
        );

        // Calculate APO (short_ema - long_ema)
        let apo_vec = f64x2_sub(new_short_ema_vec, new_long_ema_vec);

        // Store results
        v128_store(&mut out[i] as *mut f64 as *mut v128, apo_vec);

        // Update EMAs for next iteration (use second value)
        short_ema = f64x2_extract_lane::<1>(new_short_ema_vec);
        long_ema = f64x2_extract_lane::<1>(new_long_ema_vec);

        i += 2;
    }

    // Handle remaining value if any
    if i < data.len() {
        let price = data[i];
        short_ema = alpha_short * price + one_minus_alpha_short * short_ema;
        long_ema = alpha_long * price + one_minus_alpha_long * long_ema;
        out[i] = short_ema - long_ema;
    }
}

// --- Batch, Streaming, and Builder APIs

// Decision: Streaming update uses parity-preserving form s = a*x + (1-a)*s_prev
// with precomputed (1-a) to reduce per-tick work. An opt-in update_fastmath()
// implements the delta+FMA form; default update() remains parity-aligned.

#[derive(Clone, Debug)]
pub struct ApoStream {
    short: usize,
    long: usize,
    alpha_short: f64,
    alpha_long: f64,
    // precomputed 1 - alpha terms
    oma_short: f64,
    oma_long: f64,
    short_ema: f64,
    long_ema: f64,
    filled: bool,
    nan_leading: usize,
    seen: usize,
}

impl ApoStream {
    #[inline(always)]
    pub fn try_new(params: ApoParams) -> Result<Self, ApoError> {
        let short = params.short_period.unwrap_or(10);
        let long = params.long_period.unwrap_or(20);
        if short == 0 || long == 0 {
            return Err(ApoError::InvalidPeriod { short, long });
        }
        if short >= long {
            return Err(ApoError::ShortPeriodNotLessThanLong { short, long });
        }
        // α = 2/(n+1)
        let alpha_short = 2.0 / (short as f64 + 1.0);
        let alpha_long = 2.0 / (long as f64 + 1.0);
        Ok(Self {
            short,
            long,
            alpha_short,
            alpha_long,
            oma_short: 1.0 - alpha_short,
            oma_long: 1.0 - alpha_long,
            short_ema: f64::NAN,
            long_ema: f64::NAN,
            filled: false,
            nan_leading: 0,
            seen: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64) -> Option<f64> {
        // Preserve leading-NaN semantics
        if !self.filled {
            if price.is_nan() {
                self.nan_leading += 1;
                return None;
            }
            self.short_ema = price;
            self.long_ema = price;
            self.filled = true;
            self.seen = 1;
            return Some(0.0);
        }

        self.seen += 1;

        // Mid-stream NaN: propagate and taint state (mirrors batch behavior)
        if price.is_nan() {
            self.short_ema = f64::NAN;
            self.long_ema = f64::NAN;
            return Some(f64::NAN);
        }

        // s = a*x + (1-a)*s_prev (parity-preserving order)
        let se_prev = self.short_ema;
        let le_prev = self.long_ema;
        self.short_ema = self.alpha_short * price + self.oma_short * se_prev;
        self.long_ema = self.alpha_long * price + self.oma_long * le_prev;
        Some(self.short_ema - self.long_ema)
    }

    /// Optional fast-math streaming update using delta form with mul_add (FMA where available).
    /// Default tests use `update`; use this when tiny rounding differences are acceptable.
    #[inline(always)]
    pub fn update_fastmath(&mut self, price: f64) -> Option<f64> {
        if !self.filled {
            if price.is_nan() {
                self.nan_leading += 1;
                return None;
            }
            self.short_ema = price;
            self.long_ema = price;
            self.filled = true;
            self.seen = 1;
            return Some(0.0);
        }

        self.seen += 1;

        if price.is_nan() {
            self.short_ema = f64::NAN;
            self.long_ema = f64::NAN;
            return Some(f64::NAN);
        }

        let ds = price - self.short_ema;
        let dl = price - self.long_ema;
        self.short_ema = ds.mul_add(self.alpha_short, self.short_ema);
        self.long_ema = dl.mul_add(self.alpha_long, self.long_ema);
        Some(self.short_ema - self.long_ema)
    }
}

/// Helper function for WASM bindings - writes directly to output slice with zero allocations
pub fn apo_into_slice(dst: &mut [f64], input: &ApoInput, kern: Kernel) -> Result<(), ApoError> {
    let (data, first, short, long, len, chosen) = apo_prepare(input, kern)?;
    if dst.len() != len {
        return Err(ApoError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }
    apo_compute_into(data, first, short, long, chosen, dst);
    for v in &mut dst[..first] {
        *v = f64::NAN;
    }
    Ok(())
}

// --- Batch Sweeping API
// Decision: Row-specific SIMD batch kernels not attempted here. With the current
// row-major output layout, cross-row SIMD would require scatter stores or a
// layout change to time-major blocks to be beneficial. Keep per-row kernels and
// revisit if batch layout changes to enable contiguous vector stores.

#[derive(Clone, Debug)]
pub struct ApoBatchRange {
    pub short: (usize, usize, usize),
    pub long: (usize, usize, usize),
}
impl Default for ApoBatchRange {
    fn default() -> Self {
        Self {
            short: (5, 20, 5),
            long: (15, 50, 5),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ApoBatchBuilder {
    range: ApoBatchRange,
    kernel: Kernel,
}
impl ApoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short = (start, end, step);
        self
    }
    pub fn short_static(mut self, s: usize) -> Self {
        self.range.short = (s, s, 0);
        self
    }
    pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long = (start, end, step);
        self
    }
    pub fn long_static(mut self, s: usize) -> Self {
        self.range.long = (s, s, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<ApoBatchOutput, ApoError> {
        apo_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ApoBatchOutput, ApoError> {
        ApoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ApoBatchOutput, ApoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ApoBatchOutput, ApoError> {
        ApoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct ApoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ApoParams>,
    pub rows: usize,
    pub cols: usize,
}
impl ApoBatchOutput {
    pub fn row_for_params(&self, p: &ApoParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_period.unwrap_or(10) == p.short_period.unwrap_or(10)
                && c.long_period.unwrap_or(20) == p.long_period.unwrap_or(20)
        })
    }
    pub fn values_for(&self, p: &ApoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

// --- Grid Expansion

#[inline(always)]
fn expand_grid(r: &ApoBatchRange) -> Result<Vec<ApoParams>, ApoError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, ApoError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                match cur.checked_add(step) {
                    Some(n) => cur = n,
                    None => break,
                }
            }
        } else {
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if let Some(n) = cur.checked_sub(step) { cur = n; } else { break; }
                if cur == usize::MAX { break; }
            }
        }
        if v.is_empty() {
            return Err(ApoError::InvalidRange { start, end, step });
        }
        Ok(v)
    }
    let shorts = axis(r.short)?;
    let longs = axis(r.long)?;
    let mut out = Vec::with_capacity(shorts.len().saturating_mul(longs.len()));
    for &s in &shorts {
        for &l in &longs {
            if s < l && s > 0 && l > 0 {
                out.push(ApoParams { short_period: Some(s), long_period: Some(l) });
            }
        }
    }
    Ok(out)
}

// --- Batch Slice API

#[inline(always)]
pub fn apo_batch_with_kernel(
    data: &[f64],
    sweep: &ApoBatchRange,
    k: Kernel,
) -> Result<ApoBatchOutput, ApoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(ApoError::InvalidKernelForBatch(other)),
    };
    apo_batch_par_slice(data, sweep, kernel)
}

#[inline(always)]
pub fn apo_batch_slice(
    data: &[f64],
    sweep: &ApoBatchRange,
    kern: Kernel,
) -> Result<ApoBatchOutput, ApoError> {
    apo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn apo_batch_par_slice(
    data: &[f64],
    sweep: &ApoBatchRange,
    kern: Kernel,
) -> Result<ApoBatchOutput, ApoError> {
    apo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn apo_batch_inner(
    data: &[f64],
    sweep: &ApoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ApoBatchOutput, ApoError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(ApoError::InvalidRange { start: sweep.short.0, end: sweep.short.1, step: sweep.short.2 });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ApoError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    if data.len() - first < max_long {
        return Err(ApoError::NotEnoughValidData {
            needed: max_long,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    // Checked arithmetic guard
    let _ = rows.checked_mul(cols).ok_or(ApoError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;

    // Step 1: Allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Step 2: Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|_c| {
            // For APO, warmup is simply the first valid data index
            first
        })
        .collect();

    // Step 3: Initialize NaN prefixes for each row
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Step 4: Convert to mutable slice for computation
    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    match kern {
        Kernel::Scalar | Kernel::ScalarBatch => {
            let do_row = |row: usize, out_row: &mut [f64]| unsafe {
                let s = combos[row].short_period.unwrap();
                let l = combos[row].long_period.unwrap();
                apo_row_scalar(data, first, s, l, out_row)
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    values
                        .par_chunks_mut(cols)
                        .enumerate()
                        .for_each(|(row, slice)| do_row(row, slice));
                }
                #[cfg(target_arch = "wasm32")]
                {
                    for (row, slice) in values.chunks_mut(cols).enumerate() {
                        do_row(row, slice);
                    }
                }
            } else {
                for (row, slice) in values.chunks_mut(cols).enumerate() {
                    do_row(row, slice);
                }
            }
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => {
            // Per-row AVX2 (existing behavior)
            let do_row = |row: usize, out_row: &mut [f64]| unsafe {
                let s = combos[row].short_period.unwrap();
                let l = combos[row].long_period.unwrap();
                apo_row_avx2(data, first, s, l, out_row)
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                values
                    .par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(row, slice)| do_row(row, slice));
                #[cfg(target_arch = "wasm32")]
                for (row, slice) in values.chunks_mut(cols).enumerate() {
                    do_row(row, slice);
                }
            } else {
                for (row, slice) in values.chunks_mut(cols).enumerate() {
                    do_row(row, slice);
                }
            }
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => {
            // Per-row AVX512 (existing behavior)
            let do_row = |row: usize, out_row: &mut [f64]| unsafe {
                let s = combos[row].short_period.unwrap();
                let l = combos[row].long_period.unwrap();
                apo_row_avx512(data, first, s, l, out_row)
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                values
                    .par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(row, slice)| do_row(row, slice));
                #[cfg(target_arch = "wasm32")]
                for (row, slice) in values.chunks_mut(cols).enumerate() {
                    do_row(row, slice);
                }
            } else {
                for (row, slice) in values.chunks_mut(cols).enumerate() {
                    do_row(row, slice);
                }
            }
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => {
            const LANES: usize = 4;
            let blocks = (rows + LANES - 1) / LANES;
            let do_block = |b: usize, blk: &mut [f64]| unsafe {
                let start_row = b * LANES;
                let end_row = usize::min(start_row + LANES, rows);
                apo_batch_rows_avx2(data, first, cols, &combos[start_row..end_row], blk);
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                values
                    .par_chunks_mut(cols * LANES)
                    .enumerate()
                    .for_each(|(b, blk)| do_block(b, blk));
                #[cfg(target_arch = "wasm32")]
                for (b, blk) in values.chunks_mut(cols * LANES).enumerate() {
                    do_block(b, blk);
                }
            } else {
                for (b, blk) in values.chunks_mut(cols * LANES).enumerate() {
                    do_block(b, blk);
                }
            }
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => {
            const LANES: usize = 8;
            let blocks = (rows + LANES - 1) / LANES;
            let do_block = |b: usize, blk: &mut [f64]| unsafe {
                let start_row = b * LANES;
                let end_row = usize::min(start_row + LANES, rows);
                apo_batch_rows_avx512(data, first, cols, &combos[start_row..end_row], blk);
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                values
                    .par_chunks_mut(cols * LANES)
                    .enumerate()
                    .for_each(|(b, blk)| do_block(b, blk));
                #[cfg(target_arch = "wasm32")]
                for (b, blk) in values.chunks_mut(cols * LANES).enumerate() {
                    do_block(b, blk);
                }
            } else {
                for (b, blk) in values.chunks_mut(cols * LANES).enumerate() {
                    do_block(b, blk);
                }
            }
        }
        _ => unreachable!(),
    }

    // Step 6: Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(ApoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn apo_batch_inner_into(
    data: &[f64],
    sweep: &ApoBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<ApoParams>, ApoError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(ApoError::InvalidRange { start: sweep.short.0, end: sweep.short.1, step: sweep.short.2 });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ApoError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    if data.len() - first < max_long {
        return Err(ApoError::NotEnoughValidData {
            needed: max_long,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let expected = rows.checked_mul(cols).ok_or(ApoError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;
    if out.len() != expected {
        return Err(ApoError::OutputLengthMismatch { expected, got: out.len() });
    }

    // 1) Treat caller buffer as uninitialized and set NaN warm prefixes with helper.
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm = vec![first; rows];
    init_matrix_prefixes(out_mu, cols, &warm);

    // 2) Compute into initialized region; support row-optimized batch kernels
    match kern {
        Kernel::Scalar | Kernel::ScalarBatch => {
            let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
                let s = combos[row].short_period.unwrap();
                let l = combos[row].long_period.unwrap();
                let dst: &mut [f64] =
                    core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
                apo_row_scalar(data, first, s, l, dst)
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                out_mu
                    .par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(r, s)| do_row(r, s));
                #[cfg(target_arch = "wasm32")]
                for (r, s) in out_mu.chunks_mut(cols).enumerate() {
                    do_row(r, s);
                }
            } else {
                for (r, s) in out_mu.chunks_mut(cols).enumerate() {
                    do_row(r, s);
                }
            }
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => {
            let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
                let s = combos[row].short_period.unwrap();
                let l = combos[row].long_period.unwrap();
                let dst: &mut [f64] =
                    core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
                apo_row_avx2(data, first, s, l, dst)
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                out_mu
                    .par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(r, s)| do_row(r, s));
                #[cfg(target_arch = "wasm32")]
                for (r, s) in out_mu.chunks_mut(cols).enumerate() {
                    do_row(r, s);
                }
            } else {
                for (r, s) in out_mu.chunks_mut(cols).enumerate() {
                    do_row(r, s);
                }
            }
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => {
            let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
                let s = combos[row].short_period.unwrap();
                let l = combos[row].long_period.unwrap();
                let dst: &mut [f64] =
                    core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
                apo_row_avx512(data, first, s, l, dst)
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                out_mu
                    .par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(r, s)| do_row(r, s));
                #[cfg(target_arch = "wasm32")]
                for (r, s) in out_mu.chunks_mut(cols).enumerate() {
                    do_row(r, s);
                }
            } else {
                for (r, s) in out_mu.chunks_mut(cols).enumerate() {
                    do_row(r, s);
                }
            }
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => {
            const LANES: usize = 4;
            let do_block = |b: usize, blk_mu: &mut [MaybeUninit<f64>]| unsafe {
                let start_row = b * LANES;
                let end_row = usize::min(start_row + LANES, rows);
                let blk: &mut [f64] =
                    core::slice::from_raw_parts_mut(blk_mu.as_mut_ptr() as *mut f64, blk_mu.len());
                apo_batch_rows_avx2(data, first, cols, &combos[start_row..end_row], blk);
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                out_mu
                    .par_chunks_mut(cols * LANES)
                    .enumerate()
                    .for_each(|(b, blk)| do_block(b, blk));
                #[cfg(target_arch = "wasm32")]
                for (b, blk) in out_mu.chunks_mut(cols * LANES).enumerate() {
                    do_block(b, blk);
                }
            } else {
                for (b, blk) in out_mu.chunks_mut(cols * LANES).enumerate() {
                    do_block(b, blk);
                }
            }
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => {
            const LANES: usize = 8;
            let do_block = |b: usize, blk_mu: &mut [MaybeUninit<f64>]| unsafe {
                let start_row = b * LANES;
                let end_row = usize::min(start_row + LANES, rows);
                let blk: &mut [f64] =
                    core::slice::from_raw_parts_mut(blk_mu.as_mut_ptr() as *mut f64, blk_mu.len());
                apo_batch_rows_avx512(data, first, cols, &combos[start_row..end_row], blk);
            };
            if parallel {
                #[cfg(not(target_arch = "wasm32"))]
                out_mu
                    .par_chunks_mut(cols * LANES)
                    .enumerate()
                    .for_each(|(b, blk)| do_block(b, blk));
                #[cfg(target_arch = "wasm32")]
                for (b, blk) in out_mu.chunks_mut(cols * LANES).enumerate() {
                    do_block(b, blk);
                }
            } else {
                for (b, blk) in out_mu.chunks_mut(cols * LANES).enumerate() {
                    do_block(b, blk);
                }
            }
        }
        _ => unreachable!(),
    }

    Ok(combos)
}

// --- Row Kernels

#[inline(always)]
pub unsafe fn apo_row_scalar(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    apo_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn apo_row_avx2(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
    apo_avx2(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_row_avx512(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    if long <= 32 {
        apo_row_avx512_short(data, first, short, long, out)
    } else {
        apo_row_avx512_long(data, first, short, long, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_row_avx512_short(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    apo_avx512_short(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_row_avx512_long(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    apo_avx512_long(data, short, long, first, out)
}

// --- Row-optimized batch kernels (vectorize across parameter rows)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn apo_batch_rows_avx2(
    data: &[f64],
    first: usize,
    cols: usize,
    combos_block: &[ApoParams],
    out_block: &mut [f64],
) {
    use core::arch::x86_64::*;
    let lanes = 4usize;
    let l = combos_block.len();

    // Precompute per-lane constants
    let mut as_arr = [0.0f64; 4];
    let mut al_arr = [0.0f64; 4];
    let mut os_arr = [1.0f64; 4];
    let mut ol_arr = [1.0f64; 4];
    for (j, p) in combos_block.iter().enumerate() {
        let s = p.short_period.unwrap_or(10);
        let g = p.long_period.unwrap_or(20);
        let a_s = 2.0 / (s as f64 + 1.0);
        let a_l = 2.0 / (g as f64 + 1.0);
        as_arr[j] = a_s;
        al_arr[j] = a_l;
        os_arr[j] = 1.0 - a_s;
        ol_arr[j] = 1.0 - a_l;
    }
    let a_s = _mm256_setr_pd(as_arr[0], as_arr[1], as_arr[2], as_arr[3]);
    let a_l = _mm256_setr_pd(al_arr[0], al_arr[1], al_arr[2], al_arr[3]);
    let o_s = _mm256_setr_pd(os_arr[0], os_arr[1], os_arr[2], os_arr[3]);
    let o_l = _mm256_setr_pd(ol_arr[0], ol_arr[1], ol_arr[2], ol_arr[3]);

    // EMA states per lane
    let x0 = *data.get_unchecked(first);
    let mut se = _mm256_set1_pd(x0);
    let mut le = _mm256_set1_pd(x0);

    // First output for each row is 0.0 at index `first`
    for j in 0..l {
        *out_block.get_unchecked_mut(j * cols + first) = 0.0;
    }
    let mut i = first + 1;
    while i < cols {
        let p = _mm256_set1_pd(*data.get_unchecked(i));
        // se = a_s*p + (1-a_s)*se; le = a_l*p + (1-a_l)*le
        let se1 = _mm256_add_pd(_mm256_mul_pd(a_s, p), _mm256_mul_pd(o_s, se));
        let le1 = _mm256_add_pd(_mm256_mul_pd(a_l, p), _mm256_mul_pd(o_l, le));
        se = se1;
        le = le1;

        let diff = _mm256_sub_pd(se, le);
        let mut tmp: [f64; 4] = [0.0; 4];
        _mm256_storeu_pd(tmp.as_mut_ptr(), diff);
        for j in 0..l {
            *out_block.get_unchecked_mut(j * cols + i) = tmp[j];
        }
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn apo_batch_rows_avx512(
    data: &[f64],
    first: usize,
    cols: usize,
    combos_block: &[ApoParams],
    out_block: &mut [f64],
) {
    use core::arch::x86_64::*;
    let lanes = 8usize;
    let l = combos_block.len();

    // Precompute per-lane constants
    let mut as_arr = [0.0f64; 8];
    let mut al_arr = [0.0f64; 8];
    let mut os_arr = [1.0f64; 8];
    let mut ol_arr = [1.0f64; 8];
    for (j, p) in combos_block.iter().enumerate() {
        let s = p.short_period.unwrap_or(10);
        let g = p.long_period.unwrap_or(20);
        let a_s = 2.0 / (s as f64 + 1.0);
        let a_l = 2.0 / (g as f64 + 1.0);
        as_arr[j] = a_s;
        al_arr[j] = a_l;
        os_arr[j] = 1.0 - a_s;
        ol_arr[j] = 1.0 - a_l;
    }
    let a_s = _mm512_setr_pd(
        as_arr[0], as_arr[1], as_arr[2], as_arr[3], as_arr[4], as_arr[5], as_arr[6], as_arr[7],
    );
    let a_l = _mm512_setr_pd(
        al_arr[0], al_arr[1], al_arr[2], al_arr[3], al_arr[4], al_arr[5], al_arr[6], al_arr[7],
    );
    let o_s = _mm512_setr_pd(
        os_arr[0], os_arr[1], os_arr[2], os_arr[3], os_arr[4], os_arr[5], os_arr[6], os_arr[7],
    );
    let o_l = _mm512_setr_pd(
        ol_arr[0], ol_arr[1], ol_arr[2], ol_arr[3], ol_arr[4], ol_arr[5], ol_arr[6], ol_arr[7],
    );

    let x0 = *data.get_unchecked(first);
    let mut se = _mm512_set1_pd(x0);
    let mut le = _mm512_set1_pd(x0);

    for j in 0..l {
        *out_block.get_unchecked_mut(j * cols + first) = 0.0;
    }
    let mut i = first + 1;
    while i < cols {
        let p = _mm512_set1_pd(*data.get_unchecked(i));
        let se1 = _mm512_add_pd(_mm512_mul_pd(a_s, p), _mm512_mul_pd(o_s, se));
        let le1 = _mm512_add_pd(_mm512_mul_pd(a_l, p), _mm512_mul_pd(o_l, le));
        se = se1;
        le = le1;

        let diff = _mm512_sub_pd(se, le);
        let mut tmp: [f64; 8] = [0.0; 8];
        _mm512_storeu_pd(tmp.as_mut_ptr(), diff);
        for j in 0..l {
            *out_block.get_unchecked_mut(j * cols + i) = tmp[j];
        }
        i += 1;
    }
}

// --- Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_apo_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Build a small but non-trivial input with a NaN warmup prefix
        let mut data: Vec<f64> = Vec::with_capacity(256);
        for _ in 0..5 {
            data.push(f64::NAN);
        }
        for i in 0..251 {
            // Mix of linear trend and mild oscillation
            let x = i as f64;
            data.push(100.0 + 0.1 * x + (x * 0.05).sin());
        }

        let input = ApoInput::from_slice(&data, ApoParams::default());

        // Baseline via existing Vec-returning API
        let baseline = apo(&input)?.values;

        // Preallocate output and call the new into API
        let mut out = vec![0.0; data.len()];
        apo_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        // Equality helper: NaN == NaN or exact equality; allow tiny epsilon fallback
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for (i, (a, b)) in baseline.iter().copied().zip(out.iter().copied()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "mismatch at index {}: api={} into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }

    fn check_apo_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ApoParams {
            short_period: None,
            long_period: None,
        };
        let input = ApoInput::from_candles(&candles, "close", default_params);
        let output = apo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_apo_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ApoInput::with_default_candles(&candles);
        let result = apo_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -429.80100015922653,
            -401.64149983850075,
            -386.13569657357584,
            -357.92775222467753,
            -374.13870680232503,
        ];
        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "[{}] APO value mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected_last_five[i],
                value
            );
        }
        Ok(())
    }

    fn check_apo_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ApoInput::with_default_candles(&candles);
        match input.data {
            ApoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected ApoData::Candles"),
        }
        let output = apo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_apo_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ApoParams {
            short_period: Some(0),
            long_period: Some(20),
        };
        let input = ApoInput::from_slice(&input_data, params);
        let res = apo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] APO should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_apo_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty_data: Vec<f64> = vec![];
        let params = ApoParams::default();
        let input = ApoInput::from_slice(&empty_data, params);
        let result = apo_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_apo_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = ApoParams::default();

        // Batch calculation
        let input = ApoInput::from_candles(&candles, "close", params.clone());
        let batch_result = apo_with_kernel(&input, kernel)?;

        // Streaming calculation
        let mut stream = ApoStream::try_new(params)?;
        let mut streaming_results = vec![];

        for &close in &candles.close {
            if let Some(val) = stream.update(close) {
                streaming_results.push(val);
            } else {
                streaming_results.push(f64::NAN);
            }
        }

        // Compare results (allowing for small floating point differences)
        assert_eq!(batch_result.values.len(), streaming_results.len());
        let first_valid = candles.close.iter().position(|x| !x.is_nan()).unwrap_or(0);

        for i in first_valid..batch_result.values.len() {
            if !batch_result.values[i].is_nan() && !streaming_results[i].is_nan() {
                let diff = (batch_result.values[i] - streaming_results[i]).abs();
                assert!(
                    diff < 1e-10,
                    "Streaming mismatch at index {}: batch={}, stream={}",
                    i,
                    batch_result.values[i],
                    streaming_results[i]
                );
            }
        }
        Ok(())
    }

    fn check_apo_period_invalid(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ApoParams {
            short_period: Some(20),
            long_period: Some(10),
        };
        let input = ApoInput::from_slice(&data_small, params);
        let res = apo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] APO should fail with invalid period",
            test_name
        );
        Ok(())
    }

    fn check_apo_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ApoParams {
            short_period: Some(9),
            long_period: Some(10),
        };
        let input = ApoInput::from_slice(&single_point, params);
        let res = apo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] APO should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_apo_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = ApoParams {
            short_period: Some(10),
            long_period: Some(20),
        };
        let first_input = ApoInput::from_candles(&candles, "close", first_params);
        let first_result = apo_with_kernel(&first_input, kernel)?;
        let second_params = ApoParams {
            short_period: Some(5),
            long_period: Some(15),
        };
        let second_input = ApoInput::from_slice(&first_result.values, second_params);
        let second_result = apo_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_apo_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ApoInput::from_candles(
            &candles,
            "close",
            ApoParams {
                short_period: Some(10),
                long_period: Some(20),
            },
        );
        let res = apo_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 30 {
            for (i, &val) in res.values[30..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    30 + i
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_apo_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = ApoInput::from_candles(&candles, "close", ApoParams::default());
        let output = apo_with_kernel(&input, kernel)?;

        for (i, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();

            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_apo_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_apo_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy 1: Random price data with realistic period ranges
        let random_data_strat = (3usize..=20, 10usize..=50)
            .prop_filter("short < long", |(s, l)| s < l)
            .prop_flat_map(|(short_period, long_period)| {
                // Generate data length that's sufficient for the long period
                let len = long_period * 2..400;
                (
                    prop::collection::vec(
                        // Generate realistic price data
                        (10f64..10000f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    Just(short_period),
                    Just(long_period),
                    Just("random"),
                )
            });

        // Strategy 2: Constant data (APO should stay near 0)
        let constant_data_strat = (3usize..=20, 10usize..=50)
            .prop_filter("short < long", |(s, l)| s < l)
            .prop_flat_map(|(short_period, long_period)| {
                let len = long_period * 2..200;
                (
                    prop::collection::vec(Just(100.0f64), len),
                    Just(short_period),
                    Just(long_period),
                    Just("constant"),
                )
            });

        // Strategy 3: Trending data (monotonic increase/decrease)
        let trending_data_strat = (3usize..=20, 10usize..=50)
            .prop_filter("short < long", |(s, l)| s < l)
            .prop_flat_map(|(short_period, long_period)| {
                let len = long_period * 2..200;
                (
                    (50..150usize).prop_flat_map(move |size| {
                        (0.1f64..5.0).prop_map(move |slope| {
                            (0..size)
                                .map(|i| 100.0 + slope * i as f64)
                                .collect::<Vec<f64>>()
                        })
                    }),
                    Just(short_period),
                    Just(long_period),
                    Just("trending"),
                )
            });

        // Combine all strategies
        let strat = prop_oneof![random_data_strat, constant_data_strat, trending_data_strat,];

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, short_period, long_period, data_type)| {
                let params = ApoParams {
                    short_period: Some(short_period),
                    long_period: Some(long_period),
                };
                let input = ApoInput::from_slice(&data, params.clone());

                let result = apo_with_kernel(&input, kernel);
                prop_assert!(result.is_ok(), "APO computation failed: {:?}", result);

                let ApoOutput { values: out } = result.unwrap();

                // Property 1: Output length matches input
                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                // Property 2: First valid value should be 0.0 (both EMAs start with same value)
                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                if first_valid < data.len() {
                    prop_assert!(
                        out[first_valid].abs() < 1e-10,
                        "First APO value should be 0, got {} at index {}",
                        out[first_valid],
                        first_valid
                    );
                }

                // Property 3: All values after warmup should be finite
                for i in first_valid..out.len() {
                    prop_assert!(
                        out[i].is_finite(),
                        "APO output at index {} should be finite, got {}",
                        i,
                        out[i]
                    );
                }

                // Property 4: Values should be bounded reasonably
                // APO is the difference between two EMAs, both bounded by data range
                // In practice, APO magnitude is much smaller than data range
                let data_min = data
                    .iter()
                    .filter(|x| x.is_finite())
                    .fold(f64::INFINITY, |a, &b| a.min(b));
                let data_max = data
                    .iter()
                    .filter(|x| x.is_finite())
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let data_range = data_max - data_min;

                // Tighter bound: APO typically doesn't exceed 30% of data range
                let apo_bound = data_range * 0.3;

                for i in first_valid..out.len() {
                    prop_assert!(
                        out[i].abs() <= apo_bound,
                        "APO value at index {} exceeds expected bound: {} > {}",
                        i,
                        out[i].abs(),
                        apo_bound
                    );
                }

                // Property 5: Data-type specific validations
                match data_type {
                    "constant" => {
                        // For constant data, APO should remain very close to 0
                        for i in first_valid..out.len() {
                            prop_assert!(
                                out[i].abs() < 1e-9,
                                "APO should be ~0 for constant data, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                    "trending" => {
                        // For monotonically increasing data, APO should generally be positive after initial warmup
                        // (short EMA responds faster, stays above long EMA)
                        if data.len() > long_period * 2 {
                            let check_start = first_valid + long_period;
                            let check_end = out.len();
                            if check_start < check_end {
                                // Check if data is increasing or decreasing
                                let is_increasing = data[first_valid] < data[data.len() - 1];

                                // Count positive vs negative values in the latter part
                                let positive_count = out[check_start..check_end]
                                    .iter()
                                    .filter(|&&v| v > 0.0)
                                    .count();
                                let total_count = check_end - check_start;

                                if is_increasing {
                                    // For uptrend, expect mostly positive APO values
                                    prop_assert!(
										positive_count > total_count / 2,
										"APO should be mostly positive for uptrend, got {} positive out of {}",
										positive_count,
										total_count
									);
                                } else {
                                    // For downtrend, expect mostly negative APO values
                                    prop_assert!(
										positive_count < total_count / 2,
										"APO should be mostly negative for downtrend, got {} positive out of {}",
										positive_count,
										total_count
									);
                                }
                            }
                        }
                    }
                    _ => {} // Random data, no specific pattern expected
                }

                // Property 6: Manual calculation verification for first few values
                if data.len() >= 3 && first_valid + 2 < data.len() {
                    let alpha_short = 2.0 / (short_period as f64 + 1.0);
                    let alpha_long = 2.0 / (long_period as f64 + 1.0);

                    // First value after warmup
                    let mut short_ema = data[first_valid];
                    let mut long_ema = data[first_valid];
                    let expected_first = 0.0; // Both EMAs start equal
                    prop_assert!(
                        (out[first_valid] - expected_first).abs() < 1e-9,
                        "First value mismatch: expected {}, got {}",
                        expected_first,
                        out[first_valid]
                    );

                    // Second value
                    if first_valid + 1 < data.len() {
                        let price = data[first_valid + 1];
                        short_ema = alpha_short * price + (1.0 - alpha_short) * short_ema;
                        long_ema = alpha_long * price + (1.0 - alpha_long) * long_ema;
                        let expected_second = short_ema - long_ema;
                        prop_assert!(
                            (out[first_valid + 1] - expected_second).abs() < 1e-9,
                            "Second value mismatch: expected {}, got {}",
                            expected_second,
                            out[first_valid + 1]
                        );
                    }

                    // Third value
                    if first_valid + 2 < data.len() {
                        let price = data[first_valid + 2];
                        short_ema = alpha_short * price + (1.0 - alpha_short) * short_ema;
                        long_ema = alpha_long * price + (1.0 - alpha_long) * long_ema;
                        let expected_third = short_ema - long_ema;
                        prop_assert!(
                            (out[first_valid + 2] - expected_third).abs() < 1e-9,
                            "Third value mismatch: expected {}, got {}",
                            expected_third,
                            out[first_valid + 2]
                        );
                    }
                }

                // Property 7: Kernel consistency - all kernels should produce the same result
                let ref_output = apo_with_kernel(&input, Kernel::Scalar);
                prop_assert!(ref_output.is_ok(), "Reference scalar computation failed");
                let ApoOutput { values: ref_out } = ref_output.unwrap();

                for (i, (&val, &ref_val)) in out.iter().zip(ref_out.iter()).enumerate() {
                    if !val.is_finite() || !ref_val.is_finite() {
                        // Both should be NaN at the same positions
                        prop_assert_eq!(
                            val.is_nan(),
                            ref_val.is_nan(),
                            "NaN mismatch at index {}: kernel={}, scalar={}",
                            i,
                            val,
                            ref_val
                        );
                    } else {
                        // Values should be nearly identical (allowing for small floating-point differences)
                        let diff = (val - ref_val).abs();
                        let ulp_diff = val.to_bits().abs_diff(ref_val.to_bits());
                        prop_assert!(
                            diff <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch at index {}: {} vs {} (diff: {}, ULP: {})",
                            i,
                            val,
                            ref_val,
                            diff,
                            ulp_diff
                        );
                    }
                }

                // Property 8: Period relationship
                prop_assert!(
                    short_period < long_period,
                    "Short period must be less than long period"
                );

                // Property 9: Streaming consistency
                let mut stream = ApoStream::try_new(params).unwrap();
                let mut stream_values = Vec::new();
                for &price in &data {
                    if let Some(val) = stream.update(price) {
                        stream_values.push(val);
                    } else {
                        stream_values.push(f64::NAN);
                    }
                }

                // Compare streaming vs batch (allowing for floating-point differences)
                for i in first_valid..out.len() {
                    if out[i].is_finite() && stream_values[i].is_finite() {
                        let diff = (out[i] - stream_values[i]).abs();
                        prop_assert!(
                            diff < 1e-10,
                            "Streaming mismatch at index {}: batch={}, stream={}, diff={}",
                            i,
                            out[i],
                            stream_values[i],
                            diff
                        );
                    }
                }

                Ok(())
            })
            .map_err(|e| e.into())
    }

    macro_rules! generate_all_apo_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    generate_all_apo_tests!(
        check_apo_partial_params,
        check_apo_accuracy,
        check_apo_default_candles,
        check_apo_zero_period,
        check_apo_empty_input,
        check_apo_streaming,
        check_apo_period_invalid,
        check_apo_very_small_dataset,
        check_apo_reinput,
        check_apo_nan_handling,
        check_apo_no_poison,
        check_apo_property
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = ApoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = ApoParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (short_start, short_end, short_step, long_start, long_end, long_step)
            (2, 10, 2, 15, 30, 5),   // Small to medium periods
            (5, 25, 5, 30, 50, 10),  // Medium periods
            (10, 20, 5, 25, 45, 10), // Common trading periods
            (12, 12, 0, 26, 26, 0),  // Static periods (default-like)
            (3, 9, 3, 10, 20, 5),    // Small range sweep
        ];

        for (cfg_idx, &(s_start, s_end, s_step, l_start, l_end, l_step)) in
            test_configs.iter().enumerate()
        {
            let output = ApoBatchBuilder::new()
                .kernel(kernel)
                .short_range(s_start, s_end, s_step)
                .long_range(l_start, l_end, l_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short={}, long={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_period.unwrap_or(12),
                        combo.long_period.unwrap_or(26)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short={}, long={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_period.unwrap_or(12),
                        combo.long_period.unwrap_or(26)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short={}, long={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_period.unwrap_or(12),
                        combo.long_period.unwrap_or(26)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_apo_simd128_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let short_period = 3;
        let long_period = 5;
        let params = ApoParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
        };
        let input = ApoInput::from_slice(&data, params);

        // Force scalar kernel (which will use SIMD128 on WASM)
        let scalar_output = apo_with_kernel(&input, Kernel::Scalar).unwrap();

        // Create a pure scalar version for comparison
        let mut pure_scalar_output = vec![f64::NAN; data.len()];
        let first = 0; // APO starts from first data point
        unsafe {
            apo_scalar(
                &data,
                short_period,
                long_period,
                first,
                &mut pure_scalar_output,
            );
        }

        // Compare results
        assert_eq!(scalar_output.values.len(), pure_scalar_output.len());
        for (i, (simd_val, scalar_val)) in scalar_output
            .values
            .iter()
            .zip(pure_scalar_output.iter())
            .enumerate()
        {
            if scalar_val.is_nan() {
                assert!(simd_val.is_nan(), "SIMD128 NaN mismatch at index {}", i);
            } else {
                assert!(
                    (scalar_val - simd_val).abs() < 1e-10,
                    "SIMD128 mismatch at index {}: scalar={}, simd128={}",
                    i,
                    scalar_val,
                    simd_val
                );
            }
        }
    }
}

// ================================================================================================
// Python Bindings
// ================================================================================================

#[cfg(feature = "python")]
#[pyfunction(name = "apo")]
#[pyo3(signature = (data, short_period=10, long_period=20, kernel=None))]
pub fn apo_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    short_period: usize,
    long_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = ApoParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let apo_in = ApoInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function and convert to NumPy with zero-copy
    let result_vec: Vec<f64> = py
        .allow_threads(|| apo_with_kernel(&apo_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ApoStream")]
pub struct ApoStreamPy {
    stream: ApoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ApoStreamPy {
    #[new]
    fn new(short_period: usize, long_period: usize) -> PyResult<Self> {
        let params = ApoParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
        };
        let stream =
            ApoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(ApoStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated APO value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "apo_batch")]
#[pyo3(signature = (data, short_period_range, long_period_range, kernel=None))]
pub fn apo_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    short_period_range: (usize, usize, usize),
    long_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = ApoBatchRange {
        short: short_period_range,
        long: long_period_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if combos.is_empty() {
        return Err(PyValueError::new_err("No valid parameter combinations"));
    }
    let rows = combos.len();
    let cols = slice_in.len();
    // Checked arithmetic for total elements
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflow"))?;

    // Pre-allocate uninitialized NumPy array
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Warm-prefix init using helper on a MaybeUninit overlay
    let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(
            slice_out.as_mut_ptr() as *mut MaybeUninit<f64>,
            slice_out.len(),
        )
    };
    let warm: Vec<usize> = std::iter::repeat(first).take(rows).collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    // Heavy work without the GIL, zero-copy into NumPy buffer
    let combos = py
        .allow_threads(|| {
            let k = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match k {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                _ => Kernel::Scalar,
            };
            apo_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "short_periods",
        combos
            .iter()
            .map(|p| p.short_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "long_periods",
        combos
            .iter()
            .map(|p| p.long_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

// ==================== PYTHON: CUDA BINDINGS (zero-copy) ====================
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context as CudaContext;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32Apo", unsendable)]
pub struct DeviceArrayF32ApoPy {
    pub(crate) inner: Option<crate::cuda::moving_averages::apo_wrapper::DeviceArrayF32>,
    stream_handle: usize,
    _ctx_guard: Arc<CudaContext>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32ApoPy {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(pyo3::exceptions::PyTypeError::new_err("use factory methods from CUDA functions"))
    }

    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let d = PyDict::new(py);
        let itemsize = std::mem::size_of::<f32>();
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (inner.cols * itemsize, itemsize))?;
        let size = inner.rows.saturating_mul(inner.cols);
        let ptr_val: usize =
            if size == 0 { 0 } else { inner.buf.as_device_ptr().as_raw() as usize };
        d.set_item("data", (ptr_val, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        Ok((2, self._device_id as i32))
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
        let (kdl, alloc_dev) = self.__dlpack_device__()?;
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
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;
        let crate::cuda::moving_averages::apo_wrapper::DeviceArrayF32 { buf, rows, cols, .. } =
            inner;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "apo_cuda_batch_dev")]
#[pyo3(signature = (data_f32, short_range=(10,10,0), long_range=(20,20,0), device_id=0))]
pub fn apo_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    short_range: (usize, usize, usize),
    long_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32ApoPy> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaApo;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = data_f32.as_slice()?;
    let sweep = ApoBatchRange {
        short: short_range,
        long: long_range,
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaApo::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.apo_batch_dev(slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let ctx = inner.ctx();
    let dev_id = inner.device_id();
    Ok(DeviceArrayF32ApoPy { inner: Some(inner), stream_handle: 0, _ctx_guard: ctx, _device_id: dev_id })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "apo_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, short_period, long_period, device_id=0))]
pub fn apo_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    short_period: usize,
    long_period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32ApoPy> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaApo;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    if short_period == 0 || long_period == 0 || short_period >= long_period {
        return Err(PyValueError::new_err("invalid short/long period"));
    }
    let flat = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = ApoParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaApo::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.apo_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let ctx = inner.ctx();
    let dev_id = inner.device_id();
    Ok(DeviceArrayF32ApoPy { inner: Some(inner), stream_handle: 0, _ctx_guard: ctx, _device_id: dev_id })
}

// ================================================================================================
// WASM Bindings
// ================================================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_js(data: &[f64], short_period: usize, long_period: usize) -> Result<Vec<f64>, JsValue> {
    let params = ApoParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let input = ApoInput::from_slice(data, params);

    // Single allocation following WASM guide pattern
    let mut output = vec![0.0; data.len()];

    apo_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    short_period: usize,
    long_period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer passed to apo_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = ApoParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
        };
        let input = ApoInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            apo_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            apo_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_batch_js(
    data: &[f64],
    short_period_start: usize,
    short_period_end: usize,
    short_period_step: usize,
    long_period_start: usize,
    long_period_end: usize,
    long_period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = ApoBatchRange {
        short: (short_period_start, short_period_end, short_period_step),
        long: (long_period_start, long_period_end, long_period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    apo_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_batch_metadata_js(
    short_period_start: usize,
    short_period_end: usize,
    short_period_step: usize,
    long_period_start: usize,
    long_period_end: usize,
    long_period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = ApoBatchRange {
        short: (short_period_start, short_period_end, short_period_step),
        long: (long_period_start, long_period_end, long_period_step),
    };

    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut metadata = Vec::with_capacity(combos.len() * 2);

    for combo in combos {
        metadata.push(combo.short_period.unwrap() as f64);
        metadata.push(combo.long_period.unwrap() as f64);
    }

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    short_start: usize,
    short_end: usize,
    short_step: usize,
    long_start: usize,
    long_end: usize,
    long_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer passed to apo_batch_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = ApoBatchRange {
            short: (short_start, short_end, short_step),
            long: (long_start, long_end, long_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
        apo_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ApoBatchConfig {
    pub short_period_range: (usize, usize, usize),
    pub long_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ApoBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ApoParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = apo_batch)]
pub fn apo_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: ApoBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = ApoBatchRange {
        short: cfg.short_period_range,
        long: cfg.long_period_range,
    };
    let out = apo_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = ApoBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}
