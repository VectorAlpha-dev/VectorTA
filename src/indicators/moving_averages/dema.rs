//! # Double Exponential Moving Average (DEMA)
//!
//! A moving average technique that seeks to reduce lag by combining two
//! exponential moving averages (EMA). First, an EMA is calculated on the input
//! data. Then, a second EMA is computed on the first EMA. Finally, the DEMA is
//! determined by subtracting the second EMA from twice the first EMA.
//!
//! ## Parameters
//! - **period**: Lookback period for the EMA calculations (must be ≥ 1).
//!
//! ## Errors
//! - **AllValuesNaN**: DEMA: All input data values are `NaN`.
//! - **InvalidPeriod**: DEMA: `period` is less than 1 or exceeds the data length.
//! - **NotEnoughData**: DEMA: Not enough data points (needs `2 * (period - 1)`).
//!
//! ## Returns
//! - **`Ok(DemaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(DemaError)`** otherwise.
//!
//! ## Developer Notes
//! - SIMD selection: AVX512 enabled by default (faster by >5% at 100k on supported CPUs).
//!   AVX2 underperforms on typical targets, so Auto skips it and falls back to `Scalar`.
//!   Users can still explicitly request `Avx2`/`Avx512` via the `kernel` parameter.
//! - Streaming update: O(1) EMA updates; warmup handled by caller via NaN prefix.
//! - Memory: uses allocation helpers with warmup prefixes (see ALMA pattern).
//! - Future: Revisit SIMD if layout or math changes improve throughput.
//!
//! Decision: Streaming kernel uses fused multiply-add on x86/x86_64 via `mul_add` for
//! parity with scalar/batch paths and improved stability; non-x86 uses `a*b + c`.
//! Warmup semantics unchanged; counters use saturating add to avoid wrap.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::{CudaDema, DeviceArrayF32};
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
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum DemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for DemaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DemaData::Slice(slice) => slice,
            DemaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct DemaParams {
    pub period: Option<usize>,
}

impl Default for DemaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct DemaInput<'a> {
    pub data: DemaData<'a>,
    pub params: DemaParams,
}

impl<'a> DemaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: DemaParams) -> Self {
        Self {
            data: DemaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: DemaParams) -> Self {
        Self {
            data: DemaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DemaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Debug, Clone)]
pub struct DemaOutput {
    pub values: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct DemaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for DemaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DemaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<DemaOutput, DemaError> {
        let p = DemaParams {
            period: self.period,
        };
        let i = DemaInput::from_candles(c, "close", p);
        dema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DemaOutput, DemaError> {
        let p = DemaParams {
            period: self.period,
        };
        let i = DemaInput::from_slice(d, p);
        dema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<DemaStream, DemaError> {
        let p = DemaParams {
            period: self.period,
        };
        DemaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DemaError {
    #[error("dema: Input data slice is empty.")]
    EmptyInputData,
    #[error("dema: All values are NaN.")]
    AllValuesNaN,
    #[error("dema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dema: Not enough data: needed = {needed}, valid = {valid}")]
    NotEnoughData { needed: usize, valid: usize },
    #[error("dema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dema: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("dema: invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("dema: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("dema: size overflow when computing {context}")]
    SizeOverflow { context: &'static str },
}

#[inline]
pub fn dema(input: &DemaInput) -> Result<DemaOutput, DemaError> {
    dema_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn dema_prepare<'a>(
    input: &'a DemaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), DemaError> {
    let data: &[f64] = match &input.data {
        DemaData::Candles { candles, source } => source_type(candles, source),
        DemaData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(DemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DemaError::AllValuesNaN)?;

    let period = input.get_period();

    if period < 1 || period > len {
        return Err(DemaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let needed = 2 * (period - 1);
    if len < needed {
        return Err(DemaError::NotEnoughData { needed, valid: len });
    }
    let valid = len - first;
    if valid < needed {
        return Err(DemaError::NotEnoughValidData { needed, valid });
    }

    
    
    let chosen = match kernel {
        Kernel::Auto => match detect_best_kernel() {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => Kernel::Avx512,
            _ => Kernel::Scalar,
        },
        other => other,
    };

    let warm = first + period - 1;

    Ok((data, period, first, warm, chosen))
}

#[inline(always)]
fn dema_compute_into(data: &[f64], period: usize, first: usize, chosen: Kernel, out: &mut [f64]) {
    unsafe {
        match chosen {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dema_avx512(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dema_avx2(data, period, first, out),
            _ => dema_scalar(data, period, first, out),
        }
    }
}

pub fn dema_with_kernel(input: &DemaInput, kernel: Kernel) -> Result<DemaOutput, DemaError> {
    let (data, period, first, warm, chosen) = dema_prepare(input, kernel)?;
    let len = data.len();
    let mut out = alloc_with_nan_prefix(len, warm);
    dema_compute_into(data, period, first, chosen, &mut out);
    
    out[..warm].fill(f64::NAN);
    Ok(DemaOutput { values: out })
}

/// Writes DEMA values into a caller-provided output buffer without allocating.
///
/// - Preserves the same NaN warmup prefix as `dema()` (up to `first + period - 1`).
/// - The output slice length must equal the input length; returns the module's
///   existing error on mismatch.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn dema_into(input: &DemaInput, out: &mut [f64]) -> Result<(), DemaError> {
    // Delegate to the internal zero-copy helper with Kernel::Auto
    dema_into_slice(out, input, Kernel::Auto)
}

/// Write DEMA values directly to output slice - no allocations.
/// This is the core helper for WASM optimizations.
/// The output slice must be the same length as the input data.
#[inline]
pub fn dema_into_slice(dst: &mut [f64], input: &DemaInput, kern: Kernel) -> Result<(), DemaError> {
    let (data, period, first, warmup, chosen) = dema_prepare(input, kern)?;

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(DemaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    // Compute DEMA values directly into dst
    dema_compute_into(data, period, first, chosen, dst);

    // Fill warmup period with NaN (dema_compute_into might not handle this)
    for v in &mut dst[..warmup] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
#[inline]
pub unsafe fn dema_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    debug_assert!(period >= 1 && data.len() == out.len());
    let n = data.len();
    if first >= n {
        return;
    }

    // α and (1-α)
    let alpha = 2.0 / (period as f64 + 1.0);
    let a = 1.0 - alpha;

    // Seed at the first non-NaN
    let mut ema1 = *data.get_unchecked(first);
    let mut ema2 = ema1;
    *out.get_unchecked_mut(first) = ema1;

    // Start one past the seed
    let mut i = first + 1;
    let mut p = data.as_ptr().add(i);
    let mut q = out.as_mut_ptr().add(i);

    // Unroll by 4 for better ILP; prefetch ahead to hide memory latency.
    // We only enter the unrolled loop when we have at least 4 samples left.
    let limit = n.saturating_sub(4);
    while i <= limit {
        // Prefetch upcoming cachelines (guarded)
        if i + 32 < n {
            core::arch::x86_64::_mm_prefetch(
                p.add(32) as *const i8,
                core::arch::x86_64::_MM_HINT_T0,
            );
        }

        // step 0
        let x0 = *p;
        ema1 = ema1.mul_add(a, x0 * alpha);
        ema2 = ema2.mul_add(a, ema1 * alpha);
        *q = ema1.mul_add(2.0, -ema2);

        // step 1
        let x1 = *p.add(1);
        ema1 = ema1.mul_add(a, x1 * alpha);
        ema2 = ema2.mul_add(a, ema1 * alpha);
        *q.add(1) = ema1.mul_add(2.0, -ema2);

        // step 2
        let x2 = *p.add(2);
        ema1 = ema1.mul_add(a, x2 * alpha);
        ema2 = ema2.mul_add(a, ema1 * alpha);
        *q.add(2) = ema1.mul_add(2.0, -ema2);

        // step 3
        let x3 = *p.add(3);
        ema1 = ema1.mul_add(a, x3 * alpha);
        ema2 = ema2.mul_add(a, ema1 * alpha);
        *q.add(3) = ema1.mul_add(2.0, -ema2);

        p = p.add(4);
        q = q.add(4);
        i += 4;
    }

    // Tail
    while i < n {
        let x = *p;
        ema1 = ema1.mul_add(a, x * alpha);
        ema2 = ema2.mul_add(a, ema1 * alpha);
        *q = ema1.mul_add(2.0, -ema2);

        p = p.add(1);
        q = q.add(1);
        i += 1;
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub unsafe fn dema_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    debug_assert!(period >= 1 && data.len() == out.len());
    let n = data.len();
    if first >= n {
        return;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let a = 1.0 - alpha;

    let mut ema1 = *data.get_unchecked(first);
    let mut ema2 = ema1;
    *out.get_unchecked_mut(first) = ema1;

    let mut i = first + 1;
    let mut p = data.as_ptr().add(i);
    let mut q = out.as_mut_ptr().add(i);

    // Unrolled by 4 (portable path uses plain mul/add)
    let limit = n.saturating_sub(4);
    while i <= limit {
        let x0 = *p;
        ema1 = ema1 * a + x0 * alpha;
        ema2 = ema2 * a + ema1 * alpha;
        *q = 2.0 * ema1 - ema2;

        let x1 = *p.add(1);
        ema1 = ema1 * a + x1 * alpha;
        ema2 = ema2 * a + ema1 * alpha;
        *q.add(1) = 2.0 * ema1 - ema2;

        let x2 = *p.add(2);
        ema1 = ema1 * a + x2 * alpha;
        ema2 = ema2 * a + ema1 * alpha;
        *q.add(2) = 2.0 * ema1 - ema2;

        let x3 = *p.add(3);
        ema1 = ema1 * a + x3 * alpha;
        ema2 = ema2 * a + ema1 * alpha;
        *q.add(3) = 2.0 * ema1 - ema2;

        p = p.add(4);
        q = q.add(4);
        i += 4;
    }

    while i < n {
        let x = *p;
        ema1 = ema1 * a + x * alpha;
        ema2 = ema2 * a + ema1 * alpha;
        *q = 2.0 * ema1 - ema2;

        p = p.add(1);
        q = q.add(1);
        i += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DEMA AVX2 / AVX512 kernels (drop‑in)
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn last_lane_256(v: __m256d) -> f64 {
    let hi: __m128d = _mm256_extractf128_pd(v, 1); // lanes [2,3]
    let dup_hi: __m128d = _mm_unpackhi_pd(hi, hi); // [lane3, lane3]
    _mm_cvtsd_f64(dup_hi)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn last_lane_512(v: __m512d) -> f64 {
    let hi2: __m128d = _mm512_extractf64x2_pd(v, 3); // lanes [6,7]
    let dup_hi: __m128d = _mm_unpackhi_pd(hi2, hi2); // [lane7, lane7]
    _mm_cvtsd_f64(dup_hi)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn shl1_256(x: __m256d) -> __m256d {
    // [0, x0, x1, x2]
    let lo: __m128d = _mm256_castpd256_pd128(x); // [x0,x1]
    let hi: __m128d = _mm256_extractf128_pd(x, 1); // [x2,x3]
    let lo_res = _mm_unpacklo_pd(_mm_setzero_pd(), lo); // [0,x0]
    let hi_res = _mm_shuffle_pd(
        _mm_unpackhi_pd(lo, lo), // [x1,x1]
        _mm_unpacklo_pd(hi, hi),
        0x0,
    ); // [x1,x2]
    _mm256_insertf128_pd(_mm256_castpd128_pd256(lo_res), hi_res, 1)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn shl2_256(x: __m256d) -> __m256d {
    // [0, 0, x0, x1]
    let lo: __m128d = _mm256_castpd256_pd128(x);
    _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_setzero_pd()), lo, 1)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn scan4(v: __m256d, a1: __m256d, a2: __m256d) -> __m256d {
    // t = v; t += a1*shl1(t); t += a2*shl2(t);
    let t1 = _mm256_fmadd_pd(a1, shl1_256(v), v);
    let t2 = _mm256_fmadd_pd(a2, shl2_256(t1), t1);
    t2
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn shl1_512(x: __m512d) -> __m512d {
    // [0, x0, x1, x2, x3, x4, x5, x6]
    let idx: __m512i = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 0);
    let mask: __mmask8 = 0b1111_1110;
    _mm512_maskz_permutexvar_pd(mask, idx, x)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn shl2_512(x: __m512d) -> __m512d {
    // [0, 0, x0, x1, x2, x3, x4, x5]
    let idx: __m512i = _mm512_set_epi64(5, 4, 3, 2, 1, 0, 0, 0);
    let mask: __mmask8 = 0b1111_1100;
    _mm512_maskz_permutexvar_pd(mask, idx, x)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn shl4_512(x: __m512d) -> __m512d {
    // [0, 0, 0, 0, x0, x1, x2, x3]
    let idx: __m512i = _mm512_set_epi64(3, 2, 1, 0, 0, 0, 0, 0);
    let mask: __mmask8 = 0b1111_0000;
    _mm512_maskz_permutexvar_pd(mask, idx, x)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn scan8(v: __m512d, a1: __m512d, a2: __m512d, a4: __m512d) -> __m512d {
    // t = v; t += a1*shl1(t); t += a2*shl2(t); t += a4*shl4(t);
    let t1 = _mm512_fmadd_pd(a1, shl1_512(v), v);
    let t2 = _mm512_fmadd_pd(a2, shl2_512(t1), t1);
    let t3 = _mm512_fmadd_pd(a4, shl4_512(t2), t2);
    t3
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn dema_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    debug_assert!(data.len() == out.len());
    if first >= data.len() {
        return;
    }

    let n = data.len();
    let alpha = 2.0 / (period as f64 + 1.0);
    let a = 1.0 - alpha;

    // initial state at first finite sample
    let mut i = first;
    let mut ema1 = *data.get_unchecked(i);
    let mut ema2 = ema1;
    *out.get_unchecked_mut(i) = ema1;
    i += 1;
    if i >= n {
        return;
    }

    // precompute constants
    let alpha_v = _mm256_set1_pd(alpha);
    let a1_s = a;
    let a2_s = a1_s * a1_s;
    let a3_s = a2_s * a1_s;
    let a4_s = a2_s * a2_s;
    let pow_vec = _mm256_set_pd(a4_s, a3_s, a2_s, a1_s);
    let a1_v = _mm256_set1_pd(a1_s);
    let a2_v = _mm256_set1_pd(a2_s);

    // main blocks of 4
    while i + 4 <= n {
        // load inputs
        let x = _mm256_loadu_pd(data.as_ptr().add(i));
        // EMA1 block
        let v1 = _mm256_mul_pd(alpha_v, x);
        let t1 = scan4(v1, a1_v, a2_v);
        let prev1 = _mm256_set1_pd(ema1);
        let ema1_vec = _mm256_fmadd_pd(pow_vec, prev1, t1);

        // EMA2 block
        let v2 = _mm256_mul_pd(alpha_v, ema1_vec);
        let t2 = scan4(v2, a1_v, a2_v);
        let prev2 = _mm256_set1_pd(ema2);
        let ema2_vec = _mm256_fmadd_pd(pow_vec, prev2, t2);

        // DEMA = 2*EMA1 - EMA2
        let two_ema1 = _mm256_add_pd(ema1_vec, ema1_vec);
        let dema_v = _mm256_sub_pd(two_ema1, ema2_vec);
        _mm256_storeu_pd(out.as_mut_ptr().add(i), dema_v);

        // carry state to next block (last lane)
        ema1 = last_lane_256(ema1_vec);
        ema2 = last_lane_256(ema2_vec);
        i += 4;
    }

    // scalar tail
    while i < n {
        let price = *data.get_unchecked(i);
        ema1 = ema1.mul_add(a, price * alpha);
        ema2 = ema2.mul_add(a, ema1 * alpha);
        *out.get_unchecked_mut(i) = (2.0 * ema1) - ema2;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
#[inline]
pub unsafe fn dema_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    debug_assert!(data.len() == out.len());
    if first >= data.len() {
        return;
    }

    let n = data.len();
    let alpha = 2.0 / (period as f64 + 1.0);
    let a = 1.0 - alpha;

    // initial state
    let mut i = first;
    let mut ema1 = *data.get_unchecked(i);
    let mut ema2 = ema1;
    *out.get_unchecked_mut(i) = ema1;
    i += 1;
    if i >= n {
        return;
    }

    // constants
    let alpha_v = _mm512_set1_pd(alpha);
    let a1_s = a;
    let a2_s = a1_s * a1_s;
    let a3_s = a2_s * a1_s;
    let a4_s = a2_s * a2_s;
    let a5_s = a4_s * a1_s;
    let a6_s = a3_s * a3_s; // a^6
    let a7_s = a6_s * a1_s; // a^7
    let a8_s = a4_s * a4_s; // a^8
    let pow_vec = _mm512_set_pd(a8_s, a7_s, a6_s, a5_s, a4_s, a3_s, a2_s, a1_s);
    let a1_v = _mm512_set1_pd(a1_s);
    let a2_v = _mm512_set1_pd(a2_s);
    let a4_v = _mm512_set1_pd(a4_s);

    while i + 8 <= n {
        let x = _mm512_loadu_pd(data.as_ptr().add(i));

        // EMA1 block
        let v1 = _mm512_mul_pd(alpha_v, x);
        let t1 = scan8(v1, a1_v, a2_v, a4_v);
        let prev1 = _mm512_set1_pd(ema1);
        let ema1_vec = _mm512_fmadd_pd(pow_vec, prev1, t1);

        // EMA2 block
        let v2 = _mm512_mul_pd(alpha_v, ema1_vec);
        let t2 = scan8(v2, a1_v, a2_v, a4_v);
        let prev2 = _mm512_set1_pd(ema2);
        let ema2_vec = _mm512_fmadd_pd(pow_vec, prev2, t2);

        // DEMA
        let two_ema1 = _mm512_add_pd(ema1_vec, ema1_vec);
        let dema_v = _mm512_sub_pd(two_ema1, ema2_vec);
        _mm512_storeu_pd(out.as_mut_ptr().add(i), dema_v);

        // carry
        ema1 = last_lane_512(ema1_vec);
        ema2 = last_lane_512(ema2_vec);
        i += 8;
    }

    // scalar tail
    while i < n {
        let price = *data.get_unchecked(i);
        ema1 = ema1.mul_add(a, price * alpha);
        ema2 = ema2.mul_add(a, ema1 * alpha);
        *out.get_unchecked_mut(i) = (2.0 * ema1) - ema2;
        i += 1;
    }
}

#[derive(Debug, Clone)]
pub struct DemaStream {
    period: usize,
    alpha: f64,
    alpha_1: f64,
    ema: f64,
    ema2: f64,
    filled: usize,
    nan_fill: usize,
}

impl DemaStream {
    pub fn try_new(params: DemaParams) -> Result<Self, DemaError> {
        let period = params.period.unwrap_or(30);
        if period < 1 {
            return Err(DemaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            alpha: 2.0 / (period as f64 + 1.0),
            alpha_1: 1.0 - 2.0 / (period as f64 + 1.0),
            ema: f64::NAN,
            ema2: f64::NAN,
            filled: 0,
            nan_fill: period - 1,
        })
    }

    #[inline(always)]
    fn fmadd(a: f64, b: f64, c: f64) -> f64 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            a.mul_add(b, c)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            a * b + c
        }
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Fast seed: identical semantics to original.
        if self.filled == 0 {
            self.ema = value;
            self.ema2 = value;
            self.filled = 1;
            // For period==1, return immediately; otherwise warmup with None.
            return if self.nan_fill == 0 {
                Some(value)
            } else {
                None
            };
        }

        // Lift constants into registers
        let a = self.alpha; // α
        let a1 = self.alpha_1; // 1−α

        // EMA1: ema = ema*(1-α) + value*α
        self.ema = Self::fmadd(self.ema, a1, value * a);

        // EMA2: ema2 = ema2*(1-α) + ema*α
        self.ema2 = Self::fmadd(self.ema2, a1, self.ema * a);

        // DEMA = 2*EMA1 - EMA2; encourage fused MAD on x86
        let y = Self::fmadd(self.ema, 2.0, -self.ema2);

        // Advance with saturating add to avoid wrap on extremely long runs
        self.filled = self.filled.saturating_add(1);

        // Warm‑up policy unchanged: emit only after period−1
        if self.filled > self.nan_fill {
            Some(y)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn update_nan(&mut self, value: f64) -> f64 {
        match self.update(value) {
            Some(v) => v,
            None => f64::NAN,
        }
    }
}

// Optional fast-math helper for reciprocal if many alphas must be computed.
// Not used by default to preserve exactness; keep available for future tuning.
#[inline(always)]
fn fast_recip_nr1(d: f64) -> f64 {
    // Precondition: d > 0
    let x0 = (d as f32).recip() as f64; // fast initial approx
    x0 * (2.0 - d * x0) // one NR step
}

#[derive(Clone, Debug)]
pub struct DemaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for DemaBatchRange {
    fn default() -> Self {
        Self {
            period: (30, 279, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DemaBatchBuilder {
    range: DemaBatchRange,
    kernel: Kernel,
}

impl DemaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<DemaBatchOutput, DemaError> {
        dema_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DemaBatchOutput, DemaError> {
        DemaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DemaBatchOutput, DemaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<DemaBatchOutput, DemaError> {
        DemaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub struct DemaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DemaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl DemaBatchOutput {
    pub fn row_for_params(&self, p: &DemaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(30) == p.period.unwrap_or(30))
    }
    pub fn values_for(&self, p: &DemaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DemaBatchRange) -> Vec<DemaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        // Zero step or equal bounds -> singleton
        if step == 0 || start == end {
            return vec![start];
        }
        let mut vals = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                vals.push(v);
                match v.checked_add(step) {
                    Some(n) if n != v => v = n,
                    _ => break,
                }
            }
        } else {
            // Support reversed bounds: descend using checked_sub
            let mut v = start;
            loop {
                vals.push(v);
                if v <= end { break; }
                match v.checked_sub(step) {
                    Some(n) if n != v => v = n,
                    _ => break,
                }
            }
        }
        vals
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(DemaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn dema_batch_slice(
    data: &[f64],
    sweep: &DemaBatchRange,
    kern: Kernel,
) -> Result<DemaBatchOutput, DemaError> {
    dema_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn dema_batch_par_slice(
    data: &[f64],
    sweep: &DemaBatchRange,
    kern: Kernel,
) -> Result<DemaBatchOutput, DemaError> {
    dema_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
pub(crate) fn dema_batch_with_kernel(
    data: &[f64],
    sweep: &DemaBatchRange,
    k: Kernel,
) -> Result<DemaBatchOutput, DemaError> {
    // Row-specific batch kernels not implemented for DEMA: little shared precompute across rows.
    // Auto short-circuits to ScalarBatch.
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(DemaError::InvalidKernelForBatch(other)),
    };
    // Keep parity and performance policy consistent with single-path:
    // - AVX512: use AVX512 (wins consistently)
    // - AVX2: prefer Scalar (AVX2 underperforms on typical targets and single Auto uses Scalar)
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Scalar,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    dema_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn dema_batch_inner(
    data: &[f64],
    sweep: &DemaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DemaBatchOutput, DemaError> {
    let combos = {
        let v = expand_grid(sweep);
        if v.is_empty() {
            return Err(DemaError::InvalidRange {
                start: sweep.period.0,
                end: sweep.period.1,
                step: sweep.period.2,
            });
        }
        v
    };
    let cols = data.len();
    let rows = combos.len();
    // Checked arithmetic to avoid overflow before allocation
    let _total = rows
        .checked_mul(cols)
        .ok_or(DemaError::SizeOverflow { context: "rows*cols for batch buffer" })?;

    if cols == 0 {
        return Err(DemaError::EmptyInputData);
    }

    // Allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            data.iter()
                .position(|x| !x.is_nan())
                .unwrap_or(0)
                .saturating_add(c.period.unwrap().saturating_sub(1))
        })
        .collect();

    // Initialize NaN prefixes efficiently
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Convert to mutable slice using ManuallyDrop pattern
    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Perform computation
    dema_batch_inner_into(data, sweep, kern, parallel, out)?;

    // Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(DemaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn dema_batch_inner_into(
    data: &[f64],
    sweep: &DemaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DemaParams>, DemaError> {
    // ── 1. validation ──────────────────────────────────────────────────────
    let combos = {
        let v = expand_grid(sweep);
        if v.is_empty() {
            return Err(DemaError::InvalidRange {
                start: sweep.period.0,
                end: sweep.period.1,
                step: sweep.period.2,
            });
        }
        v
    };

    if data.is_empty() {
        return Err(DemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DemaError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = 2 * (max_p - 1);
    if data.len() < needed {
        return Err(DemaError::NotEnoughData {
            needed,
            valid: data.len(),
        });
    }
    let valid = data.len() - first;
    if valid < needed {
        return Err(DemaError::NotEnoughValidData { needed, valid });
    }

    let rows = combos.len();
    let cols = data.len();

    // Verify output buffer size
    let expected = rows
        .checked_mul(cols)
        .ok_or(DemaError::SizeOverflow { context: "rows*cols when validating output buffer" })?;
    if out.len() != expected {
        return Err(DemaError::OutputLengthMismatch { expected, got: out.len() });
    }

    // ── 3. per-row kernel closure; dst is &mut [f64] ─────
    let do_row = |row: usize, dst: &mut [f64]| unsafe {
        let p = combos[row].period.unwrap();

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dema_row_avx512(data, first, p, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dema_row_avx2(data, first, p, dst),
            _ => dema_row_scalar(data, first, p, dst),
        }
        // Enforce warmup NaNs for this row
        let warm = first + p - 1;
        dst[..warm].fill(f64::NAN);
    };

    // ── 4. run every row kernel, parallel or sequential ────────────────────
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

    Ok(combos)
}
#[inline(always)]
unsafe fn dema_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dema_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dema_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dema_avx2(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn dema_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dema_avx512(data, period, first, out)
}

// ==================== PYTHON: Device handle with CAI v3 + DLPack ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32Dema", unsendable)]
pub struct DeviceArrayF32DemaPy {
    pub(crate) inner: DeviceArrayF32,
    _ctx_guard: std::sync::Arc<cust::context::Context>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32DemaPy {
    #[new]
    fn py_new() -> PyResult<Self> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "use factory methods from CUDA functions",
        ))
    }

    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        let itemsize = std::mem::size_of::<f32>();
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (self.inner.cols * itemsize, itemsize))?;
        let size = self.inner.rows.saturating_mul(self.inner.cols);
        let ptr_val: usize = if size == 0 { 0 } else { self.inner.buf.as_device_ptr().as_raw() as usize };
        d.set_item("data", (ptr_val, false))?;
        d.set_item("version", 3)?; 
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) { (2, self._device_id as i32) }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<pyo3::PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

        
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

        
        let dummy = cust::memory::DeviceBuffer::from_slice(&[])
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
impl DeviceArrayF32DemaPy {
    pub fn new(inner: DeviceArrayF32, ctx_guard: std::sync::Arc<cust::context::Context>, device_id: u32) -> Self {
        Self { inner, _ctx_guard: ctx_guard, _device_id: device_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use proptest::prelude::*;

    fn check_dema_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = DemaParams { period: None };
        let input_default = DemaInput::from_candles(&candles, "close", default_params);
        let output_default = dema_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = DemaParams { period: Some(14) };
        let input_period_14 = DemaInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = dema_with_kernel(&input_period_14, kernel)?;
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = DemaParams { period: Some(20) };
        let input_custom = DemaInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = dema_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = DemaInput::with_default_candles(&candles);
        let result = dema_with_kernel(&input, kernel)?;

        let expected_last_five = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
        ];
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-6,
                "DEMA mismatch at index {}: expected {}, got {}",
                start_index + i,
                exp,
                val
            );
        }
        Ok(())
    }

    fn check_dema_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DemaInput::with_default_candles(&candles);
        match input.data {
            DemaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected DemaData::Candles"),
        }
        assert_eq!(input.params.period, Some(30));
        let output = dema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dema_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = DemaParams { period: Some(0) };
        let input = DemaInput::from_slice(&input_data, params);
        let result = dema_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dema_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = DemaParams { period: Some(10) };
        let input = DemaInput::from_slice(&data_small, params);
        let result = dema_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = DemaParams { period: Some(9) };
        let input = DemaInput::from_slice(&single_point, params);
        let result = dema_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = DemaParams { period: Some(80) };
        let first_input = DemaInput::from_candles(&candles, "close", first_params);
        let first_result = dema_with_kernel(&first_input, kernel)?;

        let second_params = DemaParams { period: Some(60) };
        let second_input = DemaInput::from_slice(&first_result.values, second_params);
        let second_result = dema_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(!second_result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_dema_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = DemaParams { period: Some(30) };
        let input = DemaInput::from_candles(&candles, "close", params);
        let result = dema_with_kernel(&input, kernel)?;
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_dema_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = DemaInput::from_slice(&empty, DemaParams::default());
        let res = dema_with_kernel(&input, kernel);
        assert!(matches!(res, Err(DemaError::EmptyInputData)));
        Ok(())
    }

    fn check_dema_not_enough_valid(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, 1.0, 2.0];
        let params = DemaParams { period: Some(3) };
        let input = DemaInput::from_slice(&data, params);
        let res = dema_with_kernel(&input, kernel);
        assert!(matches!(res, Err(DemaError::NotEnoughValidData { .. })));
        Ok(())
    }

    #[allow(clippy::float_cmp)]
    fn check_dema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use float_cmp::approx_eq;
        use proptest::prelude::*;

        skip_if_unsupported!(kernel, test_name);

        
        
        let strat = (1usize..=32).prop_flat_map(|period| {
            let min_len = 2 * period.max(2); 
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    min_len..400,
                ),
                Just(period),
                (-1e3f64..1e3f64).prop_filter("non-zero scale", |a| a.is_finite() && *a != 0.0),
                -1e3f64..1e3f64, 
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, a, b)| {
                let params = DemaParams {
                    period: Some(period),
                };
                let input = DemaInput::from_slice(&data, params.clone());

                
                let fast = dema_with_kernel(&input, kernel);
                let slow = dema_with_kernel(&input, Kernel::Scalar);

                match (fast, slow) {
                    
                    (Err(e1), Err(e2))
                        if std::mem::discriminant(&e1) == std::mem::discriminant(&e2) =>
                    {
                        return Ok(())
                    }
                    
                    (Err(e1), Err(e2)) => {
                        prop_assert!(false, "different errors: fast={:?} slow={:?}", e1, e2)
                    }
                    
                    (Err(e1), Ok(_)) => {
                        prop_assert!(false, "fast errored {e1:?} but scalar succeeded")
                    }
                    (Ok(_), Err(e2)) => {
                        prop_assert!(false, "scalar errored {e2:?} but fast succeeded")
                    }

                    
                    (Ok(fast), Ok(reference)) => {
                        let DemaOutput { values: out } = fast;
                        let DemaOutput { values: rref } = reference;

                        
                        
                        let mut stream = DemaStream::try_new(params.clone()).unwrap();
                        let mut s_out = Vec::with_capacity(data.len());
                        for &v in &data {
                            s_out.push(stream.update(v).unwrap_or(f64::NAN));
                        }

                        
                        let transformed: Vec<f64> = data.iter().map(|x| a * *x + b).collect();
                        let t_out =
                            dema(&DemaInput::from_slice(&transformed, params.clone()))?.values;

                        
                        let nan_fill = period - 1; 
                        for i in 0..data.len() {
                            let y = out[i];
                            let yr = rref[i];
                            let ys = s_out[i];
                            let yt = t_out[i];

                            
                            if period == 1 && y.is_finite() {
                                prop_assert!(approx_eq!(f64, y, data[i], ulps = 2));
                            }

                            
                            
                            if i >= period - 1 {
                                let window = &data[i.saturating_sub(period - 1)..=i];
                                if window.iter().all(|v| *v == window[0]) {
                                    prop_assert!(approx_eq!(f64, y, window[0], epsilon = 1e-9));
                                }
                            } else {
                                
                                prop_assert!(y.is_nan(), "Expected NaN during warmup at index {i}");
                            }

                            
                            if i >= nan_fill {
                                
                                if y.is_finite() {
                                    let expected = a * y + b;
                                    let diff = (yt - expected).abs();
                                    
                                    
                                    
                                    let tol = 1e-7_f64.max(expected.abs() * 1e-9);
                                    let ulp = yt.to_bits().abs_diff(expected.to_bits());
                                    prop_assert!(
                                        diff <= tol || ulp <= 8,
                                        "idx {i}: affine mismatch diff={diff:e}  ULP={ulp}"
                                    );
                                } else {
                                    prop_assert_eq!(
                                        y.to_bits(),
                                        yt.to_bits(),
                                        "idx {}: special-value mismatch under affine map",
                                        i
                                    );
                                }
                            }

                            
                            let ulp = y.to_bits().abs_diff(yr.to_bits());
                            prop_assert!(
                                (y - yr).abs() <= 1e-9 || ulp <= 4,
                                "idx {i}: fast={y} ref={yr} ULP={ulp}"
                            );

                            
                            
                            
                            
                            if period == 1 {
                                
                                prop_assert!(
									(y - ys).abs() <= 1e-9 || (y.is_nan() && ys.is_nan()),
									"idx {i}: stream mismatch for period=1 - batch={y}, stream={ys}"
								);
                            } else if i < period - 1 {
                                
                                prop_assert!(
                                    ys.is_nan(),
                                    "idx {i}: stream should return NaN during warmup, got {ys}"
                                );
                            } else {
                                
                                prop_assert!(
                                    (y - ys).abs() <= 1e-9 || (y.is_nan() && ys.is_nan()),
                                    "idx {i}: stream mismatch - batch={y}, stream={ys}"
                                );
                            }
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        
        assert!(dema(&DemaInput::from_slice(&[], DemaParams::default())).is_err());
        assert!(dema(&DemaInput::from_slice(
            &[f64::NAN; 12],
            DemaParams::default()
        ))
        .is_err());
        assert!(dema(&DemaInput::from_slice(
            &[1.0; 5],
            DemaParams { period: Some(12) }
        ))
        .is_err());
        assert!(dema(&DemaInput::from_slice(
            &[1.0; 5],
            DemaParams { period: Some(0) }
        ))
        .is_err());

        Ok(())
    }

    fn check_dema_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 30;
        let input = DemaInput::from_candles(
            &candles,
            "close",
            DemaParams {
                period: Some(period),
            },
        );
        let batch_output = dema_with_kernel(&input, kernel)?.values;

        let mut stream = DemaStream::try_new(DemaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            stream_values.push(stream.update(price).unwrap_or(f64::NAN));
        }

        assert_eq!(batch_output.len(), stream_values.len());

        for (i, (&b, &s)) in batch_output
            .iter()
            .zip(&stream_values)
            .enumerate()
            .skip(period)
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }

            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] DEMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_dema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        
        let test_params = vec![
            
            DemaParams::default(),
            
            DemaParams { period: Some(2) },
            DemaParams { period: Some(3) },
            DemaParams { period: Some(5) },
            
            DemaParams { period: Some(7) },
            DemaParams { period: Some(10) },
            DemaParams { period: Some(12) },
            DemaParams { period: Some(20) },
            DemaParams { period: Some(30) },
            
            DemaParams { period: Some(50) },
            DemaParams { period: Some(100) },
            DemaParams { period: Some(200) },
            
            DemaParams { period: Some(1) },
            DemaParams { period: Some(250) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DemaInput::from_candles(&candles, "close", params.clone());
            let output = dema_with_kernel(&input, kernel)?;

            
            for (i, &val) in output.values.iter().enumerate() {
                
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(30)
                    );
                }

                
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(30)
                    );
                }

                
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(30)
                    );
                }
            }
        }

        Ok(())
    }

    
    #[cfg(not(debug_assertions))]
    fn check_dema_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_dema_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    fn check_dema_warmup_nan_preservation(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        
        let test_periods = vec![10, 20, 30, 50];

        for period in test_periods {
            let params = DemaParams {
                period: Some(period),
            };
            let input = DemaInput::from_candles(&candles, "close", params);
            let result = dema_with_kernel(&input, kernel)?;

            
            let warmup = period - 1;
            for i in 0..warmup {
                assert!(
                    result.values[i].is_nan(),
                    "[{}] Expected NaN at index {} (warmup={}) for period={}, got {}",
                    test_name,
                    i,
                    warmup,
                    period,
                    result.values[i]
                );
            }

            
            for i in warmup..warmup + 10 {
                assert!(
                    !result.values[i].is_nan(),
                    "[{}] Expected non-NaN at index {} (warmup={}) for period={}, got NaN",
                    test_name,
                    i,
                    warmup,
                    period
                );
            }
        }
        Ok(())
    }

    generate_all_dema_tests!(
        check_dema_partial_params,
        check_dema_accuracy,
        check_dema_default_candles,
        check_dema_zero_period,
        check_dema_period_exceeds_length,
        check_dema_very_small_dataset,
        check_dema_empty_input,
        check_dema_not_enough_valid,
        check_dema_reinput,
        check_dema_nan_handling,
        check_dema_streaming,
        check_dema_property,
        check_dema_no_poison,
        check_dema_warmup_nan_preservation
    );

    #[test]
    fn test_dema_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = DemaInput::with_default_candles(&candles);
        let baseline = dema(&input)?.values;

        let mut out = vec![0.0; candles.close.len()];

        
        #[cfg(not(feature = "wasm"))]
        {
            dema_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            dema_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(out.len(), baseline.len());
        for i in 0..out.len() {
            let a = out[i];
            let b = baseline[i];
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "NaN mismatch at index {}", i);
            } else {
                assert!(a == b, "Value mismatch at index {}: {} != {}", i, a, b);
            }
        }
        Ok(())
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = DemaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = DemaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
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

    
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        
        let test_configs = vec![
            
            (2, 5, 1), 
            
            (5, 25, 5), 
            
            (10, 50, 10), 
            
            (1, 3, 1), 
            
            (50, 150, 25), 
            
            (10, 30, 2), 
            
            (10, 30, 10), 
            
            (100, 300, 50), 
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = DemaBatchBuilder::new()
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
                        combo.period.unwrap_or(30)
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
                        combo.period.unwrap_or(30)
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
                        combo.period.unwrap_or(30)
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
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test]
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    fn check_batch_warmup_nan_preservation(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        
        let output = DemaBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10)
            .apply_candles(&c, "close")?;

        
        for (row_idx, combo) in output.combos.iter().enumerate() {
            let period = combo.period.unwrap_or(30);
            let warmup = period - 1;
            let row_start = row_idx * output.cols;

            
            for i in 0..warmup {
                let val = output.values[row_start + i];
                assert!(
                    val.is_nan(),
                    "[{}] Batch row {} (period={}): Expected NaN at index {}, got {}",
                    test,
                    row_idx,
                    period,
                    i,
                    val
                );
            }

            
            for i in warmup..warmup.min(output.cols).min(warmup + 10) {
                let val = output.values[row_start + i];
                assert!(
                    !val.is_nan(),
                    "[{}] Batch row {} (period={}): Expected non-NaN at index {}, got NaN",
                    test,
                    row_idx,
                    period,
                    i
                );
            }
        }
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
    gen_batch_tests!(check_batch_warmup_nan_preservation);
}

#[cfg(feature = "python")]
#[pyfunction(name = "dema")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn dema_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = DemaParams {
        period: Some(period),
    };
    let dema_in = DemaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| dema_with_kernel(&dema_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "DemaStream")]
pub struct DemaStreamPy {
    stream: DemaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DemaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = DemaParams {
            period: Some(period),
        };
        let stream =
            DemaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(DemaStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "dema_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn dema_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;
    use std::mem::ManuallyDrop;

    let slice_in = data.as_slice()?;
    let sweep = DemaBatchRange {
        period: period_range,
    };
    let kern = validate_kernel(kernel, true)?;

    
    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(PyValueError::new_err("invalid period range: empty expansion"));
    }
    let rows = combos.len();
    let cols = slice_in.len();

    
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    
    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    
    let simd = match match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    } {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Scalar,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    
    let combos = py
        .allow_threads(|| dema_batch_inner_into(slice_in, &sweep, simd, true, out))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    let arr = values.into_pyarray(py).reshape((rows, cols))?;

    let dict = PyDict::new(py);
    dict.set_item("values", arr)?;
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
#[pyfunction(name = "dema_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn dema_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32DemaPy> {
    use crate::cuda::cuda_available;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = DemaBatchRange {
        period: period_range,
    };

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaDema::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.ctx();
        let dev_id = cuda.device_id();
        let arr = cuda
            .dema_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((arr, ctx, dev_id))
    })?;

    Ok(DeviceArrayF32DemaPy::new(inner, ctx, dev_id))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "dema_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn dema_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32DemaPy> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    if period == 0 {
        return Err(PyValueError::new_err("period must be positive"));
    }
    let flat = data_tm_f32.as_slice()?;
    let shape = data_tm_f32.shape();
    let series_len = shape[0];
    let num_series = shape[1];
    let params = DemaParams {
        period: Some(period),
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaDema::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.ctx();
        let dev_id = cuda.device_id();
        let arr = cuda
            .dema_many_series_one_param_time_major_dev(flat, num_series, series_len, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((arr, ctx, dev_id))
    })?;
    Ok(DeviceArrayF32DemaPy::new(inner, ctx, dev_id))
}






































































#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dema_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = DemaParams {
        period: Some(period),
    };
    let input = DemaInput::from_slice(data, params);

    
    let mut output = vec![0.0; data.len()];

    
    dema_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DemaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DemaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DemaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = dema_batch)]
pub fn dema_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: DemaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = DemaBatchRange {
        period: config.period_range,
    };

    let output = dema_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = DemaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize output: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(since = "1.0.0", note = "Use dema_batch instead")]
pub fn dema_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = DemaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let metadata: Vec<f64> = combos
        .iter()
        .map(|combo| combo.period.unwrap() as f64)
        .collect();

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dema_alloc(len: usize) -> *mut f64 {
    
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); 
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dema_free(ptr: *mut f64, len: usize) {
    
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dema_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to dema_into"));
    }

    unsafe {
        
        let data = std::slice::from_raw_parts(in_ptr, len);

        
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        
        let params = DemaParams {
            period: Some(period),
        };
        let input = DemaInput::from_slice(data, params);

        
        if in_ptr == out_ptr {
            
            let mut temp = vec![0.0; len];
            dema_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            dema_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dema_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to dema_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = DemaBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        
        dema_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
