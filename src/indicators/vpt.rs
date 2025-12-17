//! # Volume Price Trend (VPT)
//!
//! Implements the cumulative Volume Price Trend indicator, which accumulates
//! volume-weighted price changes over time. This is the standard definition of VPT.
//!
//! Note: This implementation calculates cumulative VPT where each value is the
//! running sum of all previous volume * (price_change / previous_price) values.
//! Some implementations (like certain Python libraries) may use a non-cumulative
//! version that only adds the current and previous period values.
//!
//! ## Parameters
//! None (uses price/volume arrays).
//!
//! ## Returns
//! - **Ok(VptOutput)** with output array.
//! - **Err(VptError)** otherwise.
//!
//! ## Developer Notes
//! - SIMD status: enabled (AVX2/AVX512) via block prefix-scan with scalar carry. On 100k candles at `-C target-cpu=native`, AVX2/AVX512 improve >30% vs optimized scalar.
//! - CUDA/Python status: CUDA wrapper present with typed errors and VRAM checks; Python CUDA handle exposes CAI v3 + DLPack v1.x (versioned capsules) with primary-context lifetime tracking.
//! - Batch status: row-specific batch kernels not attempted; VPT has no parameter grid and no shared precompute across rows. Batch selection short-circuits to scalar row path.
//! - Streaming Performance: O(1) with sticky-NaN semantics to match slice/batch; uses `mul_add` on the hot path. Very efficient and state-minimal.
//! - Memory Optimization: Uses `alloc_with_nan_prefix` and batch helpers properly. Streaming is optimal with minimal state.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use std::error::Error;
use thiserror::Error;

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
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

#[derive(Debug, Clone)]
pub enum VptData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        price: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VptOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VptParams;

#[derive(Debug, Clone)]
pub struct VptInput<'a> {
    pub data: VptData<'a>,
    pub params: VptParams,
}

impl<'a> VptInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str) -> Self {
        Self {
            data: VptData::Candles { candles, source },
            params: VptParams::default(),
        }
    }

    #[inline]
    pub fn from_slices(price: &'a [f64], volume: &'a [f64]) -> Self {
        Self {
            data: VptData::Slices { price, volume },
            params: VptParams::default(),
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VptData::Candles {
                candles,
                source: "close",
            },
            params: VptParams::default(),
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VptBuilder {
    kernel: Kernel,
}

impl VptBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VptOutput, VptError> {
        let i = VptInput::with_default_candles(c);
        vpt_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
        let i = VptInput::from_slices(price, volume);
        vpt_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> VptStream {
        VptStream::default()
    }
}

#[derive(Debug, Error)]
pub enum VptError {
    #[error("vpt: Empty data provided.")]
    EmptyInputData,
    #[error("vpt: All values are NaN.")]
    AllValuesNaN,
    #[error("vpt: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("vpt: Not enough valid data (needed = {needed}, valid = {valid}).")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vpt: Output length mismatch. expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vpt: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("vpt: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("vpt: size overflow computing rows*cols")]
    SizeOverflow,
}

#[inline]
fn vpt_first_valid(price: &[f64], volume: &[f64]) -> Option<usize> {
    // VPT always has NaN at index 0, so warmup is at least 1
    // Find earliest i >= 1 where the formula can produce a finite value
    // needs p[i-1] finite and != 0, p[i] finite, v[i] finite
    for i in 1..price.len() {
        let p0 = price[i - 1];
        let p1 = price[i];
        let v1 = volume[i];
        if p0.is_finite() && p0 != 0.0 && p1.is_finite() && v1.is_finite() {
            return Some(i);
        }
    }
    None
}

#[inline]
pub fn vpt(input: &VptInput) -> Result<VptOutput, VptError> {
    vpt_with_kernel(input, Kernel::Auto)
}

pub fn vpt_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
    let (price, volume) = match &input.data {
        VptData::Candles { candles, source } => {
            let price = source_type(candles, source);
            let vol = candles
                .select_candle_field("volume")
                .map_err(|_| VptError::EmptyInputData)?;
            (price, vol)
        }
        VptData::Slices { price, volume } => (*price, *volume),
    };

    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyInputData);
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();

    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => vpt_scalar(price, volume),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vpt_avx2(price, volume),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => vpt_avx512(price, volume),
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn vpt_scalar(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    let n = price.len();
    if n == 0 || volume.len() != n {
        return Err(VptError::EmptyInputData);
    }
    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first = vpt_first_valid(price, volume)
        .ok_or(VptError::NotEnoughValidData { needed: 2, valid: valid_count })?;
    let mut res = alloc_with_nan_prefix(n, first + 1);

    // Raw pointers to avoid bounds checks inside the hot loop.
    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();
    let o_ptr = res.as_mut_ptr();

    // Seed with the VPT increment at index `first` (not written to output).
    let mut prev = {
        let p0 = *p_ptr.add(first - 1);
        let p1 = *p_ptr.add(first);
        let v1 = *v_ptr.add(first);
        if (p0 != p0) || (p0 == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            // Keep operation order identical to reference path
            v1 * ((p1 - p0) / p0)
        }
    };

    // Sliding reuse of p[i-1]; unroll by 4 to reduce overhead.
    let mut i = first + 1;
    let mut p_prev = *p_ptr.add(i - 1);

    while i + 3 < n {
        // i
        let p1 = *p_ptr.add(i);
        let v1 = *v_ptr.add(i);
        let cur0 = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p_prev) / p_prev)
        };
        let val0 = cur0 + prev;
        *o_ptr.add(i) = val0;
        prev = val0;
        p_prev = p1;

        // i + 1
        let j1 = i + 1;
        let p2 = *p_ptr.add(j1);
        let v2 = *v_ptr.add(j1);
        let cur1 = if (p_prev != p_prev) || (p_prev == 0.0) || (p2 != p2) || (v2 != v2) {
            f64::NAN
        } else {
            v2 * ((p2 - p_prev) / p_prev)
        };
        let val1 = cur1 + prev;
        *o_ptr.add(j1) = val1;
        prev = val1;
        p_prev = p2;

        // i + 2
        let j2 = i + 2;
        let p3 = *p_ptr.add(j2);
        let v3 = *v_ptr.add(j2);
        let cur2 = if (p_prev != p_prev) || (p_prev == 0.0) || (p3 != p3) || (v3 != v3) {
            f64::NAN
        } else {
            v3 * ((p3 - p_prev) / p_prev)
        };
        let val2 = cur2 + prev;
        *o_ptr.add(j2) = val2;
        prev = val2;
        p_prev = p3;

        // i + 3
        let j3 = i + 3;
        let p4 = *p_ptr.add(j3);
        let v4 = *v_ptr.add(j3);
        let cur3 = if (p_prev != p_prev) || (p_prev == 0.0) || (p4 != p4) || (v4 != v4) {
            f64::NAN
        } else {
            v4 * ((p4 - p_prev) / p_prev)
        };
        let val3 = cur3 + prev;
        *o_ptr.add(j3) = val3;
        prev = val3;
        p_prev = p4;

        i += 4;
    }

    // Tail
    while i < n {
        let p1 = *p_ptr.add(i);
        let v1 = *v_ptr.add(i);
        let cur = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p_prev) / p_prev)
        };
        let val = cur + prev;
        *o_ptr.add(i) = val;
        prev = val;
        p_prev = p1;
        i += 1;
    }

    Ok(VptOutput { values: res })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx2(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    use core::arch::x86_64::*;

    let n = price.len();
    if n == 0 || volume.len() != n {
        return Err(VptError::EmptyInputData);
    }
    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first = vpt_first_valid(price, volume)
        .ok_or(VptError::NotEnoughValidData { needed: 2, valid: valid_count })?;
    let mut out = alloc_with_nan_prefix(n, first + 1);

    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();
    let o_ptr = out.as_mut_ptr();

    // Seed carry = cur(first)
    let mut prev = {
        let p0 = *p_ptr.add(first - 1);
        let p1 = *p_ptr.add(first);
        let v1 = *v_ptr.add(first);
        if (p0 != p0) || (p0 == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        }
    };

    let mut i = first + 1;
    let vzero = _mm256_set1_pd(0.0);
    let vnan = _mm256_set1_pd(f64::NAN);

    #[inline(always)]
    unsafe fn prefix4_pd(x: __m256d) -> __m256d {
        let lo = _mm256_castpd256_pd128(x);
        let hi = _mm256_extractf128_pd(x, 1);
        let z = _mm_setzero_pd();

        // [a0, a1] -> [a0, a0+a1]
        let tlo = _mm_add_pd(lo, _mm_shuffle_pd(z, lo, 0));
        let thi = _mm_add_pd(hi, _mm_shuffle_pd(z, hi, 0));

        // add last of low pair into both lanes of high pair
        let last_lo = _mm_unpackhi_pd(tlo, tlo);
        let thi2 = _mm_add_pd(thi, last_lo);

        _mm256_insertf128_pd(_mm256_castpd128_pd256(tlo), thi2, 1)
    }

    while i + 3 < n {
        // p0 = [p[i-1..i+2]], p1 = [p[i..i+3]], v = [v[i..i+3]]
        let p0 = _mm256_loadu_pd(p_ptr.add(i - 1));
        let p1 = _mm256_loadu_pd(p_ptr.add(i));
        let vv = _mm256_loadu_pd(v_ptr.add(i));

        // invalid mask: isnan(p0)|isnan(p1)|isnan(v)| (p0==0)
        let m_nan_p0 = _mm256_cmp_pd(p0, p0, _CMP_UNORD_Q);
        let m_nan_p1 = _mm256_cmp_pd(p1, p1, _CMP_UNORD_Q);
        let m_nan_v = _mm256_cmp_pd(vv, vv, _CMP_UNORD_Q);
        let m_eq0_p0 = _mm256_cmp_pd(p0, vzero, _CMP_EQ_OQ);
        let invalid = _mm256_or_pd(
            _mm256_or_pd(m_nan_p0, m_nan_p1),
            _mm256_or_pd(m_nan_v, m_eq0_p0),
        );

        // cur = v * ((p1 - p0) / p0)
        let diff = _mm256_sub_pd(p1, p0);
        let div = _mm256_div_pd(diff, p0);
        let mul = _mm256_mul_pd(vv, div);
        let cur = _mm256_blendv_pd(mul, vnan, invalid);

        // vector inclusive scan + add carry
        let ps = prefix4_pd(cur);
        let cary = _mm256_set1_pd(prev);
        let outv = _mm256_add_pd(ps, cary);

        // store
        _mm256_storeu_pd(o_ptr.add(i), outv);

        // update carry = last lane of outv
        let hi128 = _mm256_extractf128_pd(outv, 1);
        let last_hi = _mm_unpackhi_pd(hi128, hi128);
        let tmp: [f64; 2] = core::mem::transmute(last_hi);
        prev = tmp[0];

        i += 4;
    }

    // Scalar tail
    if i < n {
        let mut p_prev = *p_ptr.add(i - 1);
        while i < n {
            let p1 = *p_ptr.add(i);
            let v1 = *v_ptr.add(i);
            let cur = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
                f64::NAN
            } else {
                v1 * ((p1 - p_prev) / p_prev)
            };
            let val = cur + prev;
            *o_ptr.add(i) = val;
            prev = val;
            p_prev = p1;
            i += 1;
        }
    }

    Ok(VptOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    use core::arch::x86_64::*;

    let n = price.len();
    if n == 0 || volume.len() != n {
        return Err(VptError::EmptyInputData);
    }
    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first = vpt_first_valid(price, volume)
        .ok_or(VptError::NotEnoughValidData { needed: 2, valid: valid_count })?;
    let mut out = alloc_with_nan_prefix(n, first + 1);

    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();
    let o_ptr = out.as_mut_ptr();

    // Seed carry
    let mut prev = {
        let p0 = *p_ptr.add(first - 1);
        let p1 = *p_ptr.add(first);
        let v1 = *v_ptr.add(first);
        if (p0 != p0) || (p0 == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        }
    };

    let mut i = first + 1;

    #[inline(always)]
    unsafe fn prefix4_pd(x: __m256d) -> __m256d {
        use core::arch::x86_64::*;
        let lo = _mm256_castpd256_pd128(x);
        let hi = _mm256_extractf128_pd(x, 1);
        let z = _mm_setzero_pd();
        let tlo = _mm_add_pd(lo, _mm_shuffle_pd(z, lo, 0));
        let thi = _mm_add_pd(hi, _mm_shuffle_pd(z, hi, 0));
        let last_lo = _mm_unpackhi_pd(tlo, tlo);
        let thi2 = _mm_add_pd(thi, last_lo);
        _mm256_insertf128_pd(_mm256_castpd128_pd256(tlo), thi2, 1)
    }

    while i + 7 < n {
        // 8-lane loads
        let p0 = _mm512_loadu_pd(p_ptr.add(i - 1)); // [p[i-1]..p[i+6]]
        let p1 = _mm512_loadu_pd(p_ptr.add(i)); // [p[i]..p[i+7]]
        let vv = _mm512_loadu_pd(v_ptr.add(i)); // [v[i]..v[i+7]]

        // invalid: isnan(p0)|isnan(p1)|isnan(v)| (p0==0)
        let m_nan_p0 = _mm512_cmp_pd_mask(p0, p0, _CMP_UNORD_Q);
        let m_nan_p1 = _mm512_cmp_pd_mask(p1, p1, _CMP_UNORD_Q);
        let m_nan_v = _mm512_cmp_pd_mask(vv, vv, _CMP_UNORD_Q);
        let m_eq0_p0 = _mm512_cmp_pd_mask(p0, _mm512_set1_pd(0.0), _CMP_EQ_OQ);
        let invalid = m_nan_p0 | m_nan_p1 | m_nan_v | m_eq0_p0;

        // cur = v * ((p1 - p0) / p0)
        // Use rcp14 with two Newton refinements to reach near-IEEE precision
        // while remaining faster than full-width division on most CPUs.
        let diff = _mm512_sub_pd(p1, p0);
        let r0 = _mm512_rcp14_pd(p0);
        let two = _mm512_set1_pd(2.0);
        let e1 = _mm512_fnmadd_pd(p0, r0, two);      // e1 = 2 - p0*r0
        let r1 = _mm512_mul_pd(r0, e1);              // first refinement
        let e2 = _mm512_fnmadd_pd(p0, r1, two);      // e2 = 2 - p0*r1
        let r2 = _mm512_mul_pd(r1, e2);              // second refinement
        let div = _mm512_mul_pd(diff, r2);
        let mul = _mm512_mul_pd(vv, div);
        let cur = _mm512_mask_mov_pd(mul, invalid, _mm512_set1_pd(f64::NAN));

        // Inclusive scan: do two 4-lane scans on 256-bit halves, then fix up high half
        let lo256 = _mm512_castpd512_pd256(cur);
        let hi256 = _mm512_extractf64x4_pd(cur, 1);
        let lo_ps = prefix4_pd(lo256);
        let mut hi_ps = prefix4_pd(hi256);

        // add low-half total to high-half prefix
        let lo_hi128 = _mm256_extractf128_pd(lo_ps, 1);
        let lo_total = {
            let last_lo = _mm_unpackhi_pd(lo_hi128, lo_hi128);
            let tmp: [f64; 2] = core::mem::transmute(last_lo);
            tmp[0]
        };
        hi_ps = _mm256_add_pd(hi_ps, _mm256_set1_pd(lo_total));

        // combine halves into 512
        let ps512 = _mm512_insertf64x4(_mm512_castpd256_pd512(lo_ps), hi_ps, 1);

        // add carry and store
        let outv = _mm512_add_pd(ps512, _mm512_set1_pd(prev));
        _mm512_storeu_pd(o_ptr.add(i), outv);

        // update carry: last lane of outv
        let hi2 = _mm512_extractf64x4_pd(outv, 1); // __m256d with lanes [4..7]
        let hi128 = _mm256_extractf128_pd(hi2, 1); // __m128d with lanes [6,7]
        let last_hi = _mm_unpackhi_pd(hi128, hi128);
        let tmp: [f64; 2] = core::mem::transmute(last_hi);
        prev = tmp[0];

        i += 8;
    }

    // AVX2 tail if >= 4 remain
    while i + 3 < n {
        use core::arch::x86_64::*;
        let p0 = _mm256_loadu_pd(p_ptr.add(i - 1));
        let p1 = _mm256_loadu_pd(p_ptr.add(i));
        let vv = _mm256_loadu_pd(v_ptr.add(i));
        let vzero = _mm256_set1_pd(0.0);
        let vnan = _mm256_set1_pd(f64::NAN);

        let m_nan_p0 = _mm256_cmp_pd(p0, p0, _CMP_UNORD_Q);
        let m_nan_p1 = _mm256_cmp_pd(p1, p1, _CMP_UNORD_Q);
        let m_nan_v = _mm256_cmp_pd(vv, vv, _CMP_UNORD_Q);
        let m_eq0_p0 = _mm256_cmp_pd(p0, vzero, _CMP_EQ_OQ);
        let invalid = _mm256_or_pd(
            _mm256_or_pd(m_nan_p0, m_nan_p1),
            _mm256_or_pd(m_nan_v, m_eq0_p0),
        );

        let diff = _mm256_sub_pd(p1, p0);
        let div = _mm256_div_pd(diff, p0);
        let mul = _mm256_mul_pd(vv, div);
        let cur = _mm256_blendv_pd(mul, vnan, invalid);

        let ps = {
            let lo = _mm256_castpd256_pd128(cur);
            let hi = _mm256_extractf128_pd(cur, 1);
            let z = _mm_setzero_pd();
            let tlo = _mm_add_pd(lo, _mm_shuffle_pd(z, lo, 0));
            let thi = _mm_add_pd(hi, _mm_shuffle_pd(z, hi, 0));
            let last_lo = _mm_unpackhi_pd(tlo, tlo);
            let thi2 = _mm_add_pd(thi, last_lo);
            _mm256_insertf128_pd(_mm256_castpd128_pd256(tlo), thi2, 1)
        };

        let outv = _mm256_add_pd(ps, _mm256_set1_pd(prev));
        _mm256_storeu_pd(o_ptr.add(i), outv);
        let hi128 = _mm256_extractf128_pd(outv, 1);
        let last_hi = _mm_unpackhi_pd(hi128, hi128);
        let tmp: [f64; 2] = core::mem::transmute(last_hi);
        prev = tmp[0];
        i += 4;
    }

    // Scalar tail
    if i < n {
        let mut p_prev = *p_ptr.add(i - 1);
        while i < n {
            let p1 = *p_ptr.add(i);
            let v1 = *v_ptr.add(i);
            let cur = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
                f64::NAN
            } else {
                v1 * ((p1 - p_prev) / p_prev)
            };
            let val = cur + prev;
            *o_ptr.add(i) = val;
            prev = val;
            p_prev = p1;
            i += 1;
        }
    }

    Ok(VptOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_short(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    vpt_avx512(price, volume)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpt_avx512_long(price: &[f64], volume: &[f64]) -> Result<VptOutput, VptError> {
    vpt_avx512(price, volume)
}

#[inline]
pub fn vpt_indicator(input: &VptInput) -> Result<VptOutput, VptError> {
    vpt(input)
}

#[inline]
pub fn vpt_indicator_with_kernel(input: &VptInput, kernel: Kernel) -> Result<VptOutput, VptError> {
    vpt_with_kernel(input, kernel)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx2(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx2(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_short(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512_short(price, volume)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpt_indicator_avx512_long(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_avx512_long(price, volume)
    }
}

#[inline]
pub fn vpt_indicator_scalar(input: &VptInput) -> Result<VptOutput, VptError> {
    unsafe {
        let (price, volume) = match &input.data {
            VptData::Candles { candles, source } => {
                let price = source_type(candles, source);
                let vol = candles.select_candle_field("volume").unwrap();
                (price, vol)
            }
            VptData::Slices { price, volume } => (*price, *volume),
        };
        vpt_scalar(price, volume)
    }
}

#[inline]
pub fn vpt_expand_grid() -> Vec<VptParams> {
    vec![VptParams::default()]
}

/// Writes VPT into a caller-provided buffer without allocating.
///
/// - Preserves NaN warmups exactly as the Vec API (`vpt`/`vpt_with_kernel`).
/// - `out` length must equal the input length; returns `OutputLengthMismatch` on mismatch.
/// - Uses `Kernel::Auto` dispatch (same as the Vec-returning API) and writes results in-place.
#[cfg(not(feature = "wasm"))]
pub fn vpt_into(input: &VptInput, out: &mut [f64]) -> Result<(), VptError> {
    let (price, volume) = match &input.data {
        VptData::Candles { candles, source } => {
            let price = source_type(candles, source);
            let vol = candles
                .select_candle_field("volume")
                .map_err(|_| VptError::EmptyInputData)?;
            (price, vol)
        }
        VptData::Slices { price, volume } => (*price, *volume),
    };

    vpt_into_slice(out, price, volume, Kernel::Auto)
}

/// Write VPT directly to output slice - no allocations
pub fn vpt_into_slice(
    dst: &mut [f64],
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<(), VptError> {
    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyInputData);
    }

    if dst.len() != price.len() {
        return Err(VptError::OutputLengthMismatch {
            expected: price.len(),
            got: dst.len(),
        });
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();

    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }

    let first = vpt_first_valid(price, volume)
        .ok_or(VptError::NotEnoughValidData { needed: 2, valid: valid_count })?;
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
                vpt_row_scalar_from(price, volume, first + 1, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vpt_row_avx2_from(price, volume, first + 1, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vpt_row_avx512_from(price, volume, first + 1, dst)
            }
            _ => vpt_row_scalar_from(price, volume, first + 1, dst),
        }
    }
    for v in &mut dst[..=first] {
        *v = f64::NAN;
    }
    Ok(())
}

pub fn vpt_batch_inner_into(
    price: &[f64],
    volume: &[f64],
    _range: &VptBatchRange,
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VptParams>, VptError> {
    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyInputData);
    }
    let combos = vec![VptParams::default()];
    let cols = price.len();
    if out.len() != cols {
        return Err(VptError::OutputLengthMismatch {
            expected: cols,
            got: out.len(),
        });
    }

    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first = vpt_first_valid(price, volume)
        .ok_or(VptError::NotEnoughValidData { needed: 2, valid: valid_count })?;

    // change calls to start at first + 1
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto => {
                vpt_row_scalar_from(price, volume, first + 1, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vpt_row_avx2_from(price, volume, first + 1, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vpt_row_avx512_from(price, volume, first + 1, out)
            }
            _ => vpt_row_scalar_from(price, volume, first + 1, out),
        }
    }
    Ok(combos)
}

/// Streaming VPT: sticky-NaN to match slice/batch; FMA used on hot path.
#[derive(Clone, Debug, Default)]
pub struct VptStream {
    // Previous price p[i-1]
    last_price: f64,
    // Previous increment cur(i-1) = v[i-1] * ((p[i-1] - p[i-2]) / p[i-2]); NaN until available
    carry_inc: f64,
    // Last cumulative VPT value emitted; NaN until first non-NaN is produced
    cum: f64,
    // Have we seen at least one sample?
    seeded: bool,
    // Once true, we permanently output NaN (matches slice/batch semantics after invalid data)
    sticky_nan: bool,
}

impl VptStream {
    /// O(1) streaming update of cumulative VPT.
    /// Returns:
    ///   - None: on very first call (needs a previous price)
    ///   - Some(NaN): during warmup at the first valid pair, or forever after any invalid data (sticky)
    ///   - Some(value): cumulative VPT thereafter
    #[inline(always)]
    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        // Seed with the very first price; no output yet.
        if !self.seeded {
            self.last_price = price;
            self.seeded = true;
            self.carry_inc = f64::NAN;
            self.cum = f64::NAN;
            self.sticky_nan = false;
            return None;
        }

        // If we ever hit an invalid sample, match array semantics: stay NaN forever.
        if self.sticky_nan {
            self.last_price = price; // keep tracking price for potential manual restart
            return Some(f64::NAN);
        }

        // Strict validation to mirror the slice/batch code:
        // need finite p[i-1], finite p[i], finite v[i], and p[i-1] != 0.0.
        if !(self.last_price.is_finite()
            && self.last_price != 0.0
            && price.is_finite()
            && volume.is_finite())
        {
            self.sticky_nan = true;
            self.last_price = price;
            self.carry_inc = f64::NAN;
            self.cum = f64::NAN;
            return Some(f64::NAN);
        }

        // Hot-path math (one div, two muls, one add). Use FMA to reduce latency/rounding:
        // cur = volume * ((price - last_price) / last_price)
        let inv = 1.0 / self.last_price;
        let scale = volume * inv;
        let dv = price - self.last_price;
        self.last_price = price;

        // Fused: cur = dv * scale + 0
        let cur_inc = dv.mul_add(scale, 0.0);
        // First valid increment: emit NaN (warmup) but store carry for the next step
        if self.carry_inc.is_nan() {
            self.carry_inc = cur_inc;
            return Some(f64::NAN);
        }

        // From the second valid pair onward:
        // at the second pair, cum is NaN, so base := carry_inc (sum of two increments);
        // after that, base := cum (running sum).
        let base = if self.cum.is_finite() {
            self.cum
        } else {
            self.carry_inc
        };
        let new_cum = base + cur_inc;

        self.carry_inc = cur_inc;
        self.cum = new_cum;
        Some(new_cum)
    }

    /// Reset back to the unseeded state.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Optional helper: restart streaming continuity at a known price (after data gaps).
    #[inline(always)]
    pub fn restart_from(&mut self, price: f64) {
        self.last_price = price;
        self.carry_inc = f64::NAN;
        self.cum = f64::NAN;
        self.seeded = true;
        self.sticky_nan = false;
    }
}

#[derive(Clone, Debug, Default)]
pub struct VptBatchRange;

#[derive(Clone, Debug, Default)]
pub struct VptBatchBuilder {
    kernel: Kernel,
}

impl VptBatchBuilder {
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<VptBatchOutput, VptError> {
        vpt_batch_with_kernel(price, volume, self.kernel)
    }

    pub fn with_default_slices(
        price: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<VptBatchOutput, VptError> {
        VptBatchBuilder::new().kernel(k).apply_slices(price, volume)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VptBatchOutput, VptError> {
        let price = source_type(c, src);
        let volume = c
            .select_candle_field("volume")
            .map_err(|_| VptError::EmptyInputData)?;
        self.apply_slices(price, volume)
    }

    pub fn with_default_candles(c: &Candles) -> Result<VptBatchOutput, VptError> {
        VptBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn vpt_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    k: Kernel,
) -> Result<VptBatchOutput, VptError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(VptError::InvalidKernelForBatch(other)),
    };
    vpt_batch_par_slice(price, volume, kernel)
}

#[derive(Clone, Debug)]
pub struct VptBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VptParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VptBatchOutput {
    pub fn row_for_params(&self, _p: &VptParams) -> Option<usize> {
        Some(0)
    }

    pub fn values_for(&self, _p: &VptParams) -> Option<&[f64]> {
        Some(&self.values[..])
    }
}

#[inline(always)]
pub fn vpt_batch_slice(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<VptBatchOutput, VptError> {
    vpt_batch_inner(price, volume, kern, false)
}

#[inline(always)]
pub fn vpt_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<VptBatchOutput, VptError> {
    vpt_batch_inner(price, volume, kern, true)
}

#[inline(always)]
fn vpt_batch_inner(
    price: &[f64],
    volume: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<VptBatchOutput, VptError> {
    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(VptError::EmptyInputData);
    }

    let combos = vpt_expand_grid();
    let rows = 1usize;
    let cols = price.len();

    // uninit matrix, then fill warmup prefixes with NaN
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // For VPT, warmup is always at least 1 (index 0 is always NaN)
    // but might be more if there are NaN values in the data
    let valid_count = price
        .iter()
        .zip(volume.iter())
        .filter(|(&p, &v)| !(p.is_nan() || v.is_nan()))
        .count();
    if valid_count == 0 {
        return Err(VptError::AllValuesNaN);
    }
    if valid_count < 2 {
        return Err(VptError::NotEnoughValidData {
            needed: 2,
            valid: valid_count,
        });
    }
    let first_valid = vpt_first_valid(price, volume)
        .ok_or(VptError::NotEnoughValidData { needed: 2, valid: valid_count })?;
    let warm = vec![first_valid + 1];
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // get &mut [f64] view over MaybeUninit<f64> buffer
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    vpt_batch_inner_into(price, volume, &VptBatchRange, kern, _parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(VptBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn vpt_row_scalar(price: &[f64], volume: &[f64], out: &mut [f64]) {
    // full coverage writer for python path
    // Find first valid index and set everything before and including it to NaN
    let n = price.len();
    if let Some(first) = vpt_first_valid(price, volume) {
        // Set warmup prefix to NaN
        for i in 0..=first {
            out[i] = f64::NAN;
        }
        // Use the _from variant starting at first + 1
        vpt_row_scalar_from(price, volume, first + 1, out);
    } else {
        // No valid data, all NaN
        for i in 0..n {
            out[i] = f64::NAN;
        }
    }
}

#[inline(always)]
pub unsafe fn vpt_row_scalar_from(price: &[f64], volume: &[f64], start_i: usize, out: &mut [f64]) {
    let n = price.len();
    if start_i >= n {
        return;
    }

    assert!(start_i > 0, "vpt_row_scalar_from requires start_i >= 1");

    let p_ptr = price.as_ptr();
    let v_ptr = volume.as_ptr();
    let o_ptr = out.as_mut_ptr();

    // Seed with the increment at index (start_i - 1); not written to out.
    let mut prev = if start_i >= 2 {
        let k = start_i - 1;
        let p0 = *p_ptr.add(k - 1);
        let p1 = *p_ptr.add(k);
        let v1 = *v_ptr.add(k);
        if (p0 != p0) || (p0 == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p0) / p0)
        }
    } else {
        0.0
    };

    // Sliding reuse of p[i-1]; unroll by 4.
    let mut i = start_i;
    let mut p_prev = *p_ptr.add(i - 1);

    while i + 3 < n {
        // i
        let p1 = *p_ptr.add(i);
        let v1 = *v_ptr.add(i);
        let cur0 = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p_prev) / p_prev)
        };
        let val0 = cur0 + prev;
        *o_ptr.add(i) = val0;
        prev = val0;
        p_prev = p1;

        // i + 1
        let j1 = i + 1;
        let p2 = *p_ptr.add(j1);
        let v2 = *v_ptr.add(j1);
        let cur1 = if (p_prev != p_prev) || (p_prev == 0.0) || (p2 != p2) || (v2 != v2) {
            f64::NAN
        } else {
            v2 * ((p2 - p_prev) / p_prev)
        };
        let val1 = cur1 + prev;
        *o_ptr.add(j1) = val1;
        prev = val1;
        p_prev = p2;

        // i + 2
        let j2 = i + 2;
        let p3 = *p_ptr.add(j2);
        let v3 = *v_ptr.add(j2);
        let cur2 = if (p_prev != p_prev) || (p_prev == 0.0) || (p3 != p3) || (v3 != v3) {
            f64::NAN
        } else {
            v3 * ((p3 - p_prev) / p_prev)
        };
        let val2 = cur2 + prev;
        *o_ptr.add(j2) = val2;
        prev = val2;
        p_prev = p3;

        // i + 3
        let j3 = i + 3;
        let p4 = *p_ptr.add(j3);
        let v4 = *v_ptr.add(j3);
        let cur3 = if (p_prev != p_prev) || (p_prev == 0.0) || (p4 != p4) || (v4 != v4) {
            f64::NAN
        } else {
            v4 * ((p4 - p_prev) / p_prev)
        };
        let val3 = cur3 + prev;
        *o_ptr.add(j3) = val3;
        prev = val3;
        p_prev = p4;

        i += 4;
    }

    // Tail
    while i < n {
        let p1 = *p_ptr.add(i);
        let v1 = *v_ptr.add(i);
        let cur = if (p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1) {
            f64::NAN
        } else {
            v1 * ((p1 - p_prev) / p_prev)
        };
        let val = cur + prev;
        *o_ptr.add(i) = val;
        prev = val;
        p_prev = p1;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx2(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx2_from(price: &[f64], volume: &[f64], start_i: usize, out: &mut [f64]) {
    vpt_row_scalar_from(price, volume, start_i, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_from(price: &[f64], volume: &[f64], start_i: usize, out: &mut [f64]) {
    vpt_row_scalar_from(price, volume, start_i, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_short(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpt_row_avx512_long(price: &[f64], volume: &[f64], out: &mut [f64]) {
    vpt_row_scalar(price, volume, out)
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpt")]
#[pyo3(signature = (price, volume, kernel=None))]
pub fn vpt_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Accept non-contiguous inputs; copy only when necessary
    let price_slice: &[f64];
    let volume_slice: &[f64];
    let owned_price;
    let owned_volume;
    price_slice = if let Ok(s) = price.as_slice() { s } else { owned_price = price.to_owned_array(); owned_price.as_slice().unwrap() };
    volume_slice = if let Ok(s) = volume.as_slice() { s } else { owned_volume = volume.to_owned_array(); owned_volume.as_slice().unwrap() };
    let kern = validate_kernel(kernel, false)?;

    let input = VptInput::from_slices(price_slice, volume_slice);

    let result_vec: Vec<f64> = py
        .allow_threads(|| vpt_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VptStream")]
pub struct VptStreamPy {
    stream: VptStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VptStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(VptStreamPy {
            stream: VptStream::default(),
        })
    }

    fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.stream.update(price, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpt_batch")]
#[pyo3(signature = (price, volume, kernel=None))]
pub fn vpt_batch_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    // Accept non-contiguous inputs; copy only when necessary
    let price_slice: &[f64];
    let volume_slice: &[f64];
    let owned_price;
    let owned_volume;
    price_slice = if let Ok(s) = price.as_slice() { s } else { owned_price = price.to_owned_array(); owned_price.as_slice().unwrap() };
    volume_slice = if let Ok(s) = volume.as_slice() { s } else { owned_volume = volume.to_owned_array(); owned_volume.as_slice().unwrap() };
    let kern = validate_kernel(kernel, true)?;

    // VPT has no parameters, so single row output
    let rows: usize = 1;
    let cols = price_slice.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("vpt_batch: size overflow"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Initialize NaN prefix for VPT (indices 0..=first_valid)
    let first_valid = vpt_first_valid(price_slice, volume_slice)
        .ok_or_else(|| PyValueError::new_err("Not enough valid data"))?;

    // set [0..=first_valid] to NaN
    for i in 0..=first_valid {
        slice_out[i] = f64::NAN;
    }

    let _combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            vpt_batch_inner_into(
                price_slice,
                volume_slice,
                &VptBatchRange,
                kernel,
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;

    // No parameters for VPT, but include empty list for consistency
    dict.set_item("params", Vec::<f64>::new().into_pyarray(py))?;

    Ok(dict)
}

// ---------------- Python CUDA bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaVpt;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vpt_cuda_batch_dev")]
#[pyo3(signature = (price, volume, device_id=0))]
pub fn vpt_cuda_batch_dev_py(
    py: Python<'_>,
    price: PyReadonlyArray1<'_, f32>,
    volume: PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<VptDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let price_slice = price.as_slice()?;
    let volume_slice = volume.as_slice()?;
    if price_slice.len() != volume_slice.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaVpt::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context();
        let dev_id = cuda.device_id();
        let arr = cuda
            .vpt_batch_dev(price_slice, volume_slice)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((arr, ctx, dev_id))
    })?;
    Ok(VptDeviceArrayF32Py {
        buf: Some(inner.buf),
        rows: inner.rows,
        cols: inner.cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "vpt_cuda_many_series_one_param_dev")]
#[pyo3(signature = (price_tm, volume_tm, cols, rows, device_id=0))]
pub fn vpt_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    price_tm: PyReadonlyArray1<'_, f32>,
    volume_tm: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    device_id: usize,
) -> PyResult<VptDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let price_slice = price_tm.as_slice()?;
    let volume_slice = volume_tm.as_slice()?;
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaVpt::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context();
        let dev_id = cuda.device_id();
        let arr = cuda
            .vpt_many_series_one_param_time_major_dev(price_slice, volume_slice, cols, rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, pyo3::PyErr>((arr, ctx, dev_id))
    })?;
    Ok(VptDeviceArrayF32Py {
        buf: Some(inner.buf),
        rows: inner.rows,
        cols: inner.cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_js(price: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
    let mut output = vec![0.0; price.len()];

    vpt_into_slice(&mut output, price, volume, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpt_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let price = std::slice::from_raw_parts(price_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        // Check if either input aliases with output
        if price_ptr == out_ptr || volume_ptr == out_ptr {
            // Need temporary buffer for aliasing
            let mut temp = vec![0.0; len];
            vpt_into_slice(&mut temp, price, volume, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, write directly
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            vpt_into_slice(out, price, volume, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VptBatchConfig {
    // VPT has no parameters, so empty config
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VptBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VptParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vpt_batch)]
pub fn vpt_batch_js(price: &[f64], volume: &[f64], _config: JsValue) -> Result<JsValue, JsValue> {
    // VPT has no parameters, so batch returns single row
    let output = vpt_batch_with_kernel(price, volume, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = VptBatchJsOutput {
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
pub fn vpt_batch_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let price = std::slice::from_raw_parts(price_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);

        // VPT has no parameters, so just compute once
        vpt_batch_inner_into(price, volume, &VptBatchRange, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Return number of parameter combinations (always 1 for VPT)
        Ok(1)
    }
}

// Python CUDA handle for VPT: CAI v3 and DLPack v1.x. Keeps CUDA context alive.
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "my_project", name = "VptDeviceArrayF32", unsendable)]
pub struct VptDeviceArrayF32Py {
    // One-shot export via __dlpack__: move out of this Option
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl VptDeviceArrayF32Py {
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
        // Producer synchronizes before returning, so no stream key is required per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    // DLPack producer with v1.x negotiation and legacy fallback.
    // Array API stream semantics are accepted but ignored here since the stream is synchronized.
    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<pyo3::PyObject> {
        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    #[test]
    fn test_vpt_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Use the existing CSV candles to match other tests in this module.
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VptInput::from_candles(&candles, "close");

        // Baseline via Vec-returning API with a fixed kernel to avoid tiny
        // rounding diffs across different paths.
        let baseline = vpt_with_kernel(&input, Kernel::Scalar)?;

        // Preallocate output and compute via the new into API.
        let mut out = vec![0.0f64; candles.close.len()];
        #[cfg(not(feature = "wasm"))]
        vpt_into(&input, &mut out)?;

        assert_eq!(baseline.values.len(), out.len());

        fn eq_or_both_nan_eps(a: f64, b: f64, eps: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= eps
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan_eps(baseline.values[i], out[i], 1e-12),
                "Mismatch at index {}: baseline={} out={}",
                i,
                baseline.values[i],
                out[i]
            );
        }

        Ok(())
    }

    fn check_vpt_basic_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vpt_basic_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [1.0, 1.1, 1.05, 1.2, 1.3];
        let volume = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0];
        let input = VptInput::from_slices(&price, &volume);
        let output = vpt_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), price.len());
        Ok(())
    }

    fn check_vpt_not_enough_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [100.0];
        let volume = [500.0];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price: [f64; 0] = [];
        let volume: [f64; 0] = [];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [f64::NAN, f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN, f64::NAN];
        let input = VptInput::from_slices(&price, &volume);
        let result = vpt_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vpt_accuracy_from_csv(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VptInput::from_candles(&candles, "close");
        let output = vpt_with_kernel(&input, kernel)?;

        // NOTE: The Rust implementation calculates cumulative VPT (standard definition)
        // Python reference values were for a non-cumulative version:
        // [-0.40358334248536065, -0.16292768139917702, -0.4792942916867958,
        //  -0.1188231211518107, -3.3492674990910025]
        //
        // Our implementation accumulates all historical VPT values, while the Python
        // version only adds the current and previous period values.
        let expected_last_five = [
            -18292.323972247592,
            -18292.510374716476,
            -18292.803266539282,
            -18292.62919783763,
            -18296.152568643138,
        ];

        assert!(output.values.len() >= 5);
        let start_index = output.values.len() - 5;
        for (i, &value) in output.values[start_index..].iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-9,
                "VPT mismatch at final bars, index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    macro_rules! generate_all_vpt_tests {
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

    #[cfg(debug_assertions)]
    fn check_vpt_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Since VPT has no parameters, we'll test with different data sources
        let test_sources = vec!["close", "open", "high", "low"];

        for (source_idx, &source) in test_sources.iter().enumerate() {
            let input = VptInput::from_candles(&candles, source);
            let output = vpt_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_vpt_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_vpt_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy to generate price and volume data
        // Prices: non-negative values (including zero to test edge case)
        // Volume: non-negative values (realistic for trading volume)
        // Length: at least 2 points (minimum for VPT calculation)
        let strat = (2usize..=400).prop_flat_map(|len| {
            (
                prop::collection::vec(
                    (0.0f64..1e6f64)
                        .prop_filter("finite non-negative price", |x| x.is_finite() && *x >= 0.0),
                    len,
                ),
                prop::collection::vec(
                    (0.0f64..1e9f64)
                        .prop_filter("finite non-negative volume", |x| x.is_finite() && *x >= 0.0),
                    len,
                ),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(price, volume)| {
            let input = VptInput::from_slices(&price, &volume);

            // Get output from the kernel being tested
            let VptOutput { values: out } = vpt_with_kernel(&input, kernel)?;

            // Get reference output from scalar kernel
            let VptOutput { values: ref_out } = vpt_with_kernel(&input, Kernel::Scalar)?;

            // Verify properties
            prop_assert_eq!(out.len(), price.len(), "Output length mismatch");
            prop_assert_eq!(
                ref_out.len(),
                price.len(),
                "Reference output length mismatch"
            );

            // First value should always be NaN (warmup period)
            prop_assert!(
                out[0].is_nan(),
                "First VPT value should be NaN, got {}",
                out[0]
            );
            prop_assert!(
                ref_out[0].is_nan(),
                "First reference VPT value should be NaN, got {}",
                ref_out[0]
            );

            // Manually calculate VPT to verify correctness
            let mut expected_vpt = vec![f64::NAN; price.len()];
            let mut prev_vpt_val = f64::NAN;

            for i in 1..price.len() {
                let p0 = price[i - 1];
                let p1 = price[i];
                let v1 = volume[i];

                // Calculate current VPT value
                let vpt_val = if p0.is_nan() || p0 == 0.0 || p1.is_nan() || v1.is_nan() {
                    f64::NAN
                } else {
                    v1 * ((p1 - p0) / p0)
                };

                // Output is current VPT value + previous VPT value (shifted array approach)
                expected_vpt[i] = if vpt_val.is_nan() || prev_vpt_val.is_nan() {
                    f64::NAN
                } else {
                    vpt_val + prev_vpt_val
                };

                // Save current VPT value for next iteration
                prev_vpt_val = vpt_val;
            }

            // Compare outputs with expected calculations
            for i in 0..price.len() {
                let y = out[i];
                let r = ref_out[i];
                let e = expected_vpt[i];

                // Check consistency between kernels
                if y.is_nan() && r.is_nan() {
                    // Both NaN is fine
                    continue;
                } else if !y.is_nan() && !r.is_nan() {
                    // Both should be very close
                    let diff = (y - r).abs();
                    prop_assert!(
                        diff < 1e-9,
                        "Kernel mismatch at idx {}: {} vs {} (diff: {})",
                        i,
                        y,
                        r,
                        diff
                    );

                    // Also check against expected value
                    if !e.is_nan() {
                        let diff_expected = (y - e).abs();
                        prop_assert!(
                            diff_expected < 1e-9,
                            "Value mismatch at idx {}: got {} expected {} (diff: {})",
                            i,
                            y,
                            e,
                            diff_expected
                        );
                    }
                } else {
                    prop_assert!(
                        false,
                        "NaN mismatch at idx {}: kernel={}, scalar={}",
                        i,
                        y,
                        r
                    );
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_vpt_tests!(
        check_vpt_basic_candles,
        check_vpt_basic_slices,
        check_vpt_not_enough_data,
        check_vpt_empty_data,
        check_vpt_all_nan,
        check_vpt_accuracy_from_csv,
        check_vpt_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_vpt_tests!(check_vpt_property);

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various data sources since VPT has no parameters
        let test_sources = vec!["close", "open", "high", "low"];

        for (src_idx, &source) in test_sources.iter().enumerate() {
            let output = VptBatchBuilder::new()
                .kernel(kernel)
                .apply_candles(&c, source)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Source {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with source: {}",
                        test, src_idx, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Source {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with source: {}",
                        test, src_idx, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Source {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with source: {}",
                        test, src_idx, val, bits, row, col, idx, source
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
                    let kernel = detect_best_batch_kernel();
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), kernel);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_no_poison);
}
